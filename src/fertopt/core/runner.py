from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Set matplotlib backend to Agg for non-interactive environments
matplotlib.use("Agg")

from .config import AppConfig
from .problem import FertilizationProblem
from ..models.surrogate import SurrogateManager, SurrogateParams
from ..operators.crossover import (
    build_elite_prototypes, 
    synergistic_balance_crossover,
    sbx_crossover,
    de_crossover,
    pcx_crossover,
    adaptive_sbx_crossover,
)
from ..operators.initialization import beta_biased_initialize, save_init_distribution_plot
from ..operators.mutation import (
    build_synergy_antagonism_matrix,
    coupled_mutation,
    dynamic_mutation_probability,
)
from ..operators.selection import dynamic_elite_select_indices


@dataclass(slots=True)
class RunArtifacts:
    init_population_path: Path
    final_population_path: Path
    final_objective_path: Path
    pareto_front_path: Path


class BaselineRunner:
    """第一版最小闭环。

    说明：当前阶段先以“Beta初始化 + 随机扰动搜索”跑通工程闭环。
    下一阶段会替换为 geatpy NSGA-II 主循环和你的改进算子。
    """

    def __init__(self, config: AppConfig, problem: FertilizationProblem) -> None:
        self.config = config
        self.problem = problem
        self.rng = np.random.default_rng(config.seed)

    def run(self, artifact_dir: str | Path, engine: str | None = None) -> RunArtifacts:
        artifact_root = Path(artifact_dir)
        artifact_root.mkdir(parents=True, exist_ok=True)

        selected_engine = engine or self.config.optimizer_engine

        logger.info(f"Starting run with engine: {selected_engine}")

        if selected_engine == "random_search":
            return self._run_random_search(artifact_root)
        if selected_engine == "deap_nsga2":
            return self._run_deap_nsga2(artifact_root)
        raise ValueError(f"不支持的优化引擎: {selected_engine}")

    def _run_random_search(self, artifact_root: Path) -> RunArtifacts:

        pop = beta_biased_initialize(
            pop_size=self.config.population_size,
            dimension=self.config.dimension,
            lower_bound=self.config.var_lower_bound,
            upper_bound=self.config.var_upper_bound,
            strength_k=self.config.beta_strength_k,
            rng=self.rng,
        )

        save_init_distribution_plot(pop, artifact_root / "init_distribution.png")

        best_pop = pop.copy()
        best_obj = self.problem.evaluate(best_pop)

        for gen_idx in range(self.config.num_generations):
            noise = self.rng.normal(loc=0.0, scale=5.0, size=best_pop.shape)
            candidate = np.clip(
                best_pop + noise,
                self.config.var_lower_bound,
                self.config.var_upper_bound,
            )
            candidate_obj = self.problem.evaluate(candidate)

            # 简化选择: 逐个体按目标和比较（占位逻辑，后续替换非支配排序）
            current_score = np.sum(best_obj, axis=1)
            candidate_score = np.sum(candidate_obj, axis=1)
            improved = candidate_score < current_score
            best_pop[improved] = candidate[improved]
            best_obj[improved] = candidate_obj[improved]

        final_population_path = artifact_root / "final_population.csv"
        final_objective_path = artifact_root / "final_objectives.csv"
        pareto_front_path = artifact_root / "pareto_front.png"

        np.savetxt(final_population_path, best_pop, delimiter=",", fmt="%.6f")
        np.savetxt(final_objective_path, best_obj, delimiter=",", fmt="%.6f")
        self._save_pareto_front_plot(best_obj, pareto_front_path)

        return RunArtifacts(
            init_population_path=artifact_root / "init_distribution.png",
            final_population_path=final_population_path,
            final_objective_path=final_objective_path,
            pareto_front_path=pareto_front_path,
        )

    def _run_deap_nsga2(self, artifact_root: Path) -> RunArtifacts:
        try:
            from deap import base, creator, tools
            from tqdm import tqdm
        except ImportError as exc:
            raise ModuleNotFoundError(
                "Required dependencies (deap, tqdm) not found. Please install them."
            ) from exc

        # Initialize Population
        init_pop = beta_biased_initialize(
            pop_size=self.config.population_size,
            dimension=self.config.dimension,
            lower_bound=self.config.var_lower_bound,
            upper_bound=self.config.var_upper_bound,
            strength_k=self.config.beta_strength_k,
            rng=self.rng,
        )
        save_init_distribution_plot(init_pop, artifact_root / "init_distribution.png")

        # Setup DEAP
        toolbox = self._setup_deap_toolbox(base, creator, tools)

        # Setup Surrogate
        surrogate_manager = SurrogateManager(
            objective_names=list(self.config.objectives),
            params=SurrogateParams(
                enabled=self.config.surrogate.enabled,
                update_interval_g=self.config.surrogate.update_interval_g,
                query_batch_size=self.config.surrogate.query_batch_size,
                target_objectives=list(self.config.surrogate.target_objectives),
                model_num_estimators=self.config.surrogate.model_num_estimators,
                model_learning_rate=self.config.surrogate.model_learning_rate,
                seed=self.config.seed,
                model_type=self.config.surrogate.model_type,
                model_path=self.config.surrogate.model_path or "",
                scaler_path=self.config.surrogate.scaler_path or "",
            ),
        )

        # Create initial population objects
        individual_cls = getattr(creator, "IndividualFertOpt")
        population = [individual_cls(row.tolist()) for row in init_pop]

        # Initial Evaluation
        init_true = self.problem.evaluate(init_pop)
        surrogate_manager.initialize(init_pop, init_true)
        init_eval = surrogate_manager.predict_objectives(init_pop, self.problem.evaluate)
        for individual, fit in zip(population, init_eval):
            individual.fitness.values = tuple(float(v) for v in fit)

        # Main Evolution Loop
        population = toolbox.select(population, len(population))
        
        matrix_m = build_synergy_antagonism_matrix(
            rho_np=self.config.mutation.rho_np,
            rho_nk=self.config.mutation.rho_nk,
            rho_pk=self.config.mutation.rho_pk,
        )
        stage_sensitivity_cfg = self.config.mutation.stage_sensitivity or [1.0] * self.config.num_stages
        if len(stage_sensitivity_cfg) != self.config.num_stages:
            stage_sensitivity_cfg = [1.0] * self.config.num_stages

        for gen_idx in tqdm(range(self.config.num_generations), desc="Evolution", unit="gen"):
            population = self._evolve_generation(
                gen_idx, population, toolbox, surrogate_manager, matrix_m, stage_sensitivity_cfg
            )

        # Save Results
        final_pop = np.asarray(population, dtype=float)
        final_obj = np.asarray([ind.fitness.values for ind in population], dtype=float)

        final_population_path = artifact_root / "final_population.csv"
        final_objective_path = artifact_root / "final_objectives.csv"
        pareto_front_path = artifact_root / "pareto_front.png"
        
        np.savetxt(final_population_path, final_pop, delimiter=",", fmt="%.6f")
        np.savetxt(final_objective_path, final_obj, delimiter=",", fmt="%.6f")
        self._save_pareto_front_plot(final_obj, pareto_front_path)

        return RunArtifacts(
            init_population_path=artifact_root / "init_distribution.png",
            final_population_path=final_population_path,
            final_objective_path=final_objective_path,
            pareto_front_path=pareto_front_path,
        )

    def _setup_deap_toolbox(self, base, creator, tools) -> base.Toolbox:
        objective_count = len(self.problem.objectives)
        fitness_cls_name = "FitnessMinFertOpt"
        individual_cls_name = "IndividualFertOpt"

        if not hasattr(creator, fitness_cls_name):
            creator.create(fitness_cls_name, base.Fitness, weights=tuple([-1.0] * objective_count))
        if not hasattr(creator, individual_cls_name):
            creator.create(individual_cls_name, list, fitness=getattr(creator, fitness_cls_name))

        toolbox = base.Toolbox()

        def evaluate_individual(individual: list[float]) -> tuple[float, ...]:
            values = self.problem.evaluate(np.asarray([individual], dtype=float))[0]
            return tuple(float(v) for v in values)

        toolbox.register("clone", copy.deepcopy)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                         low=self.config.var_lower_bound,
                         up=self.config.var_upper_bound,
                         eta=20.0)
        toolbox.register("mutate", tools.mutPolynomialBounded,
                         low=self.config.var_lower_bound,
                         up=self.config.var_upper_bound,
                         eta=20.0,
                         indpb=1.0 / self.config.dimension)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", evaluate_individual)
        
        return toolbox

    def _evolve_generation(
        self, 
        gen_idx: int, 
        population: List[Any], 
        toolbox: Any, 
        surrogate_manager: SurrogateManager,
        matrix_m: np.ndarray,
        stage_sensitivity_cfg: List[float]
    ) -> List[Any]:
        
        # Calculate Diversity Pressure
        population_array = np.asarray(population, dtype=float)
        population_std = float(np.mean(np.std(population_array, axis=0)))
        normalized_diversity = population_std / max(self.config.var_upper_bound - self.config.var_lower_bound, 1e-8)
        diversity_pressure = float(np.clip(1.0 - normalized_diversity, 0.0, 1.0))

        # Offspring Generation
        offspring = [toolbox.clone(ind) for ind in population]
        # Random shuffle implicit in pairing logic below or just shuffle now
        self.rng.shuffle(offspring) # In-place shuffle for random mating

        # Crossover
        crossover_prob = 0.9
        crossover_method = self.config.crossover.crossover_method.lower()
        
        for i in range(0, len(offspring) - 1, 2): # Step by 2
            ind1, ind2 = offspring[i], offspring[i+1]
            if self.rng.random() <= crossover_prob:
                if self.config.use_prototype_crossover:
                    parent_a = np.asarray(ind1, dtype=float)
                    parent_b = np.asarray(ind2, dtype=float)
                    
                    try:
                        if crossover_method == "sbc":
                            # 原有协同平衡交叉
                            child1_np, child2_np = synergistic_balance_crossover(
                                parent_a=parent_a,
                                parent_b=parent_b,
                                stage_count=self.config.num_stages,
                                alpha=0.5,
                                lower_bound=self.config.var_lower_bound,
                                upper_bound=self.config.var_upper_bound,
                                rng=self.rng,
                            )
                        elif crossover_method == "adaptive_sbx":
                            child1_np, child2_np = adaptive_sbx_crossover(
                                parent_a=parent_a,
                                parent_b=parent_b,
                                current_gen=gen_idx,
                                max_gen=self.config.num_generations,
                                eta_start=15.0,
                                eta_end=30.0,
                                lower_bound=self.config.var_lower_bound,
                                upper_bound=self.config.var_upper_bound,
                                rng=self.rng,
                            )
                        elif crossover_method == "sbx":
                            # SBX 交叉算子
                            child1_np, child2_np = sbx_crossover(
                                parent_a=parent_a,
                                parent_b=parent_b,
                                eta=20.0,
                                lower_bound=self.config.var_lower_bound,
                                upper_bound=self.config.var_upper_bound,
                                rng=self.rng,
                            )
                        elif crossover_method == "de":
                            # DE 差分进化交叉 (需要额外父代，使用自身作为占位)
                            parent_c = parent_a  # 简化处理
                            child1_np, _ = de_crossover(
                                parent_a=parent_a,
                                parent_b=parent_b,
                                parent_c=parent_c,
                                cr=0.9,
                                f=0.5,
                                lower_bound=self.config.var_lower_bound,
                                upper_bound=self.config.var_upper_bound,
                                rng=self.rng,
                            )
                            child2_np, _ = de_crossover(
                                parent_a=parent_b,
                                parent_b=parent_a,
                                parent_c=parent_c,
                                cr=0.9,
                                f=0.5,
                                lower_bound=self.config.var_lower_bound,
                                upper_bound=self.config.var_upper_bound,
                                rng=self.rng,
                            )
                        elif crossover_method == "pcx":
                            # PCX 父代中心交叉
                            child1_np, child2_np = pcx_crossover(
                                parent_a=parent_a,
                                parent_b=parent_b,
                                parent_c=None,
                                eta=0.5,
                                zeta=0.5,
                                lower_bound=self.config.var_lower_bound,
                                upper_bound=self.config.var_upper_bound,
                                rng=self.rng,
                            )
                        else:
                            # 默认使用 SBC
                            child1_np, child2_np = synergistic_balance_crossover(
                                parent_a=parent_a,
                                parent_b=parent_b,
                                stage_count=self.config.num_stages,
                                alpha=0.5,
                                lower_bound=self.config.var_lower_bound,
                                upper_bound=self.config.var_upper_bound,
                                rng=self.rng,
                            )
                        
                        ind1[:] = child1_np.tolist()
                        ind2[:] = child2_np.tolist()
                    except (ValueError, Exception) as e:
                        # 如果新算子失败，回退到默认 SBX
                        logger.warning(f"Crossover failed ({crossover_method}), using default SBX: {e}")
                        toolbox.mate(ind1, ind2)
                else:
                    toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                del ind2.fitness.values

        # Mutation
        mutation_prob = self.config.mutation.p0
        for mutant in offspring:
            if self.rng.random() <= mutation_prob:
                self._apply_adaptive_mutation(
                    mutant, diversity_pressure, matrix_m, stage_sensitivity_cfg, toolbox
                )
                del mutant.fitness.values

        # Evaluation (Surrogate Assisted)
        invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_individuals:
            invalid_x = np.asarray(invalid_individuals, dtype=float)
            invalid_obj = surrogate_manager.predict_objectives(invalid_x, self.problem.evaluate)
            for individual, fit in zip(invalid_individuals, invalid_obj):
                individual.fitness.values = tuple(float(v) for v in fit)

        # Selection
        combined_population = population + offspring
        if self.config.use_dynamic_elite_retention:
            combined_decision = np.asarray(combined_population, dtype=float)
            combined_objective = np.asarray(
                [individual.fitness.values for individual in combined_population],
                dtype=float,
            )
            selected_indices = dynamic_elite_select_indices(
                objective_values=combined_objective,
                decision_values=combined_decision,
                select_size=self.config.population_size,
                generation_index=gen_idx,
                max_generations=self.config.num_generations,
                alpha0=self.config.selection.alpha0,
                beta_decay=self.config.selection.beta_decay,
                omega_f=self.config.selection.omega_f,
                omega_x=self.config.selection.omega_x,
                k_neighbors=self.config.selection.k_neighbors,
            )
            population = [combined_population[idx] for idx in selected_indices]
        else:
            population = toolbox.select(combined_population, self.config.population_size)

        # Active Learning Update
        population_array_after_select = np.asarray(population, dtype=float)
        updated = surrogate_manager.active_update(
            generation_index=gen_idx + 1,
            x_candidates=population_array_after_select,
            true_eval_fn=self.problem.evaluate,
        )
        if updated:
            refreshed_obj = surrogate_manager.predict_objectives(population_array_after_select, self.problem.evaluate)
            for individual, fit in zip(population, refreshed_obj):
                individual.fitness.values = tuple(float(v) for v in fit)
                
        return population

    def _apply_adaptive_mutation(
        self, 
        mutant: Any, 
        diversity_pressure: float, 
        matrix_m: np.ndarray,
        stage_sensitivity_cfg: List[float],
        toolbox: Any
    ) -> None:
        mutant_np = np.asarray(mutant, dtype=float)
        reshaped = mutant_np.reshape(self.config.num_stages, 3)
        deficiency_score = float(
            np.mean(np.abs(reshaped[:, 0] - 0.5 * (reshaped[:, 1] + reshaped[:, 2])))
            / max(self.config.var_upper_bound - self.config.var_lower_bound, 1e-8)
        )
        stage_idx = int(self.rng.integers(0, self.config.num_stages))
        stage_sensitivity = float(stage_sensitivity_cfg[stage_idx])
        
        adaptive_prob = dynamic_mutation_probability(
            p0=self.config.mutation.p0,
            beta=self.config.mutation.beta,
            gamma=self.config.mutation.gamma,
            delta=self.config.mutation.delta,
            deficiency_score=float(np.clip(deficiency_score, 0.0, 1.0)),
            stage_sensitivity=stage_sensitivity,
            diversity_pressure=diversity_pressure,
            p_max=self.config.mutation.p_max,
        )

        if self.rng.random() <= adaptive_prob:
            if self.config.use_coupled_mutation:
                mutated = coupled_mutation(
                    individual=mutant_np,
                    num_stages=self.config.num_stages,
                    matrix_m=matrix_m,
                    sigma_base=self.config.mutation.sigma_base,
                    alpha_knowledge=self.config.mutation.alpha_knowledge,
                    lower_bound=self.config.var_lower_bound,
                    upper_bound=self.config.var_upper_bound,
                    rng=self.rng,
                )
                mutant[:] = mutated.tolist()
            else:
                toolbox.mutate(mutant)
        else:
            toolbox.mutate(mutant)

    def _save_pareto_front_plot(self, objective_values: np.ndarray, out_path: Path) -> None:
        if objective_values.ndim != 2 or objective_values.shape[0] == 0:
            logger.warning("No objective values to plot for Pareto front.")
            return

        out_path.parent.mkdir(parents=True, exist_ok=True)
        num_obj = objective_values.shape[1]

        fig = plt.figure(figsize=(7, 5))
        
        if num_obj >= 3:
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                objective_values[:, 0],
                objective_values[:, 1],
                objective_values[:, 2],
                s=18,
                alpha=0.85,
                color="#2E86C1",
            )
            ax.set_xlabel("Objective 1")
            ax.set_ylabel("Objective 2")
            ax.set_zlabel("Objective 3")
            ax.set_title("Pareto Front (First 3 Objectives)")
        else:
            plt.scatter(
                objective_values[:, 0],
                objective_values[:, 1],
                s=18,
                alpha=0.85,
                color="#2E86C1",
            )
            plt.xlabel("Objective 1")
            plt.ylabel("Objective 2")
            plt.title("Pareto Front")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
