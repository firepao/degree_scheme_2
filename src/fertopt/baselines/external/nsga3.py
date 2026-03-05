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

from ...core.config import AppConfig
from ...core.problem import FertilizationProblem
from ...models.surrogate import SurrogateManager, SurrogateParams
from ...operators.initialization import beta_biased_initialize, save_init_distribution_plot
from ...core.runner import RunArtifacts

class NSGA3Runner:
    """Wrapper for NSGA-III implementation using DEAP."""

    def __init__(self, config: AppConfig, problem: FertilizationProblem) -> None:
        self.config = config
        self.problem = problem
        self.rng = np.random.default_rng(config.seed)

    def run(self, artifact_dir: str | Path) -> RunArtifacts:
        artifact_root = Path(artifact_dir)
        artifact_root.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting NSGA-III run")
        
        try:
            from deap import base, creator, tools
            from tqdm import tqdm
        except ImportError as exc:
            raise ModuleNotFoundError("Required dependencies (deap, tqdm) not found.") from exc

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
        
        # Generate Reference Points for NSGA-III
        # For 3 objectives, p=12 gives 91 points (close to 100)
        ref_points = tools.uniform_reference_points(nobj=3, p=12)
        
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
        individual_cls = getattr(creator, "IndividualNSGA3")
        population = [individual_cls(row.tolist()) for row in init_pop]

        # Initial Evaluation
        init_true = self.problem.evaluate(init_pop)
        surrogate_manager.initialize(init_pop, init_true)
        init_eval = surrogate_manager.predict_objectives(init_pop, self.problem.evaluate)
        for individual, fit in zip(population, init_eval):
            individual.fitness.values = tuple(float(v) for v in fit)

        # Main Evolution Loop
        # Note: NSGA-III selection happens AFTER offspring generation usually, 
        # but DEAP style is often select -> vary -> replace.
        # However, NSGA-III is typically: P_t -> Q_t (offspring) -> R_t = P_t U Q_t -> P_{t+1} (select from R_t)
        # We will follow this standard flow.
        
        for gen_idx in tqdm(range(self.config.num_generations), desc="NSGA-III Evolution", unit="gen"):
            # 1. Generate Offspring
            offspring = []
            # Standard binary tournament selection for mating pool? 
            # Or just random mating? NSGA-III usually uses Tournament.
            # We'll use Tournament Selection for mating pool construction
            mating_pool = tools.selTournament(population, len(population), tournsize=2)
            offspring = [toolbox.clone(ind) for ind in mating_pool]
            
            # Apply Crossover and Mutation
            for i in range(1, len(offspring), 2):
                if self.rng.random() <= 0.9: # Crossover prob
                    toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values

            for i in range(len(offspring)):
                if self.rng.random() <= self.config.mutation.p0: # Mutation prob
                    toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # 2. Evaluate Offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                invalid_np = np.asarray(invalid_ind, dtype=float)
                # Surrogate Prediction
                pred_obj = surrogate_manager.predict_objectives(invalid_np, self.problem.evaluate)
                for ind, fit in zip(invalid_ind, pred_obj):
                    ind.fitness.values = tuple(float(v) for v in fit)

            # 3. Environmental Selection (NSGA-III)
            # Combine P_t and Q_t
            population = toolbox.select(population + offspring, self.config.population_size, ref_points)
            
            # 4. Surrogate Active Learning Update
            pop_np = np.asarray(population, dtype=float)
            updated = surrogate_manager.active_update(
                generation_index=gen_idx + 1,
                x_candidates=pop_np,
                true_eval_fn=self.problem.evaluate,
            )
            if updated:
                refreshed_obj = surrogate_manager.predict_objectives(pop_np, self.problem.evaluate)
                for individual, fit in zip(population, refreshed_obj):
                    individual.fitness.values = tuple(float(v) for v in fit)

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
        fitness_cls_name = "FitnessMinNSGA3"
        individual_cls_name = "IndividualNSGA3"

        if not hasattr(creator, fitness_cls_name):
            creator.create(fitness_cls_name, base.Fitness, weights=tuple([-1.0] * objective_count))
        if not hasattr(creator, individual_cls_name):
            creator.create(individual_cls_name, list, fitness=getattr(creator, fitness_cls_name))

        toolbox = base.Toolbox()

        def evaluate_individual(individual: list[float]) -> tuple[float, ...]:
            values = self.problem.evaluate(np.asarray([individual], dtype=float))[0]
            return tuple(float(v) for v in values)

        toolbox.register("clone", copy.deepcopy)
        # Use Standard SBX and Polynomial Mutation for fair comparison
        toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                         low=self.config.var_lower_bound,
                         up=self.config.var_upper_bound,
                         eta=20.0)
        toolbox.register("mutate", tools.mutPolynomialBounded,
                         low=self.config.var_lower_bound,
                         up=self.config.var_upper_bound,
                         eta=20.0,
                         indpb=1.0 / self.config.dimension)
        toolbox.register("select", tools.selNSGA3) # Key change
        toolbox.register("evaluate", evaluate_individual)
        
        return toolbox

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
                color="#E74C3C", # Distinct color for NSGA-III
            )
            ax.set_xlabel("Objective 1")
            ax.set_ylabel("Objective 2")
            ax.set_zlabel("Objective 3")
            ax.set_title("Pareto Front (NSGA-III)")
        else:
            plt.scatter(
                objective_values[:, 0],
                objective_values[:, 1],
                s=18,
                alpha=0.85,
                color="#E74C3C",
            )
            plt.xlabel("Objective 1")
            plt.ylabel("Objective 2")
            plt.title("Pareto Front (NSGA-III)")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
