from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.core.algorithm import Algorithm
from pymoo.optimize import minimize
from pymoo.core.initialization import Initialization
from pymoo.core.population import Population

logger = logging.getLogger(__name__)

from ...core.config import AppConfig
from ...core.problem import FertilizationProblem
from ...models.surrogate import SurrogateManager, SurrogateParams
from ...operators.initialization import beta_biased_initialize, save_init_distribution_plot
from ...core.runner import RunArtifacts

class SurrogatePyMooProblem(Problem):
    def __init__(self, n_var: int, n_obj: int, xl: float, xu: float, surrogate_manager: SurrogateManager, true_eval_fn: Callable):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        self.surrogate_manager = surrogate_manager
        self.true_eval_fn = true_eval_fn

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        if X.ndim == 1:
            X = X[np.newaxis, :]
        # Evaluate using surrogate
        pred_obj = self.surrogate_manager.predict_objectives(X, self.true_eval_fn)
        out["F"] = pred_obj

class ActiveLearningCallback(Callback):
    def __init__(self, surrogate_manager: SurrogateManager, true_eval_fn: Callable):
        super().__init__()
        self.surrogate_manager = surrogate_manager
        self.true_eval_fn = true_eval_fn

    def notify(self, algorithm: Algorithm) -> None:
        gen_idx = algorithm.n_gen
        # Extract current population
        pop = algorithm.pop
        pop_X = pop.get("X")
        
        # Active update
        updated = self.surrogate_manager.active_update(
            generation_index=gen_idx,
            x_candidates=pop_X,
            true_eval_fn=self.true_eval_fn,
        )
        if updated:
            refreshed_obj = self.surrogate_manager.predict_objectives(pop_X, self.true_eval_fn)
            pop.set("F", refreshed_obj)

class PyMooRunner:
    """Generic wrapper for pymoo algorithms with surrogate support."""

    def __init__(self, config: AppConfig, problem: FertilizationProblem, algo_factory: Callable[[], Algorithm], algo_name: str) -> None:
        self.config = config
        self.problem = problem
        self.algo_factory = algo_factory
        self.algo_name = algo_name
        self.rng = np.random.default_rng(config.seed)

    def run(self, artifact_dir: str | Path) -> RunArtifacts:
        artifact_root = Path(artifact_dir)
        artifact_root.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting {self.algo_name} run via PyMoo")

        algorithm = self.algo_factory()

        # Determine correct population size (important for MOEA/D etc. which derive it from ref_dirs)
        alg_pop_size = getattr(algorithm, "pop_size", self.config.population_size)
        if hasattr(algorithm, "ref_dirs") and algorithm.ref_dirs is not None:
            alg_pop_size = len(algorithm.ref_dirs)
        if alg_pop_size is None:
            alg_pop_size = self.config.population_size

        # Initial Population using custom beta initialization
        init_X = beta_biased_initialize(
            pop_size=alg_pop_size,
            dimension=self.config.dimension,
            lower_bound=self.config.var_lower_bound,
            upper_bound=self.config.var_upper_bound,
            strength_k=self.config.beta_strength_k,
            rng=self.rng,
        )
        save_init_distribution_plot(init_X, artifact_root / "init_distribution.png")

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

        # Initial true evaluations to train surrogate
        init_true = self.problem.evaluate(init_X)
        surrogate_manager.initialize(init_X, init_true)

        # PyMoo problem wrapper
        pymoo_problem = SurrogatePyMooProblem(
            n_var=self.config.dimension,
            n_obj=len(self.config.objectives),
            xl=self.config.var_lower_bound,
            xu=self.config.var_upper_bound,
            surrogate_manager=surrogate_manager,
            true_eval_fn=self.problem.evaluate,
        )

        # Custom initialization for pymoo
        init_pop = Population.new("X", init_X)
        # Evaluate initial using surrogate just to be consistent, though we have true
        init_eval = surrogate_manager.predict_objectives(init_X, self.problem.evaluate)
        init_pop.set("F", init_eval)
        
        algorithm.initialization = Initialization(init_pop)

        callback = ActiveLearningCallback(
            surrogate_manager=surrogate_manager,
            true_eval_fn=self.problem.evaluate,
        )

        res = minimize(
            pymoo_problem,
            algorithm,
            termination=("n_gen", self.config.num_generations),
            seed=self.config.seed,
            callback=callback,
            verbose=True,
        )

        # Save Results
        if res.pop is not None:
            final_pop = res.pop.get("X")
            final_obj = res.pop.get("F")
        else:
            final_pop = np.empty((0, self.config.dimension))
            final_obj = np.empty((0, len(self.config.objectives)))

        final_population_path = artifact_root / "final_population.csv"
        final_objective_path = artifact_root / "final_objectives.csv"
        pareto_front_path = artifact_root / "pareto_front.png"
        
        np.savetxt(final_population_path, final_pop, delimiter=",", fmt="%.6f")
        np.savetxt(final_objective_path, final_obj, delimiter=",", fmt="%.6f")
        self._save_pareto_front_plot(final_obj, pareto_front_path, self.algo_name)

        return RunArtifacts(
            init_population_path=artifact_root / "init_distribution.png",
            final_population_path=final_population_path,
            final_objective_path=final_objective_path,
            pareto_front_path=pareto_front_path,
        )

    def _save_pareto_front_plot(self, objective_values: np.ndarray, out_path: Path, algo_name: str) -> None:
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
                color="#3498DB",
            )
            ax.set_xlabel("Objective 1")
            ax.set_ylabel("Objective 2")
            ax.set_zlabel("Objective 3")
            ax.set_title(f"Pareto Front ({algo_name})")
        else:
            plt.scatter(
                objective_values[:, 0],
                objective_values[:, 1],
                s=18,
                alpha=0.85,
                color="#3498DB",
            )
            plt.xlabel("Objective 1")
            plt.ylabel("Objective 2")
            plt.title(f"Pareto Front ({algo_name})")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
