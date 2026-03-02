from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass(slots=True)
class MutationConfig:
    p0: float
    beta: float
    gamma: float
    delta: float
    sigma_base: float = 2.0
    alpha_knowledge: float = 0.6
    rho_np: float = 0.4
    rho_nk: float = -0.2
    rho_pk: float = 0.15
    p_max: float = 0.6
    stage_sensitivity: List[float] | None = None


@dataclass(slots=True)
class CrossoverConfig:
    gamma0: float
    prototype_count: int = 4
    elite_ratio: float = 0.25
    kmeans_iters: int = 15


@dataclass(slots=True)
class SurrogateConfig:
    enabled: bool
    update_interval_g: int
    query_batch_size: int
    target_objectives: List[str]
    model_num_estimators: int
    model_learning_rate: float


@dataclass(slots=True)
class SelectionConfig:
    alpha0: float = 0.5
    beta_decay: float = 3.0
    omega_f: float = 0.6
    omega_x: float = 0.4
    k_neighbors: int = 5


@dataclass(slots=True)
class AppConfig:
    seed: int
    optimizer_engine: str
    population_size: int
    num_generations: int
    num_stages: int
    nutrients: List[str]
    beta_strength_k: float
    var_lower_bound: float
    var_upper_bound: float
    objectives: List[str]
    use_prototype_crossover: bool
    use_coupled_mutation: bool
    use_dynamic_elite_retention: bool
    mutation: MutationConfig
    crossover: CrossoverConfig
    selection: SelectionConfig
    surrogate: SurrogateConfig

    @property
    def dimension(self) -> int:
        return self.num_stages * len(self.nutrients)



def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        seed=raw["seed"],
        optimizer_engine=raw.get("optimizer_engine", "random_search"),
        population_size=raw["population_size"],
        num_generations=raw["num_generations"],
        num_stages=raw["num_stages"],
        nutrients=raw["nutrients"],
        beta_strength_k=raw["beta_strength_k"],
        var_lower_bound=raw["var_lower_bound"],
        var_upper_bound=raw["var_upper_bound"],
        objectives=raw["objectives"],
        use_prototype_crossover=raw.get("use_prototype_crossover", True),
        use_coupled_mutation=raw.get("use_coupled_mutation", True),
        use_dynamic_elite_retention=raw.get("use_dynamic_elite_retention", True),
        mutation=MutationConfig(
            p0=raw["mutation"]["p0"],
            beta=raw["mutation"]["beta"],
            gamma=raw["mutation"]["gamma"],
            delta=raw["mutation"]["delta"],
            sigma_base=raw["mutation"].get("sigma_base", 2.0),
            alpha_knowledge=raw["mutation"].get("alpha_knowledge", 0.6),
            rho_np=raw["mutation"].get("rho_np", 0.4),
            rho_nk=raw["mutation"].get("rho_nk", -0.2),
            rho_pk=raw["mutation"].get("rho_pk", 0.15),
            p_max=raw["mutation"].get("p_max", 0.6),
            stage_sensitivity=raw["mutation"].get("stage_sensitivity"),
        ),
        crossover=CrossoverConfig(
            gamma0=raw["crossover"]["gamma0"],
            prototype_count=raw["crossover"].get("prototype_count", 4),
            elite_ratio=raw["crossover"].get("elite_ratio", 0.25),
            kmeans_iters=raw["crossover"].get("kmeans_iters", 15),
        ),
        selection=SelectionConfig(
            alpha0=raw.get("selection", {}).get("alpha0", 0.5),
            beta_decay=raw.get("selection", {}).get("beta_decay", 3.0),
            omega_f=raw.get("selection", {}).get("omega_f", 0.6),
            omega_x=raw.get("selection", {}).get("omega_x", 0.4),
            k_neighbors=raw.get("selection", {}).get("k_neighbors", 5),
        ),
        surrogate=SurrogateConfig(
            enabled=raw["surrogate"].get("enabled", False),
            update_interval_g=raw["surrogate"]["update_interval_g"],
            query_batch_size=raw["surrogate"]["query_batch_size"],
            target_objectives=raw["surrogate"].get("target_objectives", ["yield", "nitrogen_loss"]),
            model_num_estimators=raw["surrogate"].get("model_num_estimators", 200),
            model_learning_rate=raw["surrogate"].get("model_learning_rate", 0.05),
        ),
    )
