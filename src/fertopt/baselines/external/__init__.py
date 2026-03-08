from .nsga3 import NSGA3Runner

# Expose PyMoo Runners
def get_moead_runner(config, problem):
    from .pymoo_runner import PyMooRunner
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.util.ref_dirs import get_reference_directions
    
    def factory():
        n_obj = len(config.objectives)
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
        return MOEAD(
            ref_dirs,
            n_neighbors=15,
            prob_neighbor_mating=0.7,
        )
    return PyMooRunner(config, problem, factory, "MOEA/D")

def get_agemoea_runner(config, problem):
    from .pymoo_runner import PyMooRunner
    from pymoo.algorithms.moo.age import AGEMOEA
    
    def factory():
        return AGEMOEA(pop_size=config.population_size)
    return PyMooRunner(config, problem, factory, "AGE-MOEA")

def get_ctaea_runner(config, problem):
    from .pymoo_runner import PyMooRunner
    from pymoo.algorithms.moo.ctaea import CTAEA
    from pymoo.util.ref_dirs import get_reference_directions
    
    def factory():
        n_obj = len(config.objectives)
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
        return CTAEA(ref_dirs=ref_dirs)
    return PyMooRunner(config, problem, factory, "C-TAEA")

def get_rvea_runner(config, problem):
    from .pymoo_runner import PyMooRunner
    from pymoo.algorithms.moo.rvea import RVEA
    from pymoo.util.ref_dirs import get_reference_directions
    
    def factory():
        n_obj = len(config.objectives)
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
        return RVEA(ref_dirs=ref_dirs)
    return PyMooRunner(config, problem, factory, "RVEA")

def get_smsemoa_runner(config, problem):
    from .pymoo_runner import PyMooRunner
    from pymoo.algorithms.moo.sms import SMSEMOA
    
    def factory():
        return SMSEMOA(pop_size=config.population_size)
    return PyMooRunner(config, problem, factory, "SMS-EMOA")

def get_nsga3_runner(config, problem):
    return NSGA3Runner(config, problem)

__all__ = [
    "NSGA3Runner",
    "get_moead_runner",
    "get_agemoea_runner",
    "get_ctaea_runner",
    "get_rvea_runner",
    "get_smsemoa_runner",
    "get_nsga3_runner",
]
