| scenario | engine | population_size | num_generations | runs | HV | IGD | Runtime(s) |
|---|---|---|---|---|---|---|---|
| Main (No Surrogate) | deap_nsga2 | 40 | 12 | 2 | 80772469.599 ± 1254165.783 | 296.439 ± 59.038 | 1.255 ± 0.173 |
| Ablation - w/o DynamicElite | deap_nsga2 | 40 | 12 | 2 | 84585415.359 ± 1275875.558 | 168.077 ± 9.313 | 0.583 ± 0.013 |
| Ablation - w/o Prototype | deap_nsga2 | 40 | 12 | 2 | 77782725.272 ± 256581.556 | 223.144 ± 6.337 | 1.170 ± 0.014 |
| Ablation - w/o Surrogate | deap_nsga2 | 40 | 12 | 2 | 86006302.490 ± 848453.474 | 168.457 ± 9.979 | 0.472 ± 0.004 |
