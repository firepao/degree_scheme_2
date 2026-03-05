
import numpy as np
import matplotlib.pyplot as plt
from fertopt.operators.crossover import pcx_crossover, sbx_crossover

def test_convergence():
    rng = np.random.default_rng(42)
    
    # Simulate a converged state: Parents are very close
    center_val = 150.0
    p1 = np.array([center_val] * 12)
    p2 = np.array([center_val + 0.1] * 12) # Very close, distance 0.1
    
    print(f"Parents distance: {np.linalg.norm(p1 - p2):.4f}")
    
    # Test SBX
    c1_sbx, c2_sbx = sbx_crossover(p1, p2, eta=20, lower_bound=0, upper_bound=300, rng=rng)
    dist_sbx = np.linalg.norm(c1_sbx - p1)
    print(f"SBX Child distance from P1: {dist_sbx:.4f}")
    
    # Test PCX (Current)
    c1_pcx, c2_pcx = pcx_crossover(p1, p2, eta=0.5, zeta=0.5, lower_bound=0, upper_bound=300, rng=rng)
    dist_pcx = np.linalg.norm(c1_pcx - p1)
    print(f"PCX Child distance from P1: {dist_pcx:.4f}")
    
    # Calculate average noise magnitude per dimension
    noise_mag = np.mean(np.abs(c1_pcx - p1))
    print(f"PCX Average per-dimension perturbation: {noise_mag:.4f}")
    print(f"Expected large perturbation: 300 * 0.1 * 0.5 * avg(|N(0,1)|) ~= 15 * 0.8 = 12.0")

if __name__ == "__main__":
    test_convergence()
