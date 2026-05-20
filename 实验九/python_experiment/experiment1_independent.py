"""Experiment 1: Single pattern, 3 independent output neurons (no lateral inhibition).

Tests whether simply adding more output neurons leads to diverse representations.
"""

import numpy as np
import pickle
import os
from parameters import Params
from spike_train import generate_spike_train
from simulation import Simulation
from analysis import compute_latencies
from plotting import plot_latency_subplots
from utils import get_params_single_pattern, ensure_figure_dir, FIGURE_DIR


def run_experiment1():
    print("=" * 60)
    print("Experiment 1: Single pattern, 3 INDEPENDENT neurons (no inhibition)")
    print("=" * 60)

    ensure_figure_dir()

    # Parameters: no lateral inhibition
    params = get_params_single_pattern(inhib_strength=0.0, random_state=42)

    print(f"\nParameters:")
    print(f"  nAfferent={params.nAfferent}, nPattern={params.nPattern}")
    print(f"  nNeuron={params.nNeuron}, inhibStrength={params.inhibStrength}")
    print(f"  T={params.T:.1f}s, threshold={params.threshold:.1f}")

    # Generate spike train
    print("\n[1/4] Generating spike train...")
    spikeList, afferentList = generate_spike_train(params)

    # Initialize simulation
    print("\n[2/4] Initializing simulation...")
    sim = Simulation(params)
    sim.initialize(spikeList)

    # Run simulation
    print("\n[3/4] Running simulation...")
    sim.run(spikeList, afferentList)

    # Report firing counts
    print("\nFiring counts:")
    for i, neuron in enumerate(sim.neurons):
        print(f"  Neuron {i + 1}: {int(neuron.nFiring)} firings")

    # Analyze
    print("\n[4/4] Analyzing and plotting...")
    latency, HR, FA, final_latency = compute_latencies(sim.neurons, params)

    print("\nHit rates and false alarm rates:")
    for i in range(params.nNeuron):
        print(f"  Neuron {i + 1}: HR={HR[0, i]:.3f}, FA={FA[0, i]:.2f} Hz, "
              f"final latency={final_latency[0, i] * 1000 if not np.isnan(final_latency[0, i]) else 'N/A'} ms")

    # Plot
    plot_latency_subplots(latency, params,
                          title_prefix="Part 1: No lateral inhibition",
                          filename=os.path.join(FIGURE_DIR, "exp1_independent.png"),
                          nNeuron=params.nNeuron, nPattern=1,
                          HR=HR, FA=FA)

    # Save results
    results = {
        'params': params,
        'latency': latency,
        'HR': HR,
        'FA': FA,
        'final_latency': final_latency,
        'firing_counts': [int(n.nFiring) for n in sim.neurons],
        'weights': [n.weight.copy() for n in sim.neurons],
    }

    save_path = os.path.join(FIGURE_DIR, "exp1_results.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    print("\nExperiment 1 complete.")
    return results


if __name__ == "__main__":
    run_experiment1()
