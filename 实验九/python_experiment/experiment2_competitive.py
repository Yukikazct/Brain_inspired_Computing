"""Experiment 2: Single pattern, 3 competitive output neurons (with lateral inhibition).

Tests whether lateral inhibition enables neurons to form diverse temporal representations.
"""

import numpy as np
import pickle
import os
from spike_train import generate_spike_train
from simulation import Simulation
from analysis import compute_latencies
from plotting import plot_latency_subplots
from utils import get_params_single_pattern, ensure_figure_dir, FIGURE_DIR


def run_experiment2():
    print("=" * 60)
    print("Experiment 2: Single pattern, 3 COMPETITIVE neurons (with inhibition)")
    print("=" * 60)

    ensure_figure_dir()

    # Parameters: with lateral inhibition
    params = get_params_single_pattern(inhib_strength=0.25, random_state=42)

    print(f"\nParameters:")
    print(f"  nAfferent={params.nAfferent}, nPattern={params.nPattern}")
    print(f"  nNeuron={params.nNeuron}, inhibStrength={params.inhibStrength}")
    print(f"  T={params.T:.1f}s, threshold={params.threshold:.1f}")

    # Generate spike train (same as experiment 1)
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

    # Compute latency differences between successful neurons
    success_mask = (HR > 0.9) & (FA < 1.0)
    if np.sum(success_mask[0, :]) >= 2:
        success_lats = final_latency[0, success_mask[0, :]]
        success_lats = success_lats[~np.isnan(success_lats)]
        success_lats = np.sort(success_lats)
        diffs = np.diff(success_lats) * 1000  # ms
        print(f"\nAdjacent latency differences (ms): {diffs}")

    # Plot
    plot_latency_subplots(latency, params,
                          title_prefix="Part 2: With lateral inhibition",
                          filename=os.path.join(FIGURE_DIR, "exp2_competitive.png"),
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

    save_path = os.path.join(FIGURE_DIR, "exp2_results.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    print("\nExperiment 2 complete.")
    return results


if __name__ == "__main__":
    run_experiment2()
