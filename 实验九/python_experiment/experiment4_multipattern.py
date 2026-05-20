"""Experiment 4: Multiple patterns, multiple output neurons.

Uses the MATLAB approach: generate one T-block spike train and re-run it nRun times
with shifted spike times. This is more memory-efficient than generating all runs at once.
"""

import numpy as np
import pickle
import os
from spike_train import generate_spike_train
from simulation import Simulation
from analysis import compute_latencies
from plotting import plot_latency_matrix
from utils import get_params_multi_pattern, ensure_figure_dir, FIGURE_DIR


def run_experiment4():
    print("=" * 60)
    print("Experiment 4: Multiple patterns, multiple competing neurons")
    print("=" * 60)

    ensure_figure_dir()

    # Parameters: multi-pattern with lateral inhibition
    params = get_params_multi_pattern(inhib_strength=0.25, random_state=42)

    # Override: generate spike train for ONE T-block only
    params_one = get_params_multi_pattern(inhib_strength=0.25, random_state=42)
    params_one.nRun = 1  # Generate only one block

    print(f"\nParameters:")
    print(f"  nAfferent={params_one.nAfferent}, nPattern={params_one.nPattern}")
    print(f"  nNeuron={params_one.nNeuron}, inhibStrength={params_one.inhibStrength}")
    print(f"  T={params_one.T:.1f}s, nRun={params.nRun} (multi-run mode)")
    print(f"  Total simulation time: {params_one.T * params.nRun:.1f}s")
    print(f"  threshold={params_one.threshold:.1f}")

    # Generate spike train for ONE T-block
    print("\n[1/5] Generating spike train (one T-block)...")
    spikeList, afferentList = generate_spike_train(params_one)

    # Initialize simulation
    print("\n[2/5] Initializing simulation...")
    sim = Simulation(params)
    sim.initialize(spikeList)

    # Run simulation for nRun blocks
    print(f"\n[3/5] Running simulation ({params.nRun} runs)...")
    for run_idx in range(params.nRun):
        shift = run_idx * params_one.T
        shifted_spikes = spikeList + shift
        print(f"  Run {run_idx + 1}/{params.nRun}: {len(shifted_spikes)} spikes, "
              f"shift={shift:.1f}s")
        sim.run(shifted_spikes, afferentList)

    # Report firing counts
    print("\nFiring counts:")
    for i, neuron in enumerate(sim.neurons):
        status = "DEAD" if int(neuron.nFiring) == 0 else "alive"
        print(f"  Neuron {i + 1}: {int(neuron.nFiring)} firings [{status}]")

    # Analyze
    print("\n[4/5] Analyzing...")
    latency, HR, FA, final_latency = compute_latencies(sim.neurons, params)

    print("\nHit rates (rows=patterns, cols=neurons):")
    for pat in range(params.nPattern):
        hr_row = ", ".join(f"{HR[pat, n]:.3f}" for n in range(params.nNeuron))
        print(f"  Pattern {pat + 1}: [{hr_row}]")

    print("\nFalse alarm rates (Hz):")
    for pat in range(params.nPattern):
        fa_row = ", ".join(f"{FA[pat, n]:.2f}" for n in range(params.nNeuron))
        print(f"  Pattern {pat + 1}: [{fa_row}]")

    print("\nFinal latencies (ms):")
    for pat in range(params.nPattern):
        lat_row = ", ".join(
            f"{final_latency[pat, n] * 1000:.1f}" if not np.isnan(final_latency[pat, n])
            else "N/A"
            for n in range(params.nNeuron)
        )
        print(f"  Pattern {pat + 1}: [{lat_row}]")

    # Identify successful (pattern, neuron) pairs
    success_mask = (HR > 0.9) & (FA < 1.0)
    print("\nSuccessful (pattern, neuron) pairs (HR>0.9, FA<1Hz):")
    for pat in range(params.nPattern):
        for n in range(params.nNeuron):
            if success_mask[pat, n]:
                print(f"  Pattern {pat + 1}, Neuron {n + 1}: "
                      f"HR={HR[pat, n]:.3f}, FA={FA[pat, n]:.2f}, "
                      f"lat={final_latency[pat, n] * 1000:.1f}ms")

    # ── Plotting ──
    print("\n[5/5] Plotting...")
    plot_latency_matrix(latency, HR, FA, params,
                        filename=os.path.join(FIGURE_DIR, "exp4_multipattern.png"))

    # Save results
    results = {
        'params': params,
        'latency': latency,
        'HR': HR,
        'FA': FA,
        'final_latency': final_latency,
        'success_mask': success_mask,
        'firing_counts': [int(n.nFiring) for n in sim.neurons],
        'weights': [n.weight.copy() for n in sim.neurons],
    }

    save_path = os.path.join(FIGURE_DIR, "exp4_results.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    print("\nExperiment 4 complete.")
    return results


if __name__ == "__main__":
    run_experiment4()
