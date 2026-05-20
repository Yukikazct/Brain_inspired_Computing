"""Experiment 3: Lateral inhibition strength analysis.

Sweeps inhibition strength and measures latency differences between successful neurons.
Spike trains are pre-generated and reused across inhibition values for efficiency.
"""

import numpy as np
import pickle
import os
from spike_train import generate_spike_train
from simulation import Simulation
from analysis import compute_latencies
from plotting import plot_inhibition_strength_analysis
from utils import get_params_single_pattern, ensure_figure_dir, FIGURE_DIR


def run_single_simulation(params, spikeList, afferentList, rng_seed_offset):
    """Run one simulation with pre-generated spike train."""
    sim = Simulation(params)
    sim.initialize(spikeList, rng_seed_offset=rng_seed_offset)
    sim.run(spikeList, afferentList)
    latency, HR, FA, final_latency = compute_latencies(sim.neurons, params)
    return final_latency, HR, FA


def run_experiment3():
    print("=" * 60)
    print("Experiment 3: Lateral inhibition strength analysis")
    print("=" * 60)

    ensure_figure_dir()

    # Inhibition values to test (as fraction of threshold)
    inhib_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    n_repeats = 5
    hr_threshold = 0.9
    fa_threshold = 1.0
    T_sim = 75.0  # match Experiments 1 & 2 (per run)

    # ── Pre-generate spike trains for all repetitions ──
    print("\nPre-generating spike trains...")
    spike_trains = []  # list of (spikeList, afferentList)
    for rep in range(n_repeats):
        print(f"  Repetition {rep + 1}/{n_repeats}...", end=" ")
        rs = 100 + rep * 100  # random state for this repetition
        params_base = get_params_single_pattern(inhib_strength=0.0, random_state=rs)
        params_base.T = T_sim  # override simulation time
        sl, al = generate_spike_train(params_base)
        spike_trains.append((sl, al, params_base))
        print(f"{len(sl)} spikes")

    all_mean_diffs = []
    all_std_diffs = []

    for inhib in inhib_values:
        print(f"\n{'─' * 50}")
        print(f"Testing inhibStrength = {inhib}")
        print(f"{'─' * 50}")

        all_diffs_for_this_inhib = []

        for rep in range(n_repeats):
            print(f"  Repetition {rep + 1}/{n_repeats}...", end=" ")

            sl, al, params_base = spike_trains[rep]
            # Create new params with this inhibition strength but same pattern positions
            params = get_params_single_pattern(inhib_strength=inhib,
                                                random_state=params_base.randomState)
            params.T = T_sim
            params.posCopyPaste = params_base.posCopyPaste
            params._posCopyPasteLists = params_base._posCopyPasteLists

            try:
                final_latency, HR, FA = run_single_simulation(
                    params, sl, al, rng_seed_offset=params_base.randomState + 1
                )
            except Exception as e:
                print(f"Error: {e}")
                continue

            # Identify successful neurons
            success_mask = (HR > hr_threshold) & (FA < fa_threshold)
            success_lats = final_latency[0, success_mask[0, :]]
            success_lats = success_lats[~np.isnan(success_lats)]
            success_lats = np.sort(success_lats)

            if len(success_lats) >= 2:
                diffs = np.diff(success_lats)
                all_diffs_for_this_inhib.extend(diffs)
                print(f"OK: {len(success_lats)} neurons, "
                      f"lats={success_lats * 1000} ms, "
                      f"diffs={diffs * 1000} ms")
            else:
                print(f"only {len(success_lats)} successful neurons")

        if len(all_diffs_for_this_inhib) > 0:
            mean_diff = np.mean(all_diffs_for_this_inhib)
            std_diff = np.std(all_diffs_for_this_inhib)
        else:
            mean_diff = 0.0
            std_diff = 0.0

        all_mean_diffs.append(mean_diff)
        all_std_diffs.append(std_diff)
        print(f"  => Mean diff: {mean_diff * 1000:.2f} ms, Std: {std_diff * 1000:.2f} ms")

    # Plot
    plot_inhibition_strength_analysis(
        inhib_values, all_mean_diffs, all_std_diffs, None,
        filename=os.path.join(FIGURE_DIR, "exp3_inhib_analysis.png")
    )

    # Save results
    results = {
        'inhib_values': inhib_values,
        'mean_diffs': all_mean_diffs,
        'std_diffs': all_std_diffs,
    }
    save_path = os.path.join(FIGURE_DIR, "exp3_results.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    print("\nExperiment 3 complete.")
    return results


if __name__ == "__main__":
    run_experiment3()
