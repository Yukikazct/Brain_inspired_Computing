"""Plotting functions for experiment results."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
plt.rcParams['font.size'] = 8


def plot_latency_subplots(latency, params, title_prefix, filename,
                          nNeuron=None, nPattern=None, HR=None, FA=None):
    """Plot latency scatter plots for each (pattern, neuron) pair.

    Part 1 & 2 format: 1 row, nNeuron columns (single pattern).
    Part 4 format: nPattern rows, nNeuron columns.
    """
    if nNeuron is None:
        nNeuron = len(latency) if isinstance(latency, dict) else latency.shape[1]
    if nPattern is None:
        nPattern = 1

    if nPattern == 1:
        fig, axes = plt.subplots(1, nNeuron, figsize=(4 * nNeuron, 3.5), squeeze=False)
    else:
        fig, axes = plt.subplots(nPattern, nNeuron,
                                 figsize=(3 * nNeuron, 2.5 * nPattern),
                                 squeeze=False)

    cpDuration_ms = params.copyPasteDuration * 1000

    for pat in range(nPattern):
        for neur in range(nNeuron):
            ax = axes[pat, neur]

            if isinstance(latency, dict):
                lat_data = latency.get((pat, neur), np.array([]))
            else:
                lat_data = np.array([])

            if len(lat_data) > 0:
                lat_ms = np.array(lat_data) * 1000  # convert to ms
                ax.plot(lat_ms, '.', markersize=3, color='black')
                ax.set_xlim(0, len(lat_ms))
                ax.set_ylim(0, cpDuration_ms)

            if HR is not None and FA is not None and pat < HR.shape[0] and neur < HR.shape[1]:
                ax.set_title(f'HR={100 * HR[pat, neur]:.0f}% FA={FA[pat, neur]:.1f}Hz',
                             fontsize=7)
            elif nPattern == 1:
                ax.set_title(f'Neuron {neur + 1}', fontsize=9)

            if pat == nPattern - 1:
                ax.set_xlabel('# discharges', fontsize=7)
            if neur == 0:
                ax.set_ylabel('Latency (ms)', fontsize=7)

            ax.tick_params(labelsize=6)

    fig.suptitle(f'{title_prefix} (inhib={params.inhibStrength:.2f})', fontsize=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_inhibition_strength_analysis(inhib_values, mean_diffs, std_diffs, params, filename):
    """Plot inhibition strength vs. latency difference."""
    fig, ax = plt.subplots(figsize=(7, 4))

    inhib_pct = np.array(inhib_values) * 100  # percentage of threshold
    mean_diffs_ms = np.array(mean_diffs) * 1000

    if std_diffs is not None and len(std_diffs) > 0:
        std_diffs_ms = np.array(std_diffs) * 1000
        ax.errorbar(inhib_pct, mean_diffs_ms, yerr=std_diffs_ms,
                    fmt='o-', capsize=3, markersize=6, linewidth=1.5)
    else:
        ax.plot(inhib_pct, mean_diffs_ms, 'o-', markersize=6, linewidth=1.5)

    ax.set_xlabel('IPSP amplitude (% of threshold)')
    ax.set_ylabel('Mean latency difference (ms)')
    ax.set_title('Lateral inhibition strength vs. latency difference')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_latency_matrix(latency, HR, FA, params, filename):
    """Plot nPattern x nNeuron latency matrix for Part 4."""
    nPattern = params.nPattern
    nNeuron = params.nNeuron

    fig, axes = plt.subplots(nPattern, nNeuron,
                             figsize=(3 * nNeuron, 2.5 * nPattern),
                             squeeze=False)
    cpDuration_ms = params.copyPasteDuration * 1000

    for pat in range(nPattern):
        for neur in range(nNeuron):
            ax = axes[pat, neur]
            key = (pat, neur)
            if key in latency and len(latency[key]) > 0:
                lat_ms = np.array(latency[key]) * 1000
                ax.plot(lat_ms, '.', markersize=2, color='black')
                ax.set_xlim(0, len(lat_ms))
                ax.set_ylim(0, cpDuration_ms)

            ax.set_title(f'HR={100 * HR[pat, neur]:.0f}% FA={FA[pat, neur]:.1f}Hz',
                         fontsize=6)
            if pat == nPattern - 1:
                ax.set_xlabel('# discharges', fontsize=6)
            if neur == 0:
                ax.set_ylabel(f'Pattern {pat + 1}\nLatency (ms)', fontsize=6)
            ax.tick_params(labelsize=5)

    fig.suptitle(f'Multi-pattern latency matrix (inhib={params.inhibStrength:.2f})',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


def plot_weight_distribution(neurons, filename):
    """Plot final weight distributions for all neurons."""
    nNeuron = len(neurons)
    ncols = min(3, nNeuron)
    nrows = int(np.ceil(nNeuron / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for i, neuron in enumerate(neurons):
        r, c = i // ncols, i % ncols
        ax = axes[r, c]
        ax.hist(neuron.weight, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title(f'Neuron {i + 1} (sum={neuron.weight.sum():.1f})')
        ax.set_xlabel('Weight')
        ax.set_ylabel('Count')

    for i in range(nNeuron, nrows * ncols):
        r, c = i // ncols, i % ncols
        axes[r, c].set_visible(False)

    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")
