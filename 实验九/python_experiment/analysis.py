"""Analysis utilities: latency computation, hit rate, false alarm rate."""

import numpy as np


def compute_latencies(neurons, params):
    """Compute response latencies for each (pattern, neuron) pair.

    Parameters
    ----------
    neurons : list of Neuron
    params : Params

    Returns
    -------
    latency : dict  {(pat, neur): ndarray}
        Latency values for each firing (0 = false alarm).
    HR : ndarray  (nPattern, nNeuron)
        Hit rates.
    FA : ndarray  (nPattern, nNeuron)
        False alarm rates (Hz).
    final_latency : ndarray  (nPattern, nNeuron)
        Mean latency in the final third of simulation (NaN if not a success).
    """
    nPattern = params.nPattern
    nNeuron = len(neurons)
    T = params.T
    nRun = params.nRun

    # Range for evaluation: last run
    eval_start = (nRun - 1) * T
    eval_end = nRun * T

    latency = {}
    HR = np.zeros((nPattern, nNeuron))
    FA = np.zeros((nPattern, nNeuron))
    final_latency = np.full((nPattern, nNeuron), np.nan)

    for neur_idx, neuron in enumerate(neurons):
        nF = int(neuron.nFiring)
        if nF == 0:
            for pat in range(nPattern):
                latency[(pat, neur_idx)] = np.array([])
            continue

        firingTimes = neuron.firingTime[:nF]

        for pat in range(nPattern):
            lat_vals = []
            posCopyPaste = params.posCopyPaste[pat]

            for ft in firingTimes:
                # Determine which run this firing belongs to
                run_idx = int(np.ceil(ft / T)) - 1
                if run_idx < 0:
                    run_idx = 0

                # Determine which pattern window it falls in
                period = int(np.ceil((ft - run_idx * T) / params.copyPasteDuration))

                # Check if this is a hit
                if period - 1 in posCopyPaste:
                    # Hit: latency relative to pattern start
                    lat = ft - run_idx * T - (period - 1) * params.copyPasteDuration
                    lat_vals.append(lat)
                else:
                    # False alarm (or out-of-run reference)
                    lat_vals.append(0.0)

            latency[(pat, neur_idx)] = np.array(lat_vals)

            # Compute HR and FA in the evaluation window
            in_window = (firingTimes >= eval_start) & (firingTimes < eval_end)
            if np.any(in_window):
                lat_arr = np.array(lat_vals)
                hits = np.sum((lat_arr > 0) & in_window)

                # Count total pattern occurrences in eval window
                n_pattern_occurrences = np.sum(
                    (np.array(posCopyPaste) * params.copyPasteDuration +
                     (nRun - 1) * T >= eval_start) &
                    (np.array(posCopyPaste) * params.copyPasteDuration +
                     (nRun - 1) * T < eval_end)
                )

                if n_pattern_occurrences > 0:
                    HR[pat, neur_idx] = hits / n_pattern_occurrences

                total_firings = np.sum(in_window)
                FA[pat, neur_idx] = (total_firings - hits) / (eval_end - eval_start)

                # Final latency: last third of eval window
                final_third_start = eval_start + 2.0 / 3.0 * (eval_end - eval_start)
                in_final = (firingTimes >= final_third_start) & (firingTimes < eval_end)
                if np.any(in_final):
                    final_lats = lat_arr[in_final]
                    final_lats = final_lats[final_lats > 0]
                    if len(final_lats) > 0:
                        final_latency[pat, neur_idx] = np.mean(final_lats)

    return latency, HR, FA, final_latency


def identify_successful_neurons(HR, FA, hr_threshold=0.9, fa_threshold=1.0):
    """Identify neurons that successfully learned patterns.

    Returns
    -------
    success : ndarray (nPattern, nNeuron)
        Boolean mask of successful (pattern, neuron) pairs.
    """
    return (HR > hr_threshold) & (FA < fa_threshold)


def compute_latency_differences(final_latency, success_mask):
    """Compute adjacent latency differences for successful neurons.

    Returns
    -------
    diffs : list of float
        Latency differences between adjacent successful neurons (sorted by latency).
    """
    nPattern = final_latency.shape[0]
    diffs = []

    for pat in range(nPattern):
        latencies = final_latency[pat, :]
        successful = success_mask[pat, :]
        success_lats = latencies[successful]
        success_lats = success_lats[~np.isnan(success_lats)]
        success_lats = np.sort(success_lats)

        for i in range(1, len(success_lats)):
            diffs.append(success_lats[i] - success_lats[i - 1])

    return diffs
