"""Spike train generation with embedded repeating patterns.

Vectorized approach: generates background for all afferents at once,
then embeds patterns. Only participating afferents have spikes replaced.
"""

import numpy as np


def generate_spike_train(params):
    """Generate full spike train with embedded patterns.

    Returns spikeList, afferentList (2 values).
    """
    rng = np.random.RandomState(params.randomState)

    nAfferent = params.nAfferent
    T_total = params.T * params.nRun
    cpDuration = params.copyPasteDuration
    jitter = params.jitter
    max_total_rate = params.maxFiringRate + params.spontaneousActivity

    # ── Step 1: Background spikes ──
    print(f"  Generating background (max rate={max_total_rate:.0f}Hz, "
          f"T={T_total:.1f}s, N={nAfferent})...")

    afferent_rates = rng.uniform(0, params.maxFiringRate, nAfferent) + params.spontaneousActivity
    mean_rate = np.mean(afferent_rates)
    print(f"  Mean rate: {mean_rate:.1f} Hz")

    n_candidates = int(max_total_rate * T_total * nAfferent * 1.05)
    cand_times = rng.uniform(0, T_total, n_candidates)
    cand_afferents = rng.randint(0, nAfferent, n_candidates).astype(np.uint16)

    accept_prob = afferent_rates[cand_afferents] / max_total_rate
    mask = rng.uniform(0, 1, n_candidates) < accept_prob

    spikeList = cand_times[mask]
    afferentList = cand_afferents[mask]

    print(f"  Generated {len(spikeList)} background spikes "
          f"(expected ~{mean_rate * T_total * nAfferent:.0f})")

    # ── Step 2: Pattern embedding ──
    print(f"  Embedding {params.nPattern} pattern(s)...")

    # Build participating sets
    participating_sets = {}
    for pat in range(params.nPattern):
        if params.nPattern == 1:
            participating_sets[pat] = set(range(params.nCopyPasteAfferent))
        else:
            p_set = set()
            for a in range(nAfferent):
                if rng.rand() < params.nCopyPasteAfferent / nAfferent:
                    p_set.add(a)
            participating_sets[pat] = p_set

    # Extract templates from first occurrence of each pattern
    pattern_templates = {}
    for pat in range(params.nPattern):
        posList = params.posCopyPaste.get(pat, [])
        if len(posList) == 0:
            pattern_templates[pat] = {}
            continue
        participating = participating_sets[pat]
        first_pos = posList[0]
        win_start = first_pos * cpDuration
        win_end = win_start + cpDuration

        templates = {}
        for a in participating:
            in_window = ((spikeList >= win_start) & (spikeList < win_end) &
                         (afferentList == a))
            rel_times = spikeList[in_window] - win_start
            if len(rel_times) > 0:
                templates[a] = rel_times.copy()
        pattern_templates[pat] = templates

    # Apply copy-delete-paste: only delete spikes from PARTICIPATING afferents
    all_new_spikes = []
    all_new_afferents = []
    keep_mask = np.ones(len(spikeList), dtype=bool)

    for run_idx in range(params.nRun):
        run_offset = run_idx * params.T

        for pat in range(params.nPattern):
            posList = params.posCopyPaste.get(pat, [])
            templates = pattern_templates.get(pat, {})
            participating = participating_sets[pat]
            participating_list = list(participating)

            for p_idx in posList:
                win_start = run_offset + p_idx * cpDuration
                win_end = win_start + cpDuration

                # Only remove spikes from PARTICIPATING afferents in this window
                if len(participating_list) > 0:
                    in_win = ((spikeList >= win_start) & (spikeList < win_end) &
                              np.isin(afferentList, participating_list))
                    keep_mask[in_win] = False

                # Paste template copies
                for a, rel_times in templates.items():
                    for rt in rel_times:
                        jittered = rt + jitter * rng.randn()
                        jittered = np.clip(jittered, 0, cpDuration)
                        all_new_spikes.append(win_start + jittered)
                        all_new_afferents.append(a)

    # Combine
    if len(all_new_spikes) > 0:
        new_spikes = np.array(all_new_spikes)
        new_afferents = np.array(all_new_afferents, dtype=np.uint16)
        kept_spikes = spikeList[keep_mask]
        kept_afferents = afferentList[keep_mask]
        spikeList = np.concatenate([kept_spikes, new_spikes])
        afferentList = np.concatenate([kept_afferents, new_afferents])

    # ── Step 3: Sort ──
    sort_idx = np.argsort(spikeList)
    spikeList = spikeList[sort_idx]
    afferentList = afferentList[sort_idx].astype(np.uint16)

    print(f"  Final: {len(spikeList)} spikes")

    return spikeList, afferentList
