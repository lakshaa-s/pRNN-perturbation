"""
dqn_architecture_bible.py
=====================

Builds a canonical "bible" mapping each DQN run number -> its architecture,
for the multimodal_mazes cohort.

Why this exists
---------------
In scripts/DQN_exp.py every run's architecture is set *purely* by its
job_index, and each trained agent is pickled as "{job_index}.pickle".
The flags are generated deterministically:

    wm_flags = np.array(list(itertools.product([0, 1], repeat=7)))
    wm_flags = np.vstack((wm_flags[0], wm_flags))   # all-zeros row duplicated
    wm_flag  = wm_flags[job_index]

So the number->model map is fully recoverable from the filename alone.
This module reconstructs it (`architecture_from_index`), AND independently
reads it back off a loaded agent (`architecture_from_agent`) so the two can
be cross-checked -- that cross-check is the safeguard against loading the
wrong model.

The 7 weight-matrix flags, in order, are: [ii, hh, oo, io, oi, hi, oh]
    ii = input->input    (lateral)
    hh = hidden->hidden  (recurrence)
    oo = output->output  (lateral)
    io = input->output   (skip)
    oi = output->input   (skip)
    hi = hidden->input   (feedback)
    oh = output->hidden   (feedback)

Key facts about the cohort (verified against the source):
    * 129 indices total (0..128): itertools gives 128, plus the duplicated
      all-zeros row at the front.
    * index 0 AND index 1 are BOTH pure feedforward (all flags zero). They
      differ ONLY in hidden size: index 1 has n_hidden_units=34, every other
      index has 8. So "FF" is two distinct runs, distinguishable only by
      hidden size / filename -- NOT by which weight matrices are present.
    * index 128 = all seven matrices present = "RNN_full".

Recurrence note (verified empirically against forward()):
    Six of the seven optional connections (ii, hh, oo, oi, hi, oh) feed a
    time-(t-1) activation into time t, so they create temporal recurrence /
    memory. The ONE exception is io (input->output), a within-timestep skip
    connection that uses the current input -- it adds no memory. So an
    architecture is dynamically recurrent iff ANY non-io flag is set; the
    bible reports this as `is_recurrent`, with `has_hidden_recurrence`
    separately flagging the literal hidden->hidden loop.

Source of truth:
    The authoritative description of a run is the agent's OWN stored
    `wm_flags` (and n_hidden_units), read by `architecture_from_agent`.
    `architecture_from_index` reconstructs the same thing from the filename
    number; it agrees because the generation logic in DQN_exp.py has been
    unchanged since Oct 2024. `build_bible(..., verify=True)` loads each
    agent and asserts the two agree, so any filename/content mismatch is
    raised rather than silently mislabelled.
"""

import os
import glob
import pickle
import itertools

import numpy as np

# Canonical flag order and the connection each bit controls.
# This order is verified against agent_dqn.py (__init__ AND forward) and
# against analysis/DQN_analysis.calculate_dqn_w_norms, which assumes
# parameters() yields [input_to_hidden, hidden_to_output, *optional matrices]
# with the optional matrices in exactly this flag order.
FLAG_NAMES = ["ii", "hh", "oo", "io", "oi", "hi", "oh"]
FLAG_CONNECTION = {
    "ii": "input_to_input",
    "hh": "hidden_to_hidden",
    "oo": "output_to_output",
    "io": "input_to_output",
    "oi": "output_to_input",
    "hi": "hidden_to_input",
    "oh": "output_to_hidden",
}
# These two always exist regardless of flags.
ALWAYS_PRESENT = ["input_to_hidden", "hidden_to_output"]

# Which flags carry state ACROSS timesteps (verified empirically by feeding
# different histories into forward() and checking the current output changes).
# All optional connections are temporal EXCEPT io (input->output), which is a
# within-timestep skip: forward() feeds it new_input (time t), not a t-1 value.
# So an architecture is dynamically recurrent iff any NON-io flag is set.
TEMPORAL_FLAGS = ["ii", "hh", "oo", "oi", "hi", "oh"]  # everything but io
_TEMPORAL_IDX = [FLAG_NAMES.index(n) for n in TEMPORAL_FLAGS]
_HH_IDX = FLAG_NAMES.index("hh")

N_INDICES = 129  # 128 from itertools.product + 1 duplicated all-zeros row


def _all_wm_flags():
    """Reproduce exactly the flag table built in scripts/DQN_exp.py."""
    wm = np.array(list(itertools.product([0, 1], repeat=7)))
    wm = np.vstack((wm[0], wm))
    return wm


_WM_TABLE = _all_wm_flags()


def _label(flags):
    """Human-readable architecture label from a 7-bit flag vector."""
    flags = np.asarray(flags).astype(int)
    if flags.sum() == 0:
        return "FF"
    if flags.sum() == 7:
        return "RNN_full"
    present = [n for n, b in zip(FLAG_NAMES, flags) if b]
    return "+".join(present)


def _hidden_units_for_index(job_index):
    """Replicates the n_hidden_units rule in DQN_exp.run_exp."""
    return 34 if job_index == 1 else 8


def architecture_from_index(job_index):
    """
    Reconstruct a run's architecture from its job_index (i.e. its filename
    number) WITHOUT loading the pickle. This is the ground-truth generator.

    Returns a dict describing the architecture.
    """
    if not (0 <= job_index < N_INDICES):
        raise ValueError(
            f"job_index {job_index} out of range 0..{N_INDICES - 1}"
        )
    flags = _WM_TABLE[job_index].astype(int)
    present = [FLAG_CONNECTION[n] for n, b in zip(FLAG_NAMES, flags) if b]
    return {
        "job_index": int(job_index),
        "label": _label(flags),
        "wm_flags": "".join(str(b) for b in flags),  # e.g. "1111111"
        "n_hidden_units": _hidden_units_for_index(job_index),
        "is_feedforward": bool(flags.sum() == 0),  # lab's "FF" = no extra matrices
        "is_recurrent": bool(flags[_TEMPORAL_IDX].sum() > 0),  # ANY temporal flag
        "has_hidden_recurrence": bool(flags[_HH_IDX] == 1),  # the literal hh loop
        "n_weight_matrices": int(flags.sum()) + len(ALWAYS_PRESENT),
        "weight_matrices": ALWAYS_PRESENT + present,
    }


def architecture_from_agent(agnt):
    """
    Read a run's architecture back off a *loaded* agent, two independent ways:
      1. its stored .wm_flags vector
      2. enumerating the nn.Linear submodules actually present
    and assert the two agree. This is Ghosh's "run the agent and let it list
    its weight matrices" check, made programmatic.

    Returns a dict in the same schema as architecture_from_index, plus the
    concrete parameter count read from the live model.
    """
    flags = np.asarray(agnt.wm_flags).astype(int)

    # (2) which Linear layers does the live module actually carry?
    present_modules = {
        name for name, mod in agnt.named_modules()
        if mod.__class__.__name__ == "Linear"
    }
    expected_modules = set(ALWAYS_PRESENT) | {
        FLAG_CONNECTION[n] for n, b in zip(FLAG_NAMES, flags) if b
    }
    if present_modules != expected_modules:
        raise ValueError(
            "wm_flags disagree with the actual Linear submodules.\n"
            f"  flags imply: {sorted(expected_modules)}\n"
            f"  module has : {sorted(present_modules)}"
        )

    n_parameters = sum(p.numel() for p in agnt.parameters())
    present = [FLAG_CONNECTION[n] for n, b in zip(FLAG_NAMES, flags) if b]
    return {
        "label": _label(flags),
        "wm_flags": "".join(str(b) for b in flags),
        "n_hidden_units": int(agnt.n_hidden_units),
        "n_input_units": int(agnt.n_input_units),
        "n_output_units": int(agnt.n_output_units),
        "is_feedforward": bool(flags.sum() == 0),
        "is_recurrent": bool(flags[_TEMPORAL_IDX].sum() > 0),
        "has_hidden_recurrence": bool(flags[_HH_IDX] == 1),
        "n_weight_matrices": int(flags.sum()) + len(ALWAYS_PRESENT),
        "weight_matrices": ALWAYS_PRESENT + present,
        "n_parameters": int(n_parameters),
    }


def build_bible(results_dir, out_csv=None, out_json=None, verify=True):
    """
    Walk a results directory of "{job_index}.pickle" files, derive each run's
    architecture, and (if verify) cross-check the filename-derived identity
    against the loaded agent's own wm_flags. Writes a CSV and/or JSON manifest.

    Arguments:
        results_dir: folder containing N.pickle agent files.
        out_csv / out_json: optional output paths for the manifest.
        verify: if True, load each pickle and assert it matches its index.
                if False, build the bible from filenames alone (fast, no load).

    Returns:
        list of row dicts (one per run), sorted by job_index.
    """
    rows = []
    mismatches = []
    paths = sorted(
        glob.glob(os.path.join(results_dir, "*.pickle")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            job_index = int(stem)
        except ValueError:
            # not a numbered run file (e.g. a config pickle) -- skip
            continue

        row = architecture_from_index(job_index)
        row["file"] = os.path.basename(path)

        if verify:
            with open(path, "rb") as f:
                agnt = pickle.load(f)
            live = architecture_from_agent(agnt)
            row["n_input_units"] = live["n_input_units"]
            row["n_output_units"] = live["n_output_units"]
            row["n_parameters"] = live["n_parameters"]
            # Cross-check: do filename and live model agree?
            agree = (
                live["wm_flags"] == row["wm_flags"]
                and live["n_hidden_units"] == row["n_hidden_units"]
            )
            row["verified"] = bool(agree)
            if not agree:
                mismatches.append(
                    f"  {row['file']}: filename says wm={row['wm_flags']} "
                    f"hidden={row['n_hidden_units']}, but loaded model has "
                    f"wm={live['wm_flags']} hidden={live['n_hidden_units']}"
                )
        rows.append(row)

    if mismatches:
        raise RuntimeError(
            "Bible verification FAILED -- loaded models do not match their "
            "filenames:\n" + "\n".join(mismatches)
        )

    if out_csv:
        _write_csv(rows, out_csv)
    if out_json:
        import json
        with open(out_json, "w") as f:
            json.dump(rows, f, indent=2)

    return rows


def build_nested_bible(
    base_dir, folder_prefix="test", out_csv=None, out_json=None,
    verify="sample", sample_indices=(0, 1, 128), seed=0,
):
    """
    Build a bible for the nested cohort layout:

        base_dir/
            test1351/   0.pickle ... 128.pickle   exp_config.ini
            test1352/   0.pickle ... 128.pickle   exp_config.ini
            ...

    Emits one row per (folder, job_index). The inner job_index->architecture
    map has been verified identical across folders, so by default this does
    NOT load all pickles (50 x 129 = thousands of loads). Instead it derives
    every row from the deterministic index map and *spot-checks* a few indices
    per folder against the actual stored wm_flags.

    Arguments:
        base_dir: folder containing the testXXXX subfolders.
        folder_prefix: subfolder name prefix (default "test").
        verify:
            "sample" -> load only `sample_indices` per folder and cross-check
                        (fast; catches a folder generated differently).
            "full"   -> load and cross-check every pickle in every folder
                        (thorough but slow).
            "none"   -> trust the index map; load nothing.
        sample_indices: which indices to spot-check when verify="sample".
        seed: unused placeholder for reproducible sampling extensions.

    Returns:
        list of row dicts, each with run_folder + job_index + architecture.
    """
    folders = sorted(
        d for d in os.listdir(base_dir)
        if d.startswith(folder_prefix)
        and os.path.isdir(os.path.join(base_dir, d))
    )

    rows = []
    mismatches = []
    for folder in folders:
        fdir = os.path.join(base_dir, folder)
        pickles = sorted(
            (f for f in os.listdir(fdir) if f.endswith(".pickle")),
            key=lambda f: int(f[:-len(".pickle")]),
        )
        for fname in pickles:
            job_index = int(fname[:-len(".pickle")])
            row = {"run_folder": folder}
            row.update(architecture_from_index(job_index))
            row["file"] = os.path.join(folder, fname)

            do_load = (verify == "full") or (
                verify == "sample" and job_index in set(sample_indices)
            )
            if do_load:
                with open(os.path.join(fdir, fname), "rb") as fh:
                    agnt = pickle.load(fh)
                live = architecture_from_agent(agnt)
                agree = (
                    live["wm_flags"] == row["wm_flags"]
                    and live["n_hidden_units"] == row["n_hidden_units"]
                )
                row["verified"] = bool(agree)
                row["n_parameters"] = live["n_parameters"]
                if not agree:
                    mismatches.append(
                        f"  {row['file']}: index map says wm={row['wm_flags']}"
                        f"/h{row['n_hidden_units']}, stored model has "
                        f"wm={live['wm_flags']}/h{live['n_hidden_units']}"
                    )
            rows.append(row)

    if mismatches:
        raise RuntimeError(
            "Nested bible verification FAILED -- some stored models do not "
            "match the index map:\n" + "\n".join(mismatches)
        )

    if out_csv:
        _write_csv(rows, out_csv, nested=True)
    if out_json:
        import json
        with open(out_json, "w") as f:
            json.dump(rows, f, indent=2)

    return rows


def _write_csv(rows, out_csv, nested=False):
    import csv
    # Stable, readable column order; list-valued cols joined with ';'.
    preferred = [
        "run_folder", "job_index", "file", "label", "wm_flags",
        "n_hidden_units", "n_input_units", "n_output_units", "n_parameters",
        "is_feedforward", "is_recurrent", "has_hidden_recurrence",
        "n_weight_matrices", "weight_matrices", "verified",
    ]
    cols = [c for c in preferred if any(c in r for r in rows)]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r = dict(r)
            if isinstance(r.get("weight_matrices"), list):
                r["weight_matrices"] = ";".join(r["weight_matrices"])
            w.writerow(r)


if __name__ == "__main__":
    # Quick demo: print the bible for the whole index space, no files needed.
    print(f"{'idx':>4}  {'label':<12}  {'wm_flags':<8}  hidden")
    for i in range(N_INDICES):
        a = architecture_from_index(i)
        print(f"{a['job_index']:>4}  {a['label']:<12}  {a['wm_flags']:<8}  "
              f"{a['n_hidden_units']}")
