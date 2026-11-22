"""MMZCA: average CM-to-root hop depth versus H_max (elbow-method experiment).

Run this file INSIDE the existing MMZCA project, beside main.py.  No changes
are required in data_cluster.py, utils/, or options/constants.py.

Experiment (all values are SUMO timestep INDICES):
    H_MAX_VALUES = np.arange(0, 10, 1)
    WINDOW_STARTS = np.arange(400, 3000, 100)  # 400..2900 inclusive
    WINDOW_SECONDS = 200                       # exactly 200 update/samples
    TRANSMISSION_RANGES = (100, 200, 300)      # metres
    DATASETS = ('smallsize', 'midsize', 'largesize')

H_max=0 is INVALID for MMZCA, so it is stored as a null/undefined result and
is shown on the x axis without inventing a zero-hop CM observation.

Metric: at the end of each SUMO update, take the arithmetic mean of hop_count
across currently in-area associated ordinary CMs. The 200-s window value is
the arithmetic mean of these per-tick means over ticks that contain CMs.
CHs and SAs are not included; ticks without CMs are undefined, not zero.

Confidence intervals: circular moving-block bootstrap over the 200-s window
means, ordered by start time, which partially accounts for the overlapping
100-s starts. These are *approximate, within-trace* CIs; overlapping windows
from a single SUMO trace are NOT independent mobility realizations.
For publication-grade replication CIs, use independent SUMO seeds/traces.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import t as student_t

from configs.config import Configs
from data_cluster import DataTable
from zonex import ZoneID
import options.constants as constants_module


# ---------------------------------------------------------------------------
# Requested experiment grid
# ---------------------------------------------------------------------------
H_MAX_VALUES = np.arange(10, 21, 1)
WINDOW_STARTS = np.arange(400, 3000, 100)
WINDOW_SECONDS = 200
TRANSMISSION_RANGES = (100, 200, 300)
DATASETS = (
    ("smallsize", "High sparsity"),
    ("midsize", "Medium sparsity"),
    ("largesize", "Low sparsity"),
)
CI_LEVEL = 0.95
SEED = 2026


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traffic-root", type=Path, default=None,
        help="Directory containing final_data_Richmondhill_smallsize, _midsize, and _largesize. "
             "By default, infer the path using the same rule as options/constants.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("elbow_method_results"))
    parser.add_argument("--ci-method", choices=("block-bootstrap", "t"),
                        default="block-bootstrap",
                        help="Block bootstrap is preferable for overlapping windows. "
                             "Student-t is a nominal CI that assumes independent windows.")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-block-windows", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previously saved experiment with matching settings.")
    parser.add_argument("--no-show", action="store_true", help="Save plot without opening its window.")
    parser.add_argument("--progress-every", type=int, default=10,
                        help="Print one progress line per this many completed runs.")
    parser.add_argument("--save-tick-series", action="store_true",
                        help="Include 200 per-tick CM hop means per window in the JSON.")
    return parser


def load_original_config():
    """Allow custom args in this file without forwarding them to Configs()."""
    argv = sys.argv
    try:
        sys.argv = [argv[0]]
        return Configs().config
    finally:
        sys.argv = argv


def resolve_traffic_root(override):
    if override is not None:
        return override.expanduser().resolve()
    # Matches: Path(__file__).parent.parent.parent / 'traffic_data'
    # when __file__ belongs to <project>/options/constants.py.
    return Path(constants_module.__file__).resolve().parents[2] / "traffic_data"


def load_dataset(traffic_root, name):
    """Parse SUMO XML ONCE per density, not for every H_max/window/TR run."""
    import xml.dom.minidom

    folder = traffic_root / ("final_data_Richmondhill_" + name)
    paths = {
        "sumo_trace": folder / "sumoTrace_no_bus_and_rsu.xml",
        "sumo_edge": folder / "osm.net.xml",
        "sumo_node": folder / "osm_bbox.osm.xml",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing SUMO input: {path}\n"
                "Set --traffic-root to the folder containing the three "
                "final_data_Richmondhill_* subfolders."
            )
    docs = {field: xml.dom.minidom.parse(str(path)) for field, path in paths.items()}
    timesteps = docs["sumo_trace"].documentElement.getElementsByTagName("timestep")
    max_timestep_needed = int(WINDOW_STARTS[-1]) + WINDOW_SECONDS
    if len(timesteps) <= max_timestep_needed:
        raise ValueError(
            f"{name}: {len(timesteps)} timesteps, but index {max_timestep_needed} "
            f"is required (last start {WINDOW_STARTS[-1]} + {WINDOW_SECONDS} updates)."
        )
    docs["times"] = timesteps
    docs["fcd"] = docs["sumo_trace"].documentElement
    return docs


def make_zones(config):
    """Construct the exact same zones as the uploaded main.py."""
    zones = {}
    for level, code in (("micro", "mic"), ("meso", "mes"), ("macro", "mac")):
        obj = ZoneID(config, zoning_mode=code)
        obj.zones()
        zones[level] = obj
    return zones


def restore_zone_methods(zones):
    """DataTable caches macro.neighbor_zones per instance; prevent wrapper nesting.

    The ZoneID grid itself is immutable and can be reused between runs. Reset
    only the per-instance monkeypatched method before constructing DataTable.
    """
    macro = zones["macro"]
    if "neighbor_zones" in vars(macro):
        del macro.neighbor_zones


def stable_seed(density, tr, window_start):
    """Common pseudorandom seed across H_max for the same scenario/window."""
    payload = f"{SEED}:{density}:{tr}:{window_start}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def mean_cm_hops_for_tick(cluster, h_max):
    """Return (mean, count) for associated, in-study-area ordinary CMs.

    Uses already-committed hop_count; never substitutes a geometric distance or
    shortest path. Raises for inconsistent committed state instead of biasing
    the mean by silently ignoring bad assignments.
    """
    total_depth = 0
    cm_count = 0
    for veh_id in cluster.veh_table.ids():
        node = cluster.veh_table.values(veh_id)
        if not node or not node.get("in_area", True):
            continue
        if node.get("cluster_head") is True:
            continue
        root_id = node.get("root_ch") or node.get("primary_ch")
        if root_id is None:
            continue  # SA is NOT a zero-hop CM

        depth = node.get("hop_count")
        if (isinstance(depth, bool) or not isinstance(depth, (int, np.integer))
                or depth < 1 or depth > h_max):
            raise RuntimeError(
                f"SUMO tick {cluster.time}: CM {veh_id} has invalid "
                f"hop_count={depth!r}, root={root_id!r}, H_max={h_max}."
            )
        if node.get("root_connection_valid") is False:
            raise RuntimeError(
                f"SUMO tick {cluster.time}: CM {veh_id} has invalid root "
                "connection after cluster maintenance."
            )
        total_depth += int(depth)
        cm_count += 1

    return (total_depth / cm_count if cm_count else None), cm_count


def run_one_window(config, zones, h_max, window_start, save_tick_series):
    """Run the unchanged MMZCA clustering phases for exactly 200 updates."""
    restore_zone_methods(zones)
    cluster = DataTable(config, zones)

    tick_means = []
    valid_tick_count = 0
    cm_observations = 0
    tick_mean_sum = 0.0
    try:
        for _ in range(WINDOW_SECONDS):
            cluster.update(config)
            cluster.update_cluster(cluster.veh_table.ids(), config)
            cluster.stand_alones_cluster(config)
            cluster.update_kernels(config)

            # Existing VCSM/TVCT computations and their printouts/plots are
            # deliberately disabled: only the hop metric is needed here.
            tick_mean, cm_count = mean_cm_hops_for_tick(cluster, h_max)
            if tick_mean is not None:
                tick_mean_sum += tick_mean
                valid_tick_count += 1
                cm_observations += cm_count
            if save_tick_series:
                tick_means.append(tick_mean)

        record = {
            "start_time": int(window_start),
            "last_sampled_tick": int(cluster.time),
            "average_hops": (tick_mean_sum / valid_tick_count
                             if valid_tick_count else None),
            "ticks_with_cms": valid_tick_count,
            "cm_tick_observations": cm_observations,
        }
        if save_tick_series:
            record["tick_average_hops"] = tick_means
        return record
    finally:
        # Drop the large vehicle history before constructing the next run.
        del cluster


def confidence_interval(means, method, block_windows, repetitions, seed):
    """95% interval of the mean of WINDOW-level means, not 200*tick samples."""
    a = np.asarray([x for x in means if x is not None and math.isfinite(x)], dtype=float)
    n = len(a)
    if n == 0:
        return None, None
    if n == 1:
        return None, None
    if np.all(a == a[0]):
        return float(a[0]), float(a[0])

    if method == "t":
        half_width = student_t.ppf(0.5 + CI_LEVEL / 2, df=n - 1) * a.std(ddof=1) / np.sqrt(n)
        return float(a.mean() - half_width), float(a.mean() + half_width)

    if repetitions < 100:
        raise ValueError("--bootstrap-replicates must be >= 100")
    if block_windows < 1:
        raise ValueError("--bootstrap-block-windows must be >= 1")
    rng = np.random.default_rng(seed)
    block = min(block_windows, n)
    blocks_needed = math.ceil(n / block)
    offsets = np.arange(block)
    # Circular moving-block bootstrap: retain local ordering and overlap.
    boots = np.empty(repetitions, dtype=float)
    for rep in range(repetitions):
        starts = rng.integers(0, n, size=blocks_needed)
        sampled_indices = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[rep] = a[sampled_indices].mean()
    low, high = np.quantile(boots, [(1 - CI_LEVEL) / 2, (1 + CI_LEVEL) / 2])
    return float(low), float(high)


def produce_summary(runs, ci_method, block_windows, repetitions):
    # Group in requested, fixed order. Null h_max=0 is explicitly included.
    by_case = {}
    for rec in runs:
        by_case.setdefault((rec["density"], rec["transmission_range_m"], rec["h_max"]), []).append(rec)

    summary = []
    for density, _ in DATASETS:
        for tr in TRANSMISSION_RANGES:
            for h in H_MAX_VALUES:
                h = int(h)
                group = sorted(by_case.get((density, tr, h), []), key=lambda r: r["start_time"])
                window_means = [row["average_hops"] for row in group]
                valid = [x for x in window_means if x is not None]
                avg = float(np.mean(valid)) if valid else None
                low, high = confidence_interval(
                    window_means, ci_method, block_windows, repetitions,
                    stable_seed(density, tr, h) + 39,
                ) if h > 0 else (None, None)
                summary.append({
                    "density": density,
                    "transmission_range_m": int(tr),
                    "h_max": h,
                    "mean_average_hops": avg,
                    "ci95_lower": low,
                    "ci95_upper": high,
                    "n_windows_with_cms": len(valid),
                    "n_windows_completed": len(group),
                    "status": "invalid_h_max" if h == 0 else (
                        "complete" if len(group) == len(WINDOW_STARTS) else "incomplete"
                    ),
                })
    return summary


def write_json_atomically(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2, allow_nan=False)
    tmp.replace(path)


def plot_summary(summary, out_path, show=True):
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True, constrained_layout=True)
    for ax, (density, title) in zip(axes, DATASETS):
        for tr in TRANSMISSION_RANGES:
            points = [s for s in summary if s["density"] == density
                      and s["transmission_range_m"] == tr
                      and s["mean_average_hops"] is not None]
            points.sort(key=lambda p: p["h_max"])
            if not points:
                continue
            x = np.array([p["h_max"] for p in points], dtype=float)
            y = np.array([p["mean_average_hops"] for p in points], dtype=float)
            sns.lineplot(x=x, y=y, ax=ax, label=f"TR = {tr} m",
                         marker="o", linewidth=2, errorbar=None, estimator=None)
            color = ax.lines[-1].get_color()
            valid_bands = [p for p in points if p["ci95_lower"] is not None]
            if valid_bands:
                xb = [p["h_max"] for p in valid_bands]
                lb = [p["ci95_lower"] for p in valid_bands]
                ub = [p["ci95_upper"] for p in valid_bands]
                ax.fill_between(xb, lb, ub, color=color, alpha=0.18, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("h_max")
        ax.set_ylabel("average hops between CMs and associated CHs")
        ax.set_xticks(H_MAX_VALUES)
        ax.set_xlim(-0.25, 9.25)
        if ax.get_legend() is not None:
            ax.legend(title="Transmission range", loc="best")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = argument_parser().parse_args()
    if args.progress_every < 1:
        raise ValueError("--progress-every must be >= 1")
    if args.bootstrap_replicates < 100:
        raise ValueError("--bootstrap-replicates must be >= 100")
    if args.bootstrap_block_windows < 1:
        raise ValueError("--bootstrap-block-windows must be >= 1")

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "mmzca_elbow_results.json"
    plot_path = out_dir / "mmzca_elbow_method.png"
    traffic_root = resolve_traffic_root(args.traffic_root)

    base_config = load_original_config()
    metadata = {
        "algorithm": "MMZCA",
        "density_names": [name for name, _ in DATASETS],
        "transmission_ranges_m": list(TRANSMISSION_RANGES),
        "h_max_values": H_MAX_VALUES.tolist(),
        "h_max_zero": "invalid for MMZCA; the hop mean is undefined (null)",
        "window_starts": WINDOW_STARTS.tolist(),
        "end_time_exclusive": 3000,
        "window_seconds": WINDOW_SECONDS,
        "samples_per_window": WINDOW_SECONDS,
        "sampled_ticks": "start_time+1 through start_time+200 (inclusive)",
        "metric": "per-tick arithmetic mean of hop_count among associated in-area CMs, "
                  "then arithmetic mean across valid ticks of each 200-s window",
        "zero_cm_ticks": "excluded from the window average; not replaced with zero",
        "ci_level": CI_LEVEL,
        "ci_method": args.ci_method,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_block_windows": args.bootstrap_block_windows,
        "uncertainty_caveat": "Starts overlap by 100 seconds within each SUMO trace. "
                               "The CI is an approximate within-trace interval, "
                               "not a replication CI across independent SUMO traces.",
        "random_seed": SEED,
        "traffic_root": str(traffic_root),
        "save_tick_series": bool(args.save_tick_series),
    }
    signature_keys = tuple(k for k in metadata if k != "uncertainty_caveat")
    payload = {"metadata": metadata, "runs": [], "summary": []}
    if args.resume and json_path.exists():
        with json_path.open("r", encoding="utf-8") as source:
            previous = json.load(source)
        old = previous.get("metadata", {})
        if any(old.get(k) != metadata[k] for k in signature_keys):
            raise ValueError("Cannot resume: saved experiment metadata differs from current settings.")
        payload = previous

    completed = {(r["density"], r["transmission_range_m"], r["h_max"], r["start_time"])
                 for r in payload["runs"]}
    total_runs = len(DATASETS) * len(TRANSMISSION_RANGES) * (len(H_MAX_VALUES) - 1) * len(WINDOW_STARTS)
    done = len(completed)
    print(f"MMZCA elbow sweep: {total_runs} valid 200-s windows "
          f"(+ h_max=0, recorded as invalid).")
    print("Only the CM-to-CH average hop metric will be computed and saved.")

    for density, _ in DATASETS:
        docs = load_dataset(traffic_root, density)
        template = copy.copy(base_config)
        for key, value in docs.items():
            setattr(template, key, value)
        zones = make_zones(template)

        for tr in TRANSMISSION_RANGES:
            for h in H_MAX_VALUES:
                h = int(h)
                if h == 0:
                    continue
                for window_start in WINDOW_STARTS:
                    window_start = int(window_start)
                    case_key = (density, int(tr), h, window_start)
                    if case_key in completed:
                        continue
                    cfg = copy.copy(template)
                    cfg.start_time = window_start
                    cfg.iter = WINDOW_SECONDS
                    cfg.h_max = h
                    cfg.veh_trans_range = int(tr)
                    cfg.bus_trans_range = int(tr)
                    # This radius is defined from TR and H_max in the MMZCA code.
                    if hasattr(cfg, "kernel_max_edge_range_m"):
                        cfg.kernel_max_edge_range_m = float(max(cfg.veh_trans_range, cfg.bus_trans_range))

                    seed = stable_seed(density, tr, window_start)
                    random.seed(seed)
                    np.random.seed(seed)
                    window_result = run_one_window(
                        cfg, zones, h, window_start, args.save_tick_series,
                    )
                    window_result.update(
                        density=density, transmission_range_m=int(tr), h_max=h,
                    )
                    payload["runs"].append(window_result)
                    completed.add(case_key)
                    done += 1
                    if done % args.progress_every == 0 or done == total_runs:
                        print(f"[{done}/{total_runs}] density={density}, TR={tr} m, "
                              f"h_max={h}, start={window_start}: "
                              f"mean_hops={window_result['average_hops']}", flush=True)
                    # Preserve each finished window for --resume. Recomputing
                    # thousands of bootstrap CIs after EVERY run would itself
                    # make the experiment extremely slow.
                    write_json_atomically(json_path, payload)
                    del cfg
                payload["summary"] = produce_summary(
                    payload["runs"], args.ci_method,
                    args.bootstrap_block_windows, args.bootstrap_replicates,
                )
                write_json_atomically(json_path, payload)
                gc.collect()

        del zones, docs, template
        gc.collect()

    payload["summary"] = produce_summary(
        payload["runs"], args.ci_method, args.bootstrap_block_windows,
        args.bootstrap_replicates,
    )
    payload["metadata"]["last_completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomically(json_path, payload)
    plot_summary(payload["summary"], plot_path, show=not args.no_show)
    print(f"JSON: {json_path}")
    print(f"Figure: {plot_path}")


if __name__ == "__main__":
    main()
