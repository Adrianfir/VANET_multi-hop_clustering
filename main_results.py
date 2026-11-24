"""MMZCA final-results runner (sequential, no multiprocessing).

Replaces main.py for the *clustering-only* MMZCA experiment. The tick stages
remain: update -> update_cluster -> stand_alones_cluster -> update_kernels ->
record_mmzca_stability -> update_tvct. No routing or per-tick display.

Experiment defaults match the supplied launcher.py:
  datasets small/mid/large; TR 100/200/300 m; 20 runs per dataset/TR;
  250 updates each; start indices 300, 452, ..., 3188.

Outputs in simulation_results/MMZCA_final_Hmax<H>_250ticks/:
  results_<dataset>_TR<tr>_start<start>.csv    one flat, launcher-like CSV per run
  all_results.csv                              combined, 1 row per run
  summary_ci.csv                               long: mean and 95% CI per metric
  summary_ci_wide.csv                          wide: <metric>_mean/_ci95_* columns
  experiment_metadata.json                     settings/definitions
  failed_runs.csv                              failed simulations, if any

RCAP (Root-Connected Affiliation Persistence) is computed here, separately
from VCSM. For a fixed horizon of T_s successive 1-second intervals, its
denominator counts all vehicle-time sample ends with continuous in-area presence
through [t-T_s, t]. Its numerator counts sample ends at which that vehicle was
a valid, physically root-connected CM of the same root *incarnation* on every
committed tick in that interval. Same-root parent switches are allowed, while
root changes, gaps, or CH/SA episodes reset persistence. Default T_s=20 s;
set --rcap-horizon to choose and report the value used for publication.

The 20 windows overlap (250 ticks, 152-tick stride). The default block
bootstrap produces approximate *within-trace* intervals, not confidence
intervals across independent SUMO traffic seeds/traces.

Examples:
  python main.py --h-max 4
  python main.py --h-max 4 --rcap-horizon 30 --resume
  python main.py --h-max 4 --datasets mid --transmission-ranges 300 --num-runs 1
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
import xml.dom.minidom
from collections import defaultdict
from pathlib import Path

import numpy as np


DATASET_SUFFIXES = {"small": "smallsize", "mid": "midsize", "large": "largesize"}
DEFAULT_TRANSMISSION_RANGES = (100, 200, 300)
DEFAULT_RUN_LENGTH = 200
DEFAULT_NUMBER_OF_RUNS = 27
DEFAULT_FIRST_START = 400
DEFAULT_START_STEP = 100
DEFAULT_RCAP_HORIZON = 20  # seconds, chosen for this experiment; configurable
DEFAULT_CI_LEVEL = 0.95
DEFAULT_BOOTSTRAP_REPS = 2000
DEFAULT_BOOTSTRAP_BLOCK = 3  # overlapping, chronologically ordered windows
DEFAULT_RANDOM_SEED = 42

VCSMR_FIELDS = (
    "vcsm_r", "AR", "average_R", "CCR", "avg_L_CM", "ECHR",
    "avg_ch_lifetime_role", "avg_ch_lifetime_effective",
    "num_ch_role_episodes", "num_ch_effective_episodes", "num_cm_episodes",
    "total_ch_role_time", "total_ch_effective_time", "total_ch_role_time_raw",
    "total_empty_ch_time", "total_presence", "total_clustered_time", "total_switches",
)
VCSMCM_FIELDS = (
    "vcsm_cm", "average_R_CM", "average_R_CM_weighted", "AR_CM_conditional",
    "CCR_CM", "num_cm_vehicles", "total_A_CM", "total_P_CM_vehicles",
    "total_switches_CM",
)
# Keep the exact *flat* VCSM field names produced by the supplied launcher.
GENERAL_FIELDS = (
    "dataset", "veh_trans_range", "start_time", "clustering_algorithm",
    "routing_algorithm", "h_max", "run_length", "rcap_horizon_seconds",
    "n_vehs", "n_buses", "avg_chs", "avg_stand_alones", "execution_time",
    "average_hops", "ticks_with_cms", "cm_tick_observations",
    "avg_ch_percentage", "avg_sa_percentage", "avg_cm_count",
    "mmzca_same_root_parent_switches", "same_root_parent_switches_per_second",
    "stability_evaluation", "RCAP", "RCAP_numerator", "RCAP_denominator",
)
RESULT_FIELDS = GENERAL_FIELDS + VCSMR_FIELDS + VCSMCM_FIELDS
NON_PERFORMANCE_FIELDS = {
    "dataset", "veh_trans_range", "start_time", "clustering_algorithm",
    "routing_algorithm", "h_max", "run_length", "rcap_horizon_seconds",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traffic-root", type=Path, default=None,
                   help="Parent directory containing final_data_Richmondhill_* folders")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write individual and combined CSVs")
    p.add_argument("--datasets", nargs="+", choices=tuple(DATASET_SUFFIXES),
                   default=list(DATASET_SUFFIXES))
    p.add_argument("--transmission-ranges", type=int, nargs="+",
                   default=list(DEFAULT_TRANSMISSION_RANGES))
    p.add_argument("--run-length", type=int, default=DEFAULT_RUN_LENGTH)
    p.add_argument("--num-runs", type=int, default=DEFAULT_NUMBER_OF_RUNS)
    p.add_argument("--first-start", type=int, default=DEFAULT_FIRST_START)
    p.add_argument("--start-step", type=int, default=DEFAULT_START_STEP)
    p.add_argument("--h-max", type=int, default=None,
                   help="Fixed MMZCA H_max (default: current options/constants.py value)")
    p.add_argument("--rcap-horizon", type=int, default=DEFAULT_RCAP_HORIZON,
                   help="Fixed root-affiliation persistence horizon in seconds (default 20)")
    p.add_argument("--ci-method", choices=("block-bootstrap", "t"), default="block-bootstrap")
    p.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    p.add_argument("--bootstrap-block-windows", type=int, default=DEFAULT_BOOTSTRAP_BLOCK)
    p.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--resume", action="store_true",
                   help="Reuse valid prior per-run CSVs with matching experiment settings")
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args()


def load_base_config(traffic_root: Path):
    """Use the original Configs implementation, with no custom CLI args leaked.

    Inputs() eagerly parses the default smallsize XML. For an external traffic
    root, temporarily redirect just those three XML filename lookups.
    """
    from configs.config import Configs

    original_args = sys.argv
    original_parse = xml.dom.minidom.parse
    small_folder = traffic_root / "final_data_Richmondhill_smallsize"
    redirects = {
        "sumotrace_no_bus_and_rsu.xml": small_folder / "sumoTrace_no_bus_and_rsu.xml",
        "osm.net.xml": small_folder / "osm.net.xml",
        "osm_bbox.osm.xml": small_folder / "osm_bbox.osm.xml",
        "osm.nodes.xml": small_folder / "osm_bbox.osm.xml",
    }

    def parse_with_override(filename, *args, **kwargs):
        if isinstance(filename, (str, os.PathLike)):
            key = Path(filename).name.lower()
            if key in redirects:
                filename = str(redirects[key])
        return original_parse(filename, *args, **kwargs)

    try:
        sys.argv = [original_args[0]]
        xml.dom.minidom.parse = parse_with_override
        return Configs().config
    finally:
        xml.dom.minidom.parse = original_parse
        sys.argv = original_args


def load_traffic_documents(root: Path, suffix: str, last_required_tick: int):
    folder = root / ("final_data_Richmondhill_" + suffix)
    paths = {
        "sumo_trace": folder / "sumoTrace_no_bus_and_rsu.xml",
        "sumo_edge": folder / "osm.net.xml",
        "sumo_node": folder / "osm_bbox.osm.xml",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing: {path}")
    docs = {key: xml.dom.minidom.parse(str(path)) for key, path in paths.items()}
    steps = docs["sumo_trace"].documentElement.getElementsByTagName("timestep")
    if last_required_tick >= len(steps):
        raise ValueError(f"{suffix} trace contains {len(steps)} timestep nodes, "
                         f"but index {last_required_tick} is required")
    docs["fcd"] = docs["sumo_trace"].documentElement
    docs["times"] = steps
    return docs


def make_zones(cfg):
    from zonex import ZoneID
    zones = {}
    for name, mode in (("micro", "mic"), ("meso", "mes"), ("macro", "mac")):
        z = ZoneID(cfg, zoning_mode=mode)
        z.zones()
        zones[name] = z
    return zones


def fixed_seed(base_seed, dataset, tr, start):
    stamp = f"{base_seed}:{dataset}:{tr}:{start}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(stamp, digest_size=4).digest(), "big")


class RCAPTracker:
    """Sliding fixed-horizon RCAP using per-vehicle O(1) streak bookkeeping.

    At each committed tick a vehicle is eligible if it was present at all
    T_s+1 observed tick boundaries from t-T_s through t (one-second SUMO ticks).
    Success additionally requires CM status, a valid committed physical parent
    chain and the same (root identifier, root incarnation) at every boundary.
    """

    def __init__(self, horizon: int):
        if horizon < 1:
            raise ValueError("RCAP horizon must be >= 1 second")
        self.horizon = horizon
        self.last = {}  # veh_id -> (last_tick, presence_streak, cm_streak, root_incarnation)
        self.numerator = 0
        self.denominator = 0
        self.zero_epoch_count = 0

    def observe(self, cluster, config):
        path_check = getattr(cluster, "_mmzca_valid_committed_path", None)
        if not callable(path_check):
            raise RuntimeError("RCAP needs DataTable._mmzca_valid_committed_path() "
                               "from your MMZCA implementation. Please provide the "
                               "current data_cluster.py before using RCAP.")

        now = int(cluster.time)
        next_states = {}
        for vid in cluster.veh_table.ids():
            st = cluster.veh_table.values(vid)
            if not st or not st.get("in_area", False):
                continue

            previous = self.last.get(vid)
            consecutive = previous is not None and previous[0] == now - 1
            presence_streak = (previous[1] + 1) if consecutive else 1

            root_key = None
            root_id = st.get("root_ch") or st.get("primary_ch")
            if st.get("cluster_head") is not True and root_id is not None and \
                    st.get("root_connection_valid") is True and path_check(vid, config):
                root_st = (cluster.veh_table.values(root_id)
                           or cluster.bus_table.values(root_id))
                if root_st is not None:
                    epoch = root_st.get("kernel_root_incarnation")
                    if epoch is None:
                        # Without an incarnation token, persistence through a
                        # CH demotion/re-promotion cannot be verified correctly.
                        self.zero_epoch_count += 1
                    else:
                        root_key = (root_id, epoch)

            cm_streak = (previous[2] + 1
                         if consecutive and root_key is not None and previous[3] == root_key
                         else (1 if root_key is not None else 0))
            next_states[vid] = (now, presence_streak, cm_streak, root_key)

            if presence_streak >= self.horizon + 1:
                self.denominator += 1
                if cm_streak >= self.horizon + 1:
                    self.numerator += 1

        self.last = next_states  # absent/outside vehicles cannot bridge a gap

    def result(self):
        if self.zero_epoch_count:
            raise RuntimeError(
                f"RCAP could not verify root incarnation on {self.zero_epoch_count} "
                "valid CM observations. Fix/enable kernel_root_incarnation "
                "tracking in data_cluster.py before reporting RCAP."
            )
        return (self.numerator / self.denominator if self.denominator else None,
                self.numerator, self.denominator)


def tick_metrics(cluster, cfg):
    """Collect role counts/percentages and physical CM->root hop depth.

    Average hop depth is *not routing hop count*. Every in-area ordinary CM
    must have a committed hop_count in [1, cfg.h_max].
    """
    n_area = ch_count = sa_count = cm_count = hop_total = 0
    for vid in cluster.veh_table.ids():
        st = cluster.veh_table.values(vid)
        if not st or not st.get("in_area", False):
            continue
        n_area += 1
        if st.get("cluster_head") is True:
            ch_count += 1
            continue
        root = st.get("root_ch") or st.get("primary_ch")
        if root is None:
            sa_count += 1
            continue
        depth = st.get("hop_count")
        if (isinstance(depth, bool) or not isinstance(depth, (int, np.integer))
                or not 1 <= int(depth) <= int(cfg.h_max)):
            raise RuntimeError(f"Tick {cluster.time} CM {vid}: inconsistent "
                               f"hop_count={depth!r}, h_max={cfg.h_max}")
        if st.get("root_connection_valid") is False:
            raise RuntimeError(f"Tick {cluster.time} CM {vid}: invalid root connection")
        cm_count += 1
        hop_total += int(depth)
    return {
        "average_hops": hop_total / cm_count if cm_count else None,
        "cm_count": cm_count,
        "ch_percent": 100.0 * ch_count / n_area if n_area else None,
        "sa_percent": 100.0 * sa_count / n_area if n_area else None,
    }


def extract_required_metric_fields(dictionary, fields, name):
    if not isinstance(dictionary, dict):
        raise TypeError(f"{name} expected a dictionary, got {type(dictionary).__name__}")
    missing = [key for key in fields if key not in dictionary]
    if missing:
        raise KeyError(f"{name} missing fields {missing}; available: {list(dictionary)}")
    return {key: dictionary[key] for key in fields}


def run_one_job(cfg, dataset, tr, start, rcap_horizon, progress_every):
    """One original MMZCA run, returning flat launcher-compatible results."""
    from data_cluster import DataTable

    # Fresh zones and fresh DataTable are essential: membership and kernel
    # state must never leak between independent observation windows.
    zones = make_zones(cfg)
    cluster = DataTable(cfg, zones)
    rcap = RCAPTracker(rcap_horizon)
    rcap.observe(cluster, cfg)  # include the initial SUMO boundary in [t-T_s,t]
    ch_counts, sa_counts, cm_counts = [], [], []
    ch_pcts, sa_pcts, hop_means = [], [], []
    cm_observations = 0
    begin = time.perf_counter()

    for i in range(int(cfg.iter)):
        # The same committed-state transition sequence as the uploaded main.py.
        pre_states = cluster._snapshot_tvct_states()
        cluster.update(cfg)
        cluster.update_cluster(cluster.veh_table.ids(), cfg)
        cluster.stand_alones_cluster(cfg)
        cluster.update_kernels(cfg)
        cluster.record_mmzca_stability()
        cluster.update_tvct(pre_states)

        observations = tick_metrics(cluster, cfg)
        ch_counts.append(len(cluster.all_chs))
        sa_counts.append(len(cluster.stand_alone))
        cm_counts.append(observations["cm_count"])
        if observations["ch_percent"] is not None:
            ch_pcts.append(observations["ch_percent"])
            sa_pcts.append(observations["sa_percent"])
        if observations["average_hops"] is not None:
            hop_means.append(observations["average_hops"])
            cm_observations += observations["cm_count"]

        rcap.observe(cluster, cfg)
        if (i + 1) % progress_every == 0 or (i + 1) == cfg.iter:
            print(f"    tick {i + 1}/{cfg.iter} (SUMO index {cluster.time})", flush=True)

    execution_time = time.perf_counter() - begin
    if hasattr(cluster, "_mmzca_stability_summary"):
        stability = cluster._mmzca_stability_summary()
        base_vcsm = stability["vcsm"]
        vcsm_r = stability["vcsm_r"]
        vcsm_cm = stability["vcsm_cm"]
    else:
        base_vcsm = cluster.vcsm(cfg)
        vcsm_r = cluster.vcsm_r(cfg)
        vcsm_cm = cluster.vcsm_cm(cfg)

    rc_value, rc_numerator, rc_denominator = rcap.result()
    result = {
        "dataset": dataset,
        "veh_trans_range": tr,
        "start_time": start,
        "clustering_algorithm": "MMZCA",
        "routing_algorithm": "None",
        "h_max": cfg.h_max,
        "run_length": cfg.iter,
        "rcap_horizon_seconds": rcap_horizon,
        "n_vehs": len(cluster.veh_table.ids()),
        "n_buses": len(cluster.bus_table.ids()),
        "avg_chs": float(np.mean(ch_counts)) if ch_counts else None,
        "avg_stand_alones": float(np.mean(sa_counts)) if sa_counts else None,
        "execution_time": execution_time,
        "average_hops": float(np.mean(hop_means)) if hop_means else None,
        "ticks_with_cms": len(hop_means),
        "cm_tick_observations": cm_observations,
        "avg_ch_percentage": float(np.mean(ch_pcts)) if ch_pcts else None,
        "avg_sa_percentage": float(np.mean(sa_pcts)) if sa_pcts else None,
        "avg_cm_count": float(np.mean(cm_counts)) if cm_counts else None,
        "mmzca_same_root_parent_switches": int(getattr(cluster, "mmzca_parent_switches", 0)),
        "same_root_parent_switches_per_second": (
            float(getattr(cluster, "mmzca_parent_switches", 0)) / float(cfg.iter)
        ),
        "stability_evaluation": base_vcsm,
        "RCAP": rc_value,
        "RCAP_numerator": rc_numerator,
        "RCAP_denominator": rc_denominator,
    }
    result.update(extract_required_metric_fields(vcsm_r, VCSMR_FIELDS, "VCSM_R"))
    result.update(extract_required_metric_fields(vcsm_cm, VCSMCM_FIELDS, "VCSM_CM"))
    if set(result) != set(RESULT_FIELDS):
        raise RuntimeError(f"Result columns mismatch: {set(result) ^ set(RESULT_FIELDS)}")
    del cluster
    return result


def write_csv_atomically(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_json_atomically(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
    tmp.replace(path)


def result_path(output_dir, dataset, tr, start):
    return output_dir / f"results_{dataset}_TR{tr}_start{start}.csv"


def load_existing_result(path: Path, expected):
    if not path.is_file():
        return None
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            if rdr.fieldnames != list(RESULT_FIELDS):
                return None
            records = list(rdr)
        if len(records) != 1:
            return None
        row = records[0]
        for col, value in expected.items():
            if str(row.get(col)) != str(value):
                return None
        for col in RESULT_FIELDS:
            if col in NON_PERFORMANCE_FIELDS:
                continue
            value = row.get(col)
            if value in (None, ""):
                if col == "RCAP" and float(row.get("RCAP_denominator", "0")) == 0:
                    continue
                if col == "average_hops" and float(row.get("ticks_with_cms", "0")) == 0:
                    continue
                return None
            if not math.isfinite(float(value)):
                return None
        return row
    except (OSError, TypeError, ValueError, KeyError):
        return None


def bootstrap_interval(values, method, level, reps, block_windows, seed):
    """CI for across-window mean of finite values in start-time order."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    n = len(data)
    if n < 2:
        return (None, None)
    if np.all(data == data[0]):
        return (float(data[0]), float(data[0]))
    if method == "t":
        from scipy.stats import t
        half = float(t.ppf((1 + level) / 2, df=n - 1)) * data.std(ddof=1) / np.sqrt(n)
        return (float(data.mean() - half), float(data.mean() + half))
    rng = np.random.default_rng(seed)
    b = min(block_windows, n)
    starts = rng.integers(0, n, size=(reps, math.ceil(n / b)))
    idx = ((starts[:, :, None] + np.arange(b)[None, None, :]) % n)
    idx = idx.reshape(reps, -1)[:, :n]
    bootstrap_means = data[idx].mean(axis=1)
    lo, hi = np.quantile(bootstrap_means, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def summarize_results(rows, ci_method, ci_level, reps, block_windows, seed):
    """Create long and wide CIs across runs, one (dataset,TR) group at a time."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], int(float(row["veh_trans_range"])))].append(row)
    long_records, wide_records = [], []
    metric_fields = [key for key in RESULT_FIELDS if key not in NON_PERFORMANCE_FIELDS]
    for (dataset, tr), group in sorted(groups.items(), key=lambda item:
                                        (list(DATASET_SUFFIXES).index(item[0][0]), item[0][1])):
        group.sort(key=lambda row: int(float(row["start_time"])))
        wide = {"dataset": dataset, "veh_trans_range": tr,
                "h_max": int(float(group[0]["h_max"])), "n_runs": len(group)}
        for metric in metric_fields:
            raw = [row.get(metric) for row in group]
            values = [float(v) for v in raw if v not in (None, "")
                      and math.isfinite(float(v))]
            lo, hi = bootstrap_interval(values, ci_method, ci_level, reps,
                                        block_windows, fixed_seed(seed, dataset + ":" + metric, tr, 0))
            mean = float(np.mean(values)) if values else None
            long_records.append({
                "dataset": dataset, "veh_trans_range": tr,
                "h_max": int(float(group[0]["h_max"])), "metric": metric,
                "mean": mean, "ci95_lower": lo, "ci95_upper": hi,
                "n_runs": len(values),
            })
            wide[metric + "_mean"] = mean
            wide[metric + "_ci95_lower"] = lo
            wide[metric + "_ci95_upper"] = hi
            wide[metric + "_n_runs"] = len(values)
        wide_records.append(wide)
    return long_records, wide_records


def save_combined_and_cis(output_dir, results, args):
    if not results:
        return
    sort_key = lambda r: (list(DATASET_SUFFIXES).index(r["dataset"]),
                          int(float(r["veh_trans_range"])), int(float(r["start_time"])))
    results = sorted(results, key=sort_key)
    write_csv_atomically(output_dir / "all_results.csv", RESULT_FIELDS, results)
    long, wide = summarize_results(results, args.ci_method, DEFAULT_CI_LEVEL,
                                   args.bootstrap_replicates, args.bootstrap_block_windows, args.seed)
    write_csv_atomically(output_dir / "summary_ci.csv",
                         ("dataset", "veh_trans_range", "h_max", "metric", "mean",
                          "ci95_lower", "ci95_upper", "n_runs"), long)
    if wide:
        write_csv_atomically(output_dir / "summary_ci_wide.csv", tuple(wide[0]), wide)


def fixed_configuration_snapshot(config):
    """Record non-swept MMZCA parameters so --resume cannot mix settings."""
    excluded = {"sumo_trace", "sumo_edge", "sumo_node", "fcd", "times",
                "veh_trans_range", "start_time", "iter", "h_max"}
    snapshot = {}
    for name, value in sorted(vars(config).items()):
        if name in excluded:
            continue
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, Path):
            value = str(value)
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Cannot record fixed config parameter {name}: "
                            f"{type(value).__name__}") from exc
        snapshot[name] = value
    return snapshot


def experiment_settings(args, traffic_root, hmax, base_config):
    return {
        "algorithm": "MMZCA", "clustering_only": True, "sequential": True,
        "datasets": args.datasets, "transmission_ranges_m": args.transmission_ranges,
        "run_length": args.run_length, "num_runs": args.num_runs,
        "first_start": args.first_start, "start_step": args.start_step,
        "start_times": [args.first_start + i * args.start_step for i in range(args.num_runs)],
        "h_max": hmax, "rcap_horizon_seconds": args.rcap_horizon,
        "ci_level": DEFAULT_CI_LEVEL, "ci_method": args.ci_method,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_block_windows": args.bootstrap_block_windows,
        "random_seed": args.seed, "traffic_root": str(traffic_root),
        "fixed_config": fixed_configuration_snapshot(base_config),
        "result_columns": list(RESULT_FIELDS),
        "average_hops_definition": "Mean over ticks with CMs of the tick-wise mean committed CM-to-root hop_count, not packet routing hops",
        "avg_ch_percentage_definition": "Mean over ticks with in-area ordinary vehicles of 100*in-area CHs/in-area ordinary vehicles",
        "avg_sa_percentage_definition": "Mean over ticks with in-area ordinary vehicles of 100*in-area SAs/in-area ordinary vehicles",
        "RCAP_definition": "At time t, eligible iff present in-area at every committed tick in [t-Ts,t]; success iff valid physically root-connected CM under the same root ID and root incarnation at every such tick, permitting same-root parent changes",
        "n_vehs_definition": "Active vehicles at the final tick, as in launcher.py",
        "ci_caveat": "Starts overlap (250 ticks with 152-tick stride). Block bootstrap CIs describe within-trace variability, not independent-seed replication uncertainty.",
    }


def main():
    args = parse_args()
    if not args.datasets or not args.transmission_ranges:
        raise ValueError("Choose at least one dataset and transmission range")
    if min(args.transmission_ranges) <= 0:
        raise ValueError("Transmission ranges must be positive")
    if args.run_length < 2 or args.num_runs < 1 or args.first_start < 0 or args.start_step < 1:
        raise ValueError("Invalid run-length, num-runs, start-time or start-step")
    if not 1 <= args.rcap_horizon < args.run_length:
        raise ValueError("RCAP horizon must be >=1 and shorter than run length")
    if args.bootstrap_replicates < 100 or args.bootstrap_block_windows < 1:
        raise ValueError("Require at least 100 bootstrap replicates and block length >=1")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be >=1")

    project_root = Path(__file__).resolve().parent
    traffic_root = (args.traffic_root if args.traffic_root else Path(
        os.environ.get("VANET_TRAFFIC_DATA", str(project_root.parent / "traffic_data"))))
    traffic_root = traffic_root.expanduser().resolve()
    base = load_base_config(traffic_root)
    hmax = int(args.h_max if args.h_max is not None else base.h_max)
    if hmax < 1:
        raise ValueError("h_max must be >=1")
    output_dir = (args.output_dir if args.output_dir else
                  project_root / "simulation_results" /
                  f"MMZCA_final_Hmax{hmax}_{args.run_length}ticks")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = experiment_settings(args, traffic_root, hmax, base)
    settings_file = output_dir / "experiment_metadata.json"
    if settings_file.exists():
        with settings_file.open("r", encoding="utf-8") as f:
            previous = json.load(f)
        if previous != settings:
            raise ValueError("Output folder belongs to a different experiment. "
                             "Use --output-dir for the new settings. Do not mix CSVs.")
        if not args.resume and list(output_dir.glob("results_*_TR*_start*.csv")):
            raise ValueError("Previous results exist. Use --resume or a fresh --output-dir.")
    else:
        if list(output_dir.glob("results_*_TR*_start*.csv")):
            raise ValueError("Result files exist without metadata. Use an empty --output-dir.")
        write_json_atomically(settings_file, settings)

    starts = settings["start_times"]
    last_tick = max(starts) + args.run_length
    completed, failed = [], []
    jobs = [(dataset, tr, start) for dataset in args.datasets
            for tr in args.transmission_ranges for start in starts]
    print(f"MMZCA FINAL / sequential; H_max={hmax}; RCAP horizon={args.rcap_horizon}s")
    print(f"Datasets={args.datasets}, TR={args.transmission_ranges}, "
          f"windows={len(starts)}, updates/window={args.run_length}; jobs={len(jobs)}")
    print(f"Results: {output_dir}", flush=True)

    count = 0
    for dataset in args.datasets:
        docs = load_traffic_documents(traffic_root, DATASET_SUFFIXES[dataset], last_tick)
        template = copy.copy(base)
        for name, obj in docs.items():
            setattr(template, name, obj)
        for tr in args.transmission_ranges:
            for start in starts:
                count += 1
                out_path = result_path(output_dir, dataset, tr, start)
                expected = {"dataset": dataset, "veh_trans_range": tr,
                            "start_time": start, "h_max": hmax,
                            "run_length": args.run_length,
                            "rcap_horizon_seconds": args.rcap_horizon,
                            "clustering_algorithm": "MMZCA", "routing_algorithm": "None"}
                existing = load_existing_result(out_path, expected) if args.resume else None
                if existing is not None:
                    completed.append(existing)
                    print(f"[{count}/{len(jobs)}] SKIP {dataset} TR={tr} start={start}", flush=True)
                    continue
                cfg = copy.copy(template)
                cfg.veh_trans_range = tr  # launcher updates vehicle TR only
                cfg.h_max = hmax
                cfg.iter = args.run_length
                cfg.start_time = start
                if hasattr(cfg, "kernel_max_edge_range_m"):
                    cfg.kernel_max_edge_range_m = max(float(cfg.veh_trans_range),
                                                      float(cfg.bus_trans_range))
                trial_seed = fixed_seed(args.seed, dataset, tr, start)
                random.seed(trial_seed)
                np.random.seed(trial_seed)
                print(f"[{count}/{len(jobs)}] RUN {dataset} TR={tr} "
                      f"start={start} H_max={hmax}", flush=True)
                try:
                    result = run_one_job(cfg, dataset, tr, start,
                                         args.rcap_horizon, args.progress_every)
                    write_csv_atomically(out_path, RESULT_FIELDS, [result])
                    completed.append(result)
                    save_combined_and_cis(output_dir, completed, args)
                    print(f"  OK: AR={float(result['AR']):.4f}, "
                          f"VCSM_R={float(result['vcsm_r']):.4f}, "
                          f"RCAP={result['RCAP']}, "
                          f"avg_CH={float(result['avg_chs']):.2f}", flush=True)
                except Exception as exc:
                    failed.append({"dataset": dataset, "veh_trans_range": tr,
                                   "start_time": start, "error": repr(exc)})
                    write_csv_atomically(output_dir / "failed_runs.csv",
                                         ("dataset", "veh_trans_range", "start_time", "error"), failed)
                    print(f"  FAILED: {exc}\n{traceback.format_exc()}", flush=True)
                    if args.stop_on_error:
                        raise
                finally:
                    del cfg
                    gc.collect()
        del docs, template
        gc.collect()
    if completed:
        save_combined_and_cis(output_dir, completed, args)
    if not failed:
        (output_dir / "failed_runs.csv").unlink(missing_ok=True)
    print(f"Finished: completed={len(completed)}/{len(jobs)}, failed={len(failed)}")
    print(f"Combined CSV: {output_dir / 'all_results.csv'}")
    print(f"Mean + 95% CI: {output_dir / 'summary_ci.csv'}")


if __name__ == "__main__":
    main()
