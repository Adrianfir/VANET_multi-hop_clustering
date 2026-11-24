"""DHCV final-results runner (sequential, no multiprocessing).

Replaces main.py for the *clustering-only* DHCV experiment in the framework
containing DataTable._dhcv_update_cluster() and DHCV options in
options/constants.py. Does not change the DHCV clustering logic.

Original tick order is preserved:
    update -> update_cluster -> stand_alones_cluster -> update_kernels ->
    record_mmzca_stability -> update_tvct.
For DHCV, the two middle MMZCA-only maintenance calls are no-ops.

Defaults match the final MMZCA experiment: small/mid/large datasets,
TR 100/200/300 m, 28 sequential 250-tick windows per dataset/TR,
starts 300, 400, ..., 3000, inclusive (252 simulations in total).

Outputs in simulation_results/DHCV_final_D<D>_250ticks/:
  results_<dataset>_TR<tr>_start<start>.csv  individual flat CSV
  all_results.csv                            all successful windows
  summary_ci.csv                             mean and 95% CI by metric
  summary_ci_wide.csv                        <metric>_mean / _ci95_lower / _ci95_upper
  experiment_metadata.json                   exact settings and metric definitions
  failed_runs.csv                            failures, if any

RCAP is computed from DHCV's observed committed states. Its own measurement-
only root-incarnation registry increments when a vehicle newly becomes a CH;
this is necessary because DHCV does not maintain MMZCA kernel incarnations.
A physically broken CM->root path is not counted as a valid affiliation,
even during DHCV's VI grace period. Same-root parent changes are permitted.
This measurement does not modify DHCV state or its election policy.

The fixed D-hop bound for DHCV is --dhcv-d (synced to config.h_max for
metric validation). RI and VI defaults come from the project's config.

Overlapping windows produce approximate within-trace block-bootstrap CIs,
not independent-traffic-seed CIs.

Examples:
  python main.py --dhcv-d 3
  python main.py --dhcv-d 3 --dhcv-ri 150 --dhcv-vi 10 --resume
  python main.py --dhcv-d 3 --datasets mid --transmission-ranges 300 --num-runs 1
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
DEFAULT_FIRST_START = 400
DEFAULT_LAST_START = 3000  # inclusive final window start
DEFAULT_START_STEP = 100
DEFAULT_NUMBER_OF_RUNS = (DEFAULT_LAST_START - DEFAULT_FIRST_START) // DEFAULT_START_STEP + 1
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
    "dhcv_same_root_parent_switches", "same_root_parent_switches_per_second",
    "dhcv_renewals", "dhcv_vi_recoveries", "dhcv_ch_to_cm_changes",
    "avg_tvct", "total_tvct", "avg_valid_cm_count", "avg_invalid_cm_count",
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
    p.add_argument("--dhcv-d", type=int, default=None,
                   help="DHCV maximum cluster depth D (default: options/constants.py dhcv_d)")
    p.add_argument("--dhcv-ri", type=int, default=None,
                   help="DHCV network renewal interval RI (default: project configuration)")
    p.add_argument("--dhcv-vi", type=int, default=None,
                   help="DHCV verification/grace interval VI (default: project configuration)")
    p.add_argument("--dhcv-threshold-mode", choices=("normalized", "literal"), default=None,
                   help="DHCV relative-mobility threshold: project default unless overridden")
    p.add_argument("--dhcv-isolated-policy", choices=("sa", "paper_ch"), default=None,
                   help="DHCV treatment of isolated vehicles: project default unless overridden")
    p.add_argument("--rcap-horizon", type=int, default=DEFAULT_RCAP_HORIZON,
                   help="Fixed RCAP horizon in seconds (default 20)")
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


class DHCVRCAPTracker:
    """Manuscript RCAP for DHCV, using a measurement-only root epoch registry.

    A root's epoch changes on CH demotion->promotion, including after an absence.
    A stable CH survives RI renewals without a new incarnation. A parent change
    within the same root is permitted if each committed path remains feasible.
    The sample boundaries are [t-T_s, ..., t], requiring T_s+1 consecutive ticks.

    Since DHCV does not propagate kernel_root_incarnation, it MUST NOT use the
    MMZCA RCAP implementation's node/kernel epochs.
    """

    def __init__(self, horizon: int):
        if horizon < 1:
            raise ValueError("RCAP horizon must be >= 1")
        self.horizon = horizon
        self.last = {}   # vid -> (tick, presence_streak, cm_streak, root_key, parent)
        self.active_roots = set()
        self.root_epochs = {}
        self.numerator = 0
        self.denominator = 0

    @staticmethod
    def _valid_path(cluster, cfg, vid, root_id):
        """Check the physical committed forest and in-area status of EVERY node."""
        check = getattr(cluster, "_mmzca_valid_committed_path", None)
        if not callable(check):
            raise RuntimeError("DHCV baseline needs DataTable._mmzca_valid_committed_path")
        if not check(vid, cfg):
            return False
        # This extra check excludes a path routed through an out-of-area node.
        seen = set()
        node_id = vid
        while node_id not in seen:
            seen.add(node_id)
            st = cluster.veh_table.values(node_id) or cluster.bus_table.values(node_id)
            if st is None or not st.get("in_area", False):
                return False
            if node_id == root_id:
                return st.get("cluster_head") is True
            node_id = (st.get("parent_node") or st.get("parent_id") or
                       st.get("secondary_ch") or st.get("primary_ch"))
            if node_id is None:
                return False
        return False

    def observe(self, cluster, cfg):
        now = int(cluster.time)
        roots = {vid for vid in cluster.veh_table.ids()
                 if (cluster.veh_table.values(vid).get("in_area", False)
                     and cluster.veh_table.values(vid).get("cluster_head") is True)}
        for root_id in roots - self.active_roots:
            self.root_epochs[root_id] = self.root_epochs.get(root_id, 0) + 1
        self.active_roots = roots

        next_states = {}
        for vid in cluster.veh_table.ids():
            st = cluster.veh_table.values(vid)
            if st is None or not st.get("in_area", False):
                continue
            previous = self.last.get(vid)
            consecutive = previous is not None and previous[0] == now - 1
            presence = previous[1] + 1 if consecutive else 1
            root_id = st.get("root_ch") or st.get("primary_ch")
            parent = (st.get("parent_node") or st.get("parent_id") or
                      st.get("secondary_ch") or st.get("primary_ch"))
            root_key = None
            if (st.get("cluster_head") is not True and root_id in roots
                    and st.get("root_connection_valid") is True
                    and self._valid_path(cluster, cfg, vid, root_id)):
                root_key = (root_id, self.root_epochs[root_id])

            cm_streak = (previous[2] + 1 if consecutive and root_key is not None
                         and previous[3] == root_key else
                         (1 if root_key is not None else 0))
            next_states[vid] = (now, presence, cm_streak, root_key, parent)
            if presence >= self.horizon + 1:
                self.denominator += 1
                if cm_streak >= self.horizon + 1:
                    self.numerator += 1
        self.last = next_states
        return {vid: (v[3], v[4]) for vid, v in next_states.items()
                if v[3] is not None}

    def result(self):
        return (self.numerator / self.denominator if self.denominator else None,
                self.numerator, self.denominator)


def tick_metrics(cluster, cfg):
    """Counts for every in-area ordinary vehicle; valid CM-root depth separately.

    DHCV may retain an invalid link during VI grace. Such CMs retain their
    structural CM role (for counts), but are excluded from *physically valid*
    hop-depth averages and RCAP successes.
    """
    n_area = ch_count = sa_count = cm_count = valid_cm = 0
    invalid_cm = 0
    hop_total = 0
    check = getattr(cluster, "_mmzca_valid_committed_path", None)
    if not callable(check):
        raise RuntimeError("DHCV implementation is missing committed-path validation")
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
        cm_count += 1
        depth = st.get("hop_count")
        if not isinstance(depth, (int, np.integer)) or isinstance(depth, bool) or not 1 <= depth <= int(cfg.dhcv_d):
            raise RuntimeError(f"Tick {cluster.time}: CM {vid} has invalid depth {depth!r} for DHCV D={cfg.dhcv_d}")
        if st.get("root_connection_valid") is True and DHCVRCAPTracker._valid_path(cluster, cfg, vid, root):
            valid_cm += 1
            hop_total += int(depth)
        else:
            invalid_cm += 1
    return {
        "average_hops": hop_total / valid_cm if valid_cm else None,
        "cm_count": cm_count, "valid_cm_count": valid_cm,
        "invalid_cm_count": invalid_cm,
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
    """One DHCV clustering-only observation window; preserve the user's tick order."""
    from data_cluster import DataTable

    if str(getattr(cfg, "algorithm", "")).upper() != "DHCV":
        raise ValueError("Config algorithm is not DHCV; cannot run this baseline as MMZCA")
    zones = make_zones(cfg)
    cluster = DataTable(cfg, zones)
    if not callable(getattr(cluster, "_dhcv_update_cluster", None)):
        raise RuntimeError("Your data_cluster.py lacks _dhcv_update_cluster; "
                           "use the DHCV implementation, not the MMZCA implementation")
    if cluster.bus_table.ids():
        raise ValueError("DHCV requires bus-free input trace; found bus nodes")

    rcap = DHCVRCAPTracker(rcap_horizon)
    previous_valid_cm = rcap.observe(cluster, cfg)  # initial committed boundary
    parent_switches = 0
    ch_counts, sa_counts, cm_counts = [], [], []
    valid_cm_counts, invalid_cm_counts = [], []
    ch_pcts, sa_pcts, hop_means = [], [], []
    cm_observations = 0
    begin = time.perf_counter()

    for i in range(int(cfg.iter)):
        pre_states = cluster._snapshot_tvct_states()
        cluster.update(cfg)
        cluster.update_cluster(cluster.veh_table.ids(), cfg)
        cluster.stand_alones_cluster(cfg)  # no-op for DHCV
        cluster.update_kernels(cfg)       # no-op for DHCV
        cluster.record_mmzca_stability()  # same metric definitions across baselines
        cluster.update_tvct(pre_states)

        observations = tick_metrics(cluster, cfg)
        ch_counts.append(len(cluster.all_chs))
        sa_counts.append(len(cluster.stand_alone))
        cm_counts.append(observations["cm_count"])
        valid_cm_counts.append(observations["valid_cm_count"])
        invalid_cm_counts.append(observations["invalid_cm_count"])
        if observations["ch_percent"] is not None:
            ch_pcts.append(observations["ch_percent"])
            sa_pcts.append(observations["sa_percent"])
        if observations["average_hops"] is not None:
            hop_means.append(observations["average_hops"])
            cm_observations += observations["valid_cm_count"]

        current_valid_cm = rcap.observe(cluster, cfg)
        parent_switches += sum(
            previous_valid_cm[vid][0] == root_key and
            previous_valid_cm[vid][1] != parent
            for vid, (root_key, parent) in current_valid_cm.items()
            if vid in previous_valid_cm
        )
        previous_valid_cm = current_valid_cm

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
    tvct = list(cluster.tvct_history)
    result = {
        "dataset": dataset, "veh_trans_range": tr,
        "start_time": start,
        "clustering_algorithm": "DHCV", "routing_algorithm": "None",
        "h_max": int(cfg.dhcv_d), "run_length": int(cfg.iter),
        "rcap_horizon_seconds": rcap_horizon,
        "n_vehs": len(cluster.veh_table.ids()), "n_buses": len(cluster.bus_table.ids()),
        "avg_chs": float(np.mean(ch_counts)) if ch_counts else None,
        "avg_stand_alones": float(np.mean(sa_counts)) if sa_counts else None,
        "execution_time": execution_time,
        "average_hops": float(np.mean(hop_means)) if hop_means else None,
        "ticks_with_cms": len(hop_means), "cm_tick_observations": cm_observations,
        "avg_ch_percentage": float(np.mean(ch_pcts)) if ch_pcts else None,
        "avg_sa_percentage": float(np.mean(sa_pcts)) if sa_pcts else None,
        "avg_cm_count": float(np.mean(cm_counts)) if cm_counts else None,
        "avg_valid_cm_count": float(np.mean(valid_cm_counts)) if valid_cm_counts else None,
        "avg_invalid_cm_count": float(np.mean(invalid_cm_counts)) if invalid_cm_counts else None,
        "dhcv_same_root_parent_switches": int(parent_switches),
        "same_root_parent_switches_per_second": parent_switches / float(cfg.iter),
        "dhcv_renewals": int(getattr(cluster, "dhcv_renewals", 0)),
        "dhcv_vi_recoveries": int(getattr(cluster, "dhcv_vi_recoveries", 0)),
        "dhcv_ch_to_cm_changes": int(getattr(cluster, "dhcv_ch_to_cm_changes", 0)),
        "avg_tvct": float(np.mean(tvct)) if tvct else None,
        "total_tvct": int(sum(tvct)),
        "stability_evaluation": base_vcsm,
        "RCAP": rc_value, "RCAP_numerator": rc_numerator,
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
    """Record non-swept DHCV parameters so --resume cannot mix settings."""
    excluded = {"sumo_trace", "sumo_edge", "sumo_node", "fcd", "times",
                "veh_trans_range", "start_time", "iter", "h_max", "dhcv_d"}
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


def experiment_settings(args, traffic_root, d, base_config):
    dhcv_parameters = {name: getattr(base_config, name, None) for name in (
        "dhcv_d", "dhcv_ri", "dhcv_vi", "dhcv_alpha", "dhcv_beta",
        "dhcv_k", "dhcv_lsr_mps", "dhcv_direction_tolerance_deg",
        "dhcv_threshold_mode", "dhcv_isolated_policy")}
    return {
        "algorithm": "DHCV", "clustering_only": True, "sequential": True,
        "datasets": args.datasets, "transmission_ranges_m": args.transmission_ranges,
        "run_length": args.run_length, "num_runs": args.num_runs,
        "first_start": args.first_start, "start_step": args.start_step,
        "start_times": [args.first_start + i * args.start_step for i in range(args.num_runs)],
        "h_max": d, "dhcv_d": d, "dhcv_parameters": dhcv_parameters,
        "rcap_horizon_seconds": args.rcap_horizon,
        "ci_level": DEFAULT_CI_LEVEL, "ci_method": args.ci_method,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_block_windows": args.bootstrap_block_windows,
        "random_seed": args.seed, "traffic_root": str(traffic_root),
        "fixed_config": fixed_configuration_snapshot(base_config),
        "result_columns": list(RESULT_FIELDS),
        "average_hops_definition": "Mean across ticks with physically root-connected CMs of their mean committed DHCV hop_count; VI-grace disconnected CMs excluded; not packet-routing hops",
        "avg_ch_percentage_definition": "Mean across ticks with in-area ordinary vehicles of 100*in-area CHs/in-area ordinary vehicles",
        "avg_sa_percentage_definition": "Mean across ticks with in-area ordinary vehicles of 100*in-area SAs/in-area ordinary vehicles",
        "avg_cm_count_definition": "Mean number of in-area DHCV CM-role vehicles, including temporarily disconnected VI-grace CMs",
        "RCAP_definition": "Presence throughout [t-Ts,t], valid physically root-connected CM of the same observed DHCV root incarnation at all ticks; same-root parent changes allowed",
        "RCAP_incarnation_tracking": "Independent measurement-only root epochs advance at committed-tick CH activation; DHCV does not use MMZCA kernel_root_incarnation",
        "dhcv_renewals_definition": "Number of whole-network DHCV elections including initial formation, as counted by DHCV DataTable",
        "dhcv_same_root_parent_switches_definition": "Consecutive committed-tick, physically valid CM parent changes retaining identical root and root incarnation",
        "n_vehs_definition": "Number of active ordinary vehicles at final tick, as in MMZCA runner",
        "ci_caveat": (f"Starts overlap ({args.run_length}-tick windows with "
                      f"{args.start_step}-tick stride). Block-bootstrap CIs "
                      "describe within-trace variability, not independent-seed "
                      "replication uncertainty."),
    }


def main():
    args = parse_args()
    if not args.datasets or not args.transmission_ranges:
        raise ValueError("Choose at least one dataset and transmission range")
    if min(args.transmission_ranges) <= 0:
        raise ValueError("Transmission ranges must be positive")
    if args.run_length < 2 or args.num_runs < 1 or args.first_start < 0 or args.start_step < 1:
        raise ValueError("Invalid run-length, num-runs, first-start or start-step")
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
    if not hasattr(base, "dhcv_d"):
        raise RuntimeError("Your options/constants.py does not define DHCV settings. "
                           "Use the DHCV baseline's options/constants.py")
    base.algorithm = "DHCV"
    d = int(args.dhcv_d if args.dhcv_d is not None else base.dhcv_d)
    if d < 1:
        raise ValueError("DHCV D must be >=1")
    base.dhcv_d = d
    base.h_max = d  # shared committed-path validator and metric depth convention
    for argument, name in ((args.dhcv_ri, "dhcv_ri"), (args.dhcv_vi, "dhcv_vi"),
                           (args.dhcv_threshold_mode, "dhcv_threshold_mode"),
                           (args.dhcv_isolated_policy, "dhcv_isolated_policy")):
        if argument is not None:
            setattr(base, name, argument)
    if int(base.dhcv_vi) < 1 or int(base.dhcv_ri) < 1:
        raise ValueError("DHCV RI and VI must be positive")

    output_dir = (args.output_dir if args.output_dir else
                  project_root / "simulation_results" /
                  f"DHCV_final_D{d}_{args.run_length}ticks")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = experiment_settings(args, traffic_root, d, base)
    settings_file = output_dir / "experiment_metadata.json"
    if settings_file.exists():
        with settings_file.open("r", encoding="utf-8") as f:
            previous = json.load(f)
        if previous != settings:
            raise ValueError("Output folder belongs to a different experiment. "
                             "Use --output-dir for new settings. Do not mix CSVs.")
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
    print(f"DHCV FINAL / sequential; D={d}; RI={base.dhcv_ri}; VI={base.dhcv_vi}; RCAP horizon={args.rcap_horizon}s")
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
                            "start_time": start, "h_max": d,
                            "run_length": args.run_length,
                            "rcap_horizon_seconds": args.rcap_horizon,
                            "clustering_algorithm": "DHCV", "routing_algorithm": "None"}
                existing = load_existing_result(out_path, expected) if args.resume else None
                if existing is not None:
                    completed.append(existing)
                    print(f"[{count}/{len(jobs)}] SKIP {dataset} TR={tr} start={start}", flush=True)
                    continue
                cfg = copy.copy(template)
                cfg.veh_trans_range = tr
                cfg.dhcv_d = d
                cfg.h_max = d
                cfg.algorithm = "DHCV"
                cfg.iter = args.run_length
                cfg.start_time = start
                trial_seed = fixed_seed(args.seed, dataset, tr, start)
                random.seed(trial_seed)
                np.random.seed(trial_seed)
                print(f"[{count}/{len(jobs)}] RUN {dataset} TR={tr} "
                      f"start={start} D={d}", flush=True)
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
