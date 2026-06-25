"""
main.py for vehicle-only MetoidS clustering experiments.

Use this with the fixed vehicle-only data_cluster.py implementation.
The SUMO XML input pipeline remains unchanged: Configs/Inputs still load the
SUMO files from constants.py.
"""

__author__: str = "Pouya 'Adrian' Firouzmakan"

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt

from data_cluster import DataTable
from configs.config import Configs
from zonex import ZoneID


def _set_missing_metoids_params(configs):
    """Add MetoidS parameters without requiring changes in constants.py."""
    defaults = {
        "metoids_target_cluster_size": 8,
        "metoids_w_cosine": 0.65,
    }
    for name, value in defaults.items():
        if not hasattr(configs, name):
            setattr(configs, name, value)


def _build_zones(configs):
    area_zone_micro = ZoneID(configs, zoning_mode="mic")
    area_zone_micro.zones()

    area_zone_meso = ZoneID(configs, zoning_mode="mes")
    area_zone_meso.zones()

    area_zone_macro = ZoneID(configs, zoning_mode="mac")
    area_zone_macro.zones()

    return {
        "micro": area_zone_micro,
        "meso": area_zone_meso,
        "macro": area_zone_macro,
    }


def _safe_average(values):
    return sum(values) / len(values) if values else 0.0


def _count_active_members(cluster):
    total_members = 0
    for ch_id in cluster.all_chs:
        if ch_id in cluster.veh_table.ids():
            total_members += len(cluster.veh_table.values(ch_id).get("cluster_members", set()))
    return total_members


def _write_metrics_csv(metrics_rows, output_path="metoids_metrics.csv"):
    if not metrics_rows:
        return

    fieldnames = list(metrics_rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)


def _plot_tvct(cluster, output_path="metoids_tvct.png"):
    if not cluster.tvct_history:
        return

    plt.figure()
    plt.plot(range(len(cluster.tvct_history)), cluster.tvct_history, label="TVCT(t)")
    plt.plot(range(len(cluster.avg_tvct_history)), cluster.avg_tvct_history, label="Average TVCT")
    plt.xlabel("Simulation tick")
    plt.ylabel("TVCT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    configs = Configs().config
    _set_missing_metoids_params(configs)

    area_zones = _build_zones(configs)
    cluster = DataTable(configs, area_zones)

    n_chs = []
    n_savs = []
    n_members = []
    metrics_rows = []

    start_wall_time = time.time()

    max_available_iter = max(0, len(configs.times) - configs.start_time - 1)
    run_iter = min(configs.iter, max_available_iter)

    for tick in range(run_iter):
        cluster.update(configs)

        pre_states = cluster._snapshot_tvct_states()
        cluster.update_cluster(cluster.veh_table.ids(), configs)
        cluster.stand_alones_cluster(configs)
        tvct_t, avg_tvct_t = cluster.update_tvct(pre_states)

        ch_count = len(cluster.all_chs)
        sa_count = len(cluster.stand_alone)
        member_count = _count_active_members(cluster)
        active_count = len(cluster.veh_table.ids())

        n_chs.append(ch_count)
        n_savs.append(sa_count)
        n_members.append(member_count)

        metrics_rows.append({
            "tick_index": tick,
            "sumo_time_index": cluster.time,
            "active_vehicles": active_count,
            "chs": ch_count,
            "stand_alones": sa_count,
            "members": member_count,
            "tvct": tvct_t,
            "avg_tvct": avg_tvct_t,
        })

        print(
            f"time={cluster.time} | active={active_count} | "
            f"CH={ch_count} | CM={member_count} | SA={sa_count} | "
            f"TVCT={tvct_t} | avg_TVCT={avg_tvct_t:.4f}"
        )

    end_wall_time = time.time()

    vcsm_value = cluster.vcsm(configs)
    avg_chs = _safe_average(n_chs)
    avg_savs = _safe_average(n_savs)
    avg_members = _safe_average(n_members)
    avg_tvct = cluster.avg_tvct_history[-1] if cluster.avg_tvct_history else 0.0

    print("\n========== MetoidS clustering results ==========")
    print(f"iterations_run: {run_iter}")
    print(f"final_active_vehicles: {len(cluster.veh_table.ids())}")
    print(f"vehicles_left_area: {len(cluster.left_veh)}")
    print(f"avg_chs: {avg_chs}")
    print(f"avg_members: {avg_members}")
    print(f"avg_stand_alones: {avg_savs}")
    print(f"vcsm: {vcsm_value}")
    print(f"avg_tvct: {avg_tvct}")
    print(f"execution_time_sec: {end_wall_time - start_wall_time}")

    print(f"\nn_chs: {n_chs}")
    print(f"n_savs: {n_savs}")
    print(f"tvct_history: {cluster.tvct_history}")
    print(f"avg_tvct_history: {cluster.avg_tvct_history}")

    _write_metrics_csv(metrics_rows, "metoids_metrics.csv")
    _plot_tvct(cluster, "metoids_tvct.png")

    print("\nSaved: metoids_metrics.csv")
    print("Saved: metoids_tvct.png")
    plt.plot(range(len(cluster.tvct_history)), cluster.tvct_history)
    plt.show()


if __name__ == "__main__":
    main()
