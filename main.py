"""
<<main.py>>

This project is related to multi-hop clustering in VANET

"""
__author__: str = "Pouya 'Adrian' Firouzmakan"

import time
import os
from data_cluster import DataTable
from configs.config import Configs
from zonex import ZoneID
import utils.util as util
import re
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    configs = Configs().config

    area_zone_micro = ZoneID(configs, zoning_mode='mic')
    area_zone_micro.zones()
    area_zone_meso = ZoneID(configs, zoning_mode='mes')
    area_zone_meso.zones()
    area_zone_macro = ZoneID(configs, zoning_mode='mac')
    area_zone_macro.zones()
    area_zones = {'micro':area_zone_micro,
                  'meso':area_zone_meso,
                  'macro':area_zone_macro}
    cluster = DataTable(configs, area_zones)
    connections = list()
    n_chs = list()
    n_savs = list()
    start_time = time.time()
    collect_timings = os.environ.get('MMZCA_TIMINGS') == '1'
    phase_times = {key: 0.0 for key in (
        'mobility_and_departures', 'cluster_update', 'pch_election',
        'kernel_exchange', 'stability_and_tvct')}
    # Printing each tick can become surprisingly costly in IDE terminals.
    report_every = max(1, int(getattr(configs, 'progress_every', 25)))
    for i in range(configs.iter):
        # Capture the previous committed state BEFORE departures and updates.
        # New arrivals/departures are excluded by update_tvct's common-ID rule.
        pre_states = cluster._snapshot_tvct_states()
        t0 = time.perf_counter() if collect_timings else 0.0
        cluster.update(configs)
        if collect_timings:
            phase_times['mobility_and_departures'] += time.perf_counter() - t0
            t0 = time.perf_counter()
        if (i + 1) % report_every == 0 or i + 1 == configs.iter:
            print(f"MMZCA tick {i + 1}/{configs.iter} (SUMO {cluster.time})")

        cluster.update_cluster(cluster.veh_table.ids(), configs)
        if collect_timings:
            phase_times['cluster_update'] += time.perf_counter() - t0
            t0 = time.perf_counter()
        cluster.stand_alones_cluster(configs)
        if collect_timings:
            phase_times['pch_election'] += time.perf_counter() - t0
            t0 = time.perf_counter()
        cluster.update_kernels(configs)
        if collect_timings:
            phase_times['kernel_exchange'] += time.perf_counter() - t0
            t0 = time.perf_counter()
        cluster.record_mmzca_stability()
        tvct_t, avg_tvct_t = cluster.update_tvct(pre_states)
        if collect_timings:
            phase_times['stability_and_tvct'] += time.perf_counter() - t0
        # cluster.update_other_connections()
        # cluster.form_net_graph()
        # connection_evaluation = cluster.connected_components()
        # connections.append(connection_evaluation)
        n_chs.append(len(cluster.all_chs))
        n_savs.append(len(cluster.stand_alone))
        # cluster.show_graph(configs)
        # cluster.save_map_img(1, '/Users/pouyafirouzmakan/Desktop/slideshow/saved_imgs/Graph' + str(i))
    #
    end_time = time.time()
    if collect_timings:
        print('\nMMZCA per-stage timings (seconds):')
        for phase, seconds in phase_times.items():
            print(f'  {phase}: {seconds:.3f}')
        print(f'  wall_clock_total: {end_time - start_time:.3f}')
    # util.make_slideshow('/Users/pouyafirouzmakan/Desktop/slideshow/saved_imgs/',
    #                     '/Users/pouyafirouzmakan/Desktop/slideshow/saved_imgs/slide.mp4', configs.fps)
    # cluster.print_table()
    # nx.draw(cluster.ch_net, with_labels=False)
    print(f'stability_evaluation: {cluster.vcsm(configs)}')
    print(f'VCSM_R: {cluster.vcsm_r(configs)}')
    print(f'VCSM_CM: {cluster.vcsm_cm(configs)}')
    # print(f'connection_evaluation: {sum(connections)/len(connections)}->{connections}')
    print('\n')
    print(f'n_vehs: {len(cluster.veh_table.ids())}')
    print(f'n_buses: {len(cluster.bus_table.ids())}')
    print(f'n_chs: {n_chs}')
    print(f'n_savs: {n_savs}')
    print(f'avg_chs: {sum(n_chs)/len(n_chs)}')
    print(f'avg_stand_alones: {sum(n_savs)/len(n_savs)}')
    print(f'mmzca_same_root_parent_switches: {getattr(cluster, "mmzca_parent_switches", 0)}')

    print(f'tvct_t: {cluster.tvct_history}, avg_tvct: {cluster.avg_tvct_history}')
    # print(f'execution time: {end_time - start_time}')
    # print(f'all the edges: \n{cluster.net_graph.edges()}')
    ch_high_mems = 0
    for i in cluster.all_chs:
        temp_table = cluster.bus_table if 'bus' in i else cluster.veh_table
        if len(temp_table.values(i)['cluster_members'])>10:
            ch_high_mems += 1
    print(f'ch_high_mems: {ch_high_mems}')
    plt.plot(range(len(cluster.tvct_history)), cluster.tvct_history, label='TVCT(t)')
    plt.plot(range(len(cluster.avg_tvct_history)), cluster.avg_tvct_history,
             label='Running mean TVCT')
    plt.legend()
    plt.show()
