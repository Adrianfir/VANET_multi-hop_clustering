"""
<<main.py>>

This project is related to multi-hop clustering in VANET

"""
__author__: str = "Pouya 'Adrian' Firouzmakan"

import time
import numpy as np
import pandas as pd
from data_cluster import DataTable
from configs.config import Configs
from zonex import ZoneID
import utils.util as util
import re
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    configs = Configs().config

    dif_tr = [300, ]
    ########################### Define different weights
    # Define the size of each list and the step increment
    list_size = 3
    step = 0.1

    # Generate all possible values from 0 to 1 with the given step
    possible_values = [round(i * step, 1) for i in range(int(1 / step) + 1)]

    # Generate all possible combinations of values with sum equal to 1
    all_weight_lists = []

    for val1 in possible_values:
        for val2 in possible_values:
            remaining = round(1 - val1 - val2, 1)
            if remaining in possible_values and remaining >= 0:
                all_weight_lists.append([val1, val2, remaining])
    ###########################
    area_zones = ZoneID(configs)  # This is a hash table including all zones and their max and min lat and longs
    area_zones.zones()
    num_times = 1
    start_time = time.time()

    for configs.veh_trans_range in dif_tr:
        if configs.veh_trans_range == 100:
            configs.weights_s = [0.5, 0.5, 0.0]
        elif configs.veh_trans_range ==200:
            configs.weights_s = [0.5, 0.2, 0.3]
        elif configs.veh_trans_range == 300:
            configs.weights_s = [0.9, 0.0, 0.1]
        cols = ['rsu', 'TR', 'weights_s', 'weights_m', 'n_veh', 'n_buses', 'n_sav', 'n_chs', 'stab_eval']
        out_put = pd.DataFrame(columns=cols)
        for configs.weights_m in all_weight_lists:
            configs.weights_m = np.array(configs.weights_m)
            cluster = DataTable(configs, area_zones)
            connections = list()
            n_chs = list()
            n_savs = list()
            for i in range(configs.iter):
                cluster.update(configs, area_zones)
                print(cluster.time)
                cluster.update_cluster(cluster.veh_table.ids(), configs, area_zones)
                cluster.stand_alones_cluster(configs, area_zones)
                cluster.update_other_connections()
                cluster.form_net_graph()
                connection_evaluation = cluster.connected_components()
                connections.append(connection_evaluation)
                n_chs.append(len(cluster.all_chs))
                n_savs.append(len(cluster.stand_alone))

            eval_cluster = cluster.eval_cluster(configs)

            print(num_times, configs.veh_trans_range, configs.weights_m,
                  len(cluster.veh_table.ids()), len(cluster.bus_table.ids()),
                  len(cluster.stand_alone), len(cluster.all_chs), eval_cluster
                  )
            num_times += 1

            new_row = pd.Series(['yes', configs.veh_trans_range, configs.weights_s, configs.weights_m, 
                                 len(cluster.veh_table.ids()), len(cluster.bus_table.ids()),
                                 sum(n_savs)/len(n_savs), sum(n_chs)/len(n_chs), eval_cluster], index=cols)

            out_put = pd.concat([out_put, new_row.to_frame().T], ignore_index=True)

        out_put.to_csv('results/' + str(configs.veh_trans_range) + 'no_RSU.csv')
        end_time = time.time()
        print("execution time: ", end_time - start_time)

