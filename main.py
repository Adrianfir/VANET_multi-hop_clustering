"""
<<main.py>>

This project is related to multi-hop clustering in VANET

"""
__author__: str = "Pouya 'Adrian' Firouzmakan"

import time
from data_cluster import DataTable
from configs.config import Configs
from zonex import ZoneID
import utils.util as util
import re

if __name__ == "__main__":
    configs = Configs().config

    area_zones = ZoneID(configs)  # This is a hash table including all zones and their max and min lat and longs
    area_zones.zones()
    cluster = DataTable(configs, area_zones)
    start_time = time.time()
    for i in range(configs.iter):
        cluster.update(configs, area_zones)
        print(cluster.time)
        cluster.update_cluster(cluster.veh_table.ids(), configs, area_zones)
        cluster.stand_alones_cluster(configs, area_zones)
        if 'veh399' in cluster.veh_table.ids():
            print(cluster.veh_table.values('veh399'))
        # cluster.show_graph(configs)
        # cluster.save_map_img(1, '/Users/pouyafirouzmakan/Desktop/VANET/saved_imgs/Graph' + str(i))
    end_time = time.time()
    # util.make_slideshow('/Users/pouyafirouzmakan/Desktop/VANET/saved_imgs/',
    #                     '/Users/pouyafirouzmakan/Desktop/VANET/saved_imgs/', configs.fps)
    # cluster.show_graph(configs)
    cluster.print_table()
    eval_cluster = cluster.eval_cluster(configs)
    print('evaluation: ', eval_cluster)
    # print('n_bus: ', len(cluster.bus_table.ids()))
    # print('n_veh: ', len(cluster.veh_table.ids()))
    print('\n')
    print('chs: ', cluster.all_chs)
    print('stand_alones: ', cluster.stand_alone)
    print("execution time: ", end_time - start_time)

