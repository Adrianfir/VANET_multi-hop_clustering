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
        eval_cluster = cluster.eval_cluster(configs)
        connection_evaluation = cluster.eval_connections()
        print(f'stability_evaluation: {eval_cluster}')
        print(f'connection_evaluation: {connection_evaluation}')
    #     cluster.show_graph(configs)
    #     cluster.save_map_img(1, '/Users/pouyafirouzmakan/Desktop/slideshow/saved_imgs/Graph' + str(i))
    end_time = time.time()
    # util.make_slideshow('/Users/pouyafirouzmakan/Desktop/slideshow/saved_imgs/',
    #                     '/Users/pouyafirouzmakan/Desktop/slideshow/saved_imgs/slide.mp4', configs.fps)
    # cluster.show_graph(configs)
    # cluster.print_table()

    print('\n')
    print('chs: ', cluster.all_chs)
    print('stand_alones: ', cluster.stand_alone)
    print("execution time: ", end_time - start_time)

    print(f'all the edges: \n{cluster.net_graph.edges()}')

