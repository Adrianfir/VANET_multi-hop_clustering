"""
This is the utils file including the small functions
"""
__author__: str = "Pouya 'Adrian' Firouzmakan"
__all__ = [

]

import numpy as np
import random
import haversine as hs
from linked_list import LinkedList
from scipy import spatial
import time
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import os
import cv2
import re

import utils.util as util


def calculate_etx(m, n):
    return 1


def calculate_nfollow(veh_id, veh_table, bus_table,
                      zone_buses, zone_vehicle):
    """
    this function would calculate the N_follow in the comparison paper
    :param veh_id:
    :param veh_table:
    :param bus_table:
    :param zone_buses:
    :param zone_vehicle:
    :return:
    """
    n_follow = 0
    neighbors = set()
    d_neigh = set()
    fc = set()

    if 'bus' in veh_id:
        table = bus_table
    else:
        table = veh_table
    (bus_candidates, ch_candidates,
     sub_ch_candidates, other_vehs) = util.det_near_ch(veh_id, veh_table, bus_table,
                                                       zone_buses, zone_vehicle)

    neighbors = neighbors.union(neighbors, bus_candidates, ch_candidates,
                                sub_ch_candidates, other_vehs)

    for i in neighbors:
        connects = set()
        if 'bus' in i:
            if table.values(veh_id)['lane']['id'] == bus_table.values(i)['lane']['id']:
                d_neigh.add(i)
            connects.union(connects, bus_table.values(veh_id)['other_chs'], bus_table.values(veh_id)['other_vehs'],
                           bus_table.values(veh_id)['cluster_members'], bus_table.values(veh_id)['sub_members'])
        else:
            if table.values(veh_id)['lane']['id'] == veh_table.values(i)['lane']['id']:
                d_neigh.add(i)
            connects.union(connects, veh_table.values(veh_id)['other_chs'], veh_table.values(veh_id)['other_vehs'],
                           veh_table.values(veh_id)['cluster_members'], veh_table.values(veh_id)['sub_members'])
            if veh_table.values(i)['primary_ch'] is not None and veh_table.values(i)['secondary_ch'] is None:
                connects.union(veh_table.values['primary_ch'])
            elif veh_table.values(i)['primary_ch'] is None and veh_table.values(i)['secondary_ch'] is not None:
                connects.union(veh_table.values['secondary_ch'])
        fc = fc.union(fc, connects)

    len(fc) + len(d_neigh)
    return len(fc) + len(d_neigh)


def calculate_llt(m, n, veh_table, bus_table):
    if 'bus' in m:
        table_m = bus_table
    else:
        table_m = veh_table
    if 'bus' in n:
        table_n = bus_table
    else:
        table_n = veh_table

    delta_v_x = table_m.values(m)['prev_speed'] - table_n.values(n)['prev_speed']
    delta_v_y = table_m.values(m)['speed'] - table_n.values(n)['speed']

    delta_p_x = hs.haversine((table_m.values(m)["prev_lat"],
                         table_m.values(m)["prev_long"]),
                        (table_n.values(n)['prev_lat'],
                         table_n.values(n)['prev_long']),
                        unit=hs.Unit.METERS)

    delta_p_y = hs.haversine((table_m.values(m)["lat"],
                              table_m.values(m)["long"]),
                             (table_n.values(n)['lat'],
                              table_n.values(n)['long']),
                             unit=hs.Unit.METERS)

    d = min(table_m.values(m)['trans_range'], table_n.values(n)['trans_range'])

    llt = ((np.sqrt(np.square(d)*(np.square(delta_v_x)+np.square(delta_v_y)) -
                   np.square((delta_p_x*delta_v_y)-(delta_p_y*delta_v_x))) -
             ((delta_p_x*delta_v_x)-(delta_p_y*delta_v_y))) / (np.square(delta_v_x)+np.square(delta_v_y)))
    return llt


def calculate_pri(m, n, veh_table, bus_table, zone_buses, zone_vehicle, weights):
    alpha = weights[0]
    beta = weights[1]
    gama = weights[2]
    first_part = alpha/calculate_nfollow(n, veh_table, bus_table,
                      zone_buses, zone_vehicle)
    second_part = beta*calculate_etx(m, n)
    third_part = gama/calculate_llt(m, n, veh_table, bus_table)
    pri = first_part + second_part + third_part
    return pri


def choose_ch(veh_id, veh_table, bus_table, all_vehs):
    ch = None
    for i in all_vehs:
        if 'bus' in i:
            table_i = bus_table
        else:
            table_i = veh_table

