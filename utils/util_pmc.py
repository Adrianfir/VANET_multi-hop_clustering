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


def be_ch():



def calculate_etx(veh_id):
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


def calculate_lltm():
    lltm = 0
    return lltm


def calculate_pri():
    pri = 0
    return pri

