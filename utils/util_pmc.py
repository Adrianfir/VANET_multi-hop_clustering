"""
This is the utils file including the small functions
"""
__author__: str = "Pouya 'Adrian' Firouzmakan"
__all__ = [

]

import numpy as np
import haversine as hs

import utils.util as util


def calculate_etx(m, n):
    return 1


def calculate_nfollow(n, veh_table, bus_table,
                      zone_buses, zone_vehicles):
    """
    this function would calculate the N_follow in the comparison paper
    :param n:
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

    if 'bus' in n:
        table = bus_table
    else:
        table = veh_table
    (bus_candidates, ch_candidates,
     sub_ch_candidates, other_vehs) = util.det_near_ch(n, veh_table, bus_table,
                                                       zone_buses, zone_vehicles)
    all_vehs = set()
    all_vehs = all_vehs.union(all_vehs, bus_candidates,
                              ch_candidates, sub_ch_candidates, other_vehs)

    try:
        all_vehs.remove(n)
    except KeyError:
        pass

    for i in all_vehs:
        connects = set()

        if 'bus' in i:
            if table.values(n)['lane']['id'] == bus_table.values(i)['lane']['id']:
                d_neigh.add(i)
            connects.union(connects, bus_table.values(i)['other_chs'],
                           bus_table.values(i)['cluster_members'], {i})
        else:
            if table.values(n)['lane']['id'] == veh_table.values(i)['lane']['id']:
                d_neigh.add(i)
            connects.union(connects, veh_table.values(i)['cluster_members'],
                           veh_table.values(i)['sub_cluster_members'])

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

    llt = ((np.sqrt(np.square(d) * (np.square(delta_v_x) + np.square(delta_v_y)) -
                    np.square((delta_p_x * delta_v_y) - (delta_p_y * delta_v_x))) -
            ((delta_p_x * delta_v_x) - (delta_p_y * delta_v_y))) / (np.square(delta_v_x) + np.square(delta_v_y)))
    return llt


def calculate_pri(m, n, veh_table, bus_table, zone_buses, zone_vehicle, weights):
    alpha = weights[0]
    beta = weights[1]
    gama = weights[2]
    first_part = alpha/(calculate_nfollow(n, veh_table, bus_table,
                                           zone_buses, zone_vehicle) + 0.0001)
    second_part = beta * calculate_etx(m, n)
    third_part = gama/(calculate_llt(m, n, veh_table, bus_table) + 0.0001)
    pri = first_part + second_part + third_part
    return pri


def choose_ch(veh_id, veh_table, bus_table, all_vehs, zone_buses, zone_vehicle, weights):
    pri = 1000000
    ch = None
    ef = pri
    for i in all_vehs:
        if 'bus' in i:
            temp_pri = calculate_pri(veh_id, i, veh_table, bus_table, zone_buses, zone_vehicle, weights)
            if temp_pri < pri:
                ch = i
                ef = temp_pri
        elif (('bus' not in i) and
              (
                      (veh_table.values(i)['cluster_head'] is True) or
                      ((veh_table.values(i)['primary_ch'] is not None) and
                       (veh_table.values(i)['secondary_ch'] is not None))
              )
              ):
            temp_pri = calculate_pri(veh_id, i, all_vehs, veh_table, bus_table, zone_buses, zone_vehicle, weights)
            if temp_pri < pri:
                ch = i
                ef = temp_pri
    return ch, ef
