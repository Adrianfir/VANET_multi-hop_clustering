"""
This is the utils file including the small functions related to self.net_graph
"""
__author__: str = "Pouya 'Adrian' Firouzmakan"
__all__ = []

import networkx as nx

def veh_add_edges(veh_id, veh_table, bus_table, net_graph):
    """
    :param veh_id:
    :param veh_table:
    :param bus_table:
    :return:
    """
    all_nearby = set()
    if ((veh_table.values(veh_id)['primary_ch'] is not None) and
            (veh_table.values(veh_id)['secondary_ch'] is None)):
        net_graph.add_edges_from((veh_id, veh_table.values(veh_id)['primary_ch']),
                                 (veh_table.values(veh_id)['primary_ch'], veh_id))
    elif ((veh_table.values(veh_id)['primary_ch'] is not None) and
            (veh_table.values(veh_id)['secondary_ch'] is not None)):
        net_graph.add_edges_from((veh_id, veh_table.values(veh_id)['secondary_ch']),
                                 (veh_table.values(veh_id)['secondary_ch'], veh_id))

    all_nearby.union(all_nearby, veh_table.valeus(veh_id)['other_chs'],
                     veh_table.valeus(veh_id)['other_vehs'], veh_table.valeus(veh_id)['sub_members'])

    for i in all_nearby:
        temp_table = veh_table if 'veh' in i else bus_table
        if (('veh' in i) and (veh_table.values(veh_id)['primary_ch'] ==
                             veh_table.values(veh_id)['primary_ch']) and
                ((veh_table.values(veh_id)['secondary_ch'] != i) or
                 (veh_table.values(i)['secondary_ch'] != veh_id))):
            continue
        else:
            net_graph.add_edges_from((veh_id, i),
                                     (i, veh_id))
    return net_graph

def ch_add_edges(veh_id, veh_table, bus_table, net_graph):
    """

    :param veh_id:
    :param veh_table:
    :param bus_table:
    :return:
    """
    all_nearby = set()
    all_nearby.union(all_nearby, veh_table.valeus(veh_id)['other_chs'],
                     veh_table.valeus(veh_id)['other_vehs'], veh_table.valeus(veh_id)['cluster_members'])


def bus_add_edges(bus_id, veh_table, bus_table, net_graph):
    """

    :param bus_id:
    :param veh_table:
    :param bus_table:
    :return:
    """
    return True