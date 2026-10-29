"""
This is the utils file including the small functions
"""
__author__: str = "Pouya 'Adrian' Firouzmakan"
__all__ = [
    'add_member', 'add_sub_member', 'choose_ch', 'choose_multihop_ch', 'det_beta', 'det_buses_other_ch', 'det_dist',
    'det_near_ch', 'det_near_sa', 'det_pot_ch', 'det_pot_ch_dsca', 'image_num', 'initiate_new_bus',  'initiate_new_veh',
    'mac_address', 'make_slideshow', 'middle_zone', 'presence', 'priority_clusters',  'remove_member',
    'remove_sub_member', 'save_img', 'set_ch', 'set_ch_to_veh', 'sumo_net_info', 'update_bus_table', 'sumo_net_info',
    'update_sa_net_graph', 'update_veh_table',
    'mmzca_kernel_radius', 'mmzca_build_kernel', 'mmzca_kernel_is_usable',
    'mmzca_affinity_from_kernel', 'mmzca_raw_affinity_from_kernel',
    'det_normalized_micro_affinity', 'det_micro_affinity',
    'mmzca_affinity_status', 'mmzca_latest_root_kernel',
    'mmzca_note_ch_promotion', 'mmzca_kernel_tick',
    'mmzca_physical_link', 'choose_mmzca_parent'
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
from selenium.webdriver.firefox.service import Service
import os
import cv2
import re
from functools import lru_cache



# These caches are deliberately tick-local. SUMO positions and zone motion
# change at every timestep; DataTable calls mmzca_begin_tick before updates.
_MMZCA_DISTANCE_CACHE = None
_MMZCA_HEADING_CACHE = None


def mmzca_begin_tick():
    global _MMZCA_DISTANCE_CACHE, _MMZCA_HEADING_CACHE
    _MMZCA_DISTANCE_CACHE = {}
    _MMZCA_HEADING_CACHE = {}


@lru_cache(maxsize=50000)
def _mmzca_neighbor_zones_cached(zones, zone_id):
    return tuple(zones.neighbor_zones(zone_id))


def initiate_new_bus(veh, zones, micro_zone_id, meso_zone_id, macro_zone_id, config, understudied_area):
    """
    this function is for initialing the new bus coming to the area
    :param veh:veh extracted from .xml file
    :param zones:the hash_table related to zones
    :param micro_zone_id:micro_zone_id of the vehicle determined based on its location
    :param meso_zone_id:meso_zone_id of the vehicle determined based on its location
    :param macro_zone_id:macro_zone_id of the vehicle determined based on its location
    :param config:it is the config file
    :param understudied_area:the un_padded area
    :return:a dictionary for initiating the new bus coming to the area
    """
    return dict(long=float(veh.getAttribute('x')),
                lat=float(veh.getAttribute('y')),
                angle=float(veh.getAttribute('angle')),
                speed=float(veh.getAttribute('speed')) + 0.01,
                pos=float(veh.getAttribute('pos')),
                lane={'id': veh.getAttribute('lane'), 'timer': 0},
                micro_zone= micro_zone_id,
                prev_micro_zone=None,
                meso_zone=meso_zone_id,
                prev_meso_zone=None,
                macro_zone= macro_zone_id,
                prev_macro_zone=None,
                meso_zone_changed_this_tick=False,
                macro_zone_changed_this_tick=False,
                neighbor_zones=zones.neighbor_zones(macro_zone_id),
                in_area=presence(understudied_area, veh),
                arrive_time=None,
                depart_time=None,
                trans_range=config.bus_trans_range,
                message_dest={},
                message_source={},
                cluster_head=True,
                other_chs=set(),  # other chs in the trans range of veh.getAttribute('id)
                cluster_members=set(),
                sub_cluster_members=set(),
                gate_chs=set(),
                gates=dict(),
                other_vehs=set(),
                ip=None,
                mac=mac_address(),
                counter=config.counter,
                role='CH',
                root_ch=veh.getAttribute('id'),
                parent_id=None,
                parent_node=None,
                hop_count=0,
                current_root_cost=0.0,
                current_parent_cost=0.0,
                path_bottleneck=0.0,
                committed_link_score=None,
                root_connection_valid=True,
                direct_children=set(),
                admit_children=True,
                unresolved_start_macro_zone=None,
                kernel_root_incarnation=None,
                kernel_local_root_id=None,
                kernel_descriptor=None,
                kernel_subtree_hist={},
                kernel_subtree_version=0,
                kernel_child_reports={},
                kernel_last_signature=None,
                kernel_known_children=set(),
                kernel_snapshot_version=0,
                kernel_last_snapshot_hist=None,
                kernel_last_snapshot_anchor=None
                )


def initiate_new_veh(veh, zones, micro_zone_id, meso_zone_id, macro_zone_id, config, understudied_area):
    """
    this function is for initialing the new vehicle coming to the area
    :param veh:veh extracted from .xml file
    :param zones:the hash_table related to zones
    :param micro_zone_id:micro_zone_id of the vehicle determined based on its location
    :param meso_zone_id:meso_zone_id of the vehicle determined based on its location
    :param macro_zone_id:macro_zone_id of the vehicle determined based on its location
    :param config:it is the config file
    :param understudied_area:the un_padded area
    :return:a dictionary for initiating the new vehicle coming to the area
    """
    lane_id = veh.getAttribute('lane')
    if ":" not in lane_id:
        pattern = re.compile(f"^(.*?){re.escape('_')}")
        match = pattern.search(lane_id)
        if match:
            lane_id = match.group(1)

    return dict(long=float(veh.getAttribute('x')),
                lat=float(veh.getAttribute('y')),
                angle=float(veh.getAttribute('angle')),
                speed=float(veh.getAttribute('speed')) + 0.01,
                sai=1,  # this feature is added for BeFit factor to make comparison
                degree_n=0,  # this is the neighborhood degree for Befit factor to make comparison
                pos=float(veh.getAttribute('pos')),
                lane={'id': lane_id, 'timer': 0},
                micro_zone=micro_zone_id,
                prev_micro_zone=None,
                meso_zone=meso_zone_id,
                prev_meso_zone=None,
                macro_zone= macro_zone_id,
                prev_macro_zone=None,
                meso_zone_changed_this_tick=False,
                macro_zone_changed_this_tick=False,
                neighbor_zones=zones.neighbor_zones(macro_zone_id),
                in_area=presence(understudied_area, veh),
                arrive_time=None,
                depart_time=None,
                trans_range=config.veh_trans_range,
                message_dest={},
                message_source={},
                cluster_head=False,  # if the vehicle is a ch, it will be True
                primary_ch=None,
                secondary_ch=None,
                priority_ch=None,
                priority_counter=config.priority_counter,
                other_chs=set(),  # other chs in the trans range of veh.getAttribute('id)
                cluster_members=set(),  # This will be a Graph if the vehicle is a ch
                sub_cluster_members=set(),  # cluster members connected to claster through this vehicle would be in this set
                gates=dict(),
                gate_chs=set(),
                other_vehs=set(),
                ip=None,
                mac=mac_address(),
                counter=config.counter,  # a counter_time to search and join a cluster
                start_ch_zone=None,  # This is the zone that vehicle starts becoming a ch
                stability_episodes=[],  # compressed, root-based MMZCA metric states
                stability_last_tick=None,
                cluster_record=LinkedList(None, {'is_ch': False, 'secondary_ch': list(), 'start_time': None,
                                                 'ef': None,  'timer': None, 'interrupt': list()}),  # the linked_list
                role='SA',
                root_ch=None,
                parent_id=None,
                parent_node=None,
                hop_count=None,
                current_root_cost=None,
                current_parent_cost=None,
                path_bottleneck=None,
                committed_link_score=None,
                root_connection_valid=False,
                direct_children=set(),
                admit_children=True,
                unresolved_start_macro_zone=None,
                kernel_root_incarnation=None,
                kernel_local_root_id=None,
                kernel_descriptor=None,
                kernel_subtree_hist={},
                kernel_subtree_version=0,
                kernel_child_reports={},
                kernel_last_signature=None,
                kernel_known_children=set(),
                kernel_snapshot_version=0,
                kernel_last_snapshot_hist=None,
                kernel_last_snapshot_anchor=None
                # would record the clusters that this vehicle would join. key is the cluster_head which is None when the
                # vehicle gets initialized, value['ef'] is the "ef" and value['timer] is the amount of time that this
                # vehicle would remain in that cluster
                )


def add_member(ch_id, bus_table,
               veh_id, veh_table, config,
               ef, sec, bus_candidates,
               ch_candidates, stand_alone,
               zone_stand_alone,
               other_vehs):
    """
    this function is used for adding a vehicle to a cluster as a 'member' not 'sum_member'
    :param ch_id:
    :param bus_table:
    :param veh_id:
    :param veh_table:
    :param config:
    :param ef:
    :param sec: self.time in the data_cluster.py
    :param bus_candidates:
    :param ch_candidates:
    :param stand_alone:
    :param zone_stand_alone:
    :param other_vehs:
    :return:
    """
    veh_table.values(veh_id)['primary_ch'] = ch_id
    veh_table.values(veh_id)['secondary_ch'] = None
    veh_table.values(veh_id)['root_ch'] = ch_id
    veh_table.values(veh_id)['parent_id'] = ch_id
    veh_table.values(veh_id)['parent_node'] = ch_id
    veh_table.values(veh_id)['hop_count'] = 1
    veh_table.values(veh_id)['role'] = 'CM'
    veh_table.values(veh_id)['root_connection_valid'] = True

    if ch_id == veh_table.values(veh_id)['priority_ch']:
        veh_table.values(veh_id)['cluster_record'].pop()
        veh_table.values(veh_id)['cluster_record'].tail.value['secondary_ch'].append(None)
        veh_table.values(veh_id)['cluster_record'].tail.value['timer'] += 1
        veh_table.values(veh_id)['cluster_record'].tail.value['interrupt'].append(
            config.priority_counter - veh_table.values(veh_id)['priority_counter'])

    else:
        veh_table.values(veh_id)['cluster_record'].tail.key = ch_id
        veh_table.values(veh_id)['cluster_record'].tail.value['secondary_ch'].append(None)
        veh_table.values(veh_id)['cluster_record'].tail.value['start_time'] = sec
        veh_table.values(veh_id)['cluster_record'].tail.value['ef'] = ef
        veh_table.values(veh_id)['cluster_record'].tail.value['timer'] = 1
        veh_table.values(veh_id)['cluster_record'].tail.value['interrupt'] = list()

    veh_table.values(veh_id)['sub_cluster_members'] = set()
    veh_table.values(veh_id)['counter'] = config.counter
    veh_table.values(veh_id)['priority_ch'] = ch_id
    veh_table.values(veh_id)['priority_counter'] = config.priority_counter
    # bus_candidates.remove(bus_ch)
    veh_table.values(veh_id)['other_chs']. \
        update(veh_table.values(veh_id)['other_chs'].union(bus_candidates))

    veh_table.values(veh_id)['other_chs']. \
        update(veh_table.values(veh_id)['other_chs'].union(ch_candidates))
    if 'bus' in ch_id:
        bus_table.values(ch_id)['cluster_members'].add(veh_id)
        bus_table.values(ch_id).setdefault('direct_children', set()).add(veh_id)
        bus_table.values(ch_id)['gates'][veh_id] = veh_table.values(veh_id)['other_chs']
        bus_table.values(ch_id)['gate_chs']. \
            update(bus_table.values(ch_id)['gate_chs'].
                   union(veh_table.values(veh_id)['other_chs']))
    else:
        veh_table.values(ch_id)['cluster_members'].add(veh_id)
        veh_table.values(ch_id).setdefault('direct_children', set()).add(veh_id)
        veh_table.values(ch_id)['gates'][veh_id] = veh_table.values(veh_id)['other_chs']
        veh_table.values(ch_id)['gate_chs']. \
            update(veh_table.values(ch_id)['gate_chs'].
                   union(veh_table.values(veh_id)['other_chs']))
    stand_alone.remove(veh_id)
    zone_stand_alone[veh_table.values(veh_id)['macro_zone']].remove(veh_id)

    return (bus_table, veh_table,
            stand_alone, zone_stand_alone)


def add_sub_member(ch_id, bus_table,
                   veh_id, sub_ch_id, veh_table, config,
                   ef, sec, bus_candidates,
                   ch_candidates, stand_alone,
                   zone_stand_alone,
                   other_vehs):

    """

    :param ch_id:
    :param bus_table:
    :param veh_id:
    :param sub_ch_id:
    :param veh_table:
    :param config:
    :param ef:
    :param sec:
    :param bus_candidates:
    :param ch_candidates:
    :param stand_alone:
    :param zone_stand_alone:
    :param other_vehs:
    :return:
    """

    veh_table.values(veh_id)['primary_ch'] = ch_id
    veh_table.values(veh_id)['root_ch'] = ch_id
    veh_table.values(veh_id)['parent_id'] = sub_ch_id
    veh_table.values(veh_id)['parent_node'] = sub_ch_id

    if ch_id == veh_table.values(veh_id)['priority_ch']:
        veh_table.values(veh_id)['cluster_record'].pop()
        veh_table.values(veh_id)['cluster_record'].tail.value['secondary_ch'].append(sub_ch_id)
        veh_table.values(veh_id)['cluster_record'].tail.value['timer'] += 1
        veh_table.values(veh_id)['cluster_record'].tail.value['interrupt'].append(
            config.priority_counter - veh_table.values(veh_id)['priority_counter'])

    else:
        veh_table.values(veh_id)['cluster_record'].tail.key = ch_id
        veh_table.values(veh_id)['cluster_record'].tail.value['secondary_ch'].append(sub_ch_id)
        veh_table.values(veh_id)['cluster_record'].tail.value['start_time'] = sec
        veh_table.values(veh_id)['cluster_record'].tail.value['ef'] = ef
        veh_table.values(veh_id)['cluster_record'].tail.value['timer'] = 1
        veh_table.values(veh_id)['cluster_record'].tail.value['interrupt'] = list()

    veh_table.values(veh_id)['secondary_ch'] = sub_ch_id
    veh_table.values(veh_id)['hop_count'] = (veh_table.values(sub_ch_id).get('hop_count') or 1) + 1
    veh_table.values(veh_id)['role'] = 'CM'
    veh_table.values(veh_id)['root_connection_valid'] = True
    veh_table.values(sub_ch_id)['role'] = 'RN'
    veh_table.values(veh_id)['priority_ch'] = ch_id
    veh_table.values(veh_id)['priority_counter'] = config.priority_counter
    veh_table.values(veh_id)['counter'] = config.counter
    veh_table.values(sub_ch_id)['sub_cluster_members'].add(veh_id)
    veh_table.values(sub_ch_id).setdefault('direct_children', set()).add(veh_id)

    if 'bus' not in ch_id:
        if ch_id in ch_candidates:  # in multi-hop and when the vehicle joining the cluster through sub_ch,
            # the veh_ch might not be in the ch_candidates
            ch_candidates.remove(ch_id)
        # if a vehicle is added as sub_member, it would be in both sub_cluster_members
        # and cluster_members of the CH
        # veh_table.values(ch_id)['cluster_members'].add(veh_id)
        veh_table.values(ch_id)['sub_cluster_members'].add(veh_id)

    else:
        if ch_id in bus_candidates:  # in multi-hop and when the vehicle joining the cluster through sub_ch,
            # the veh_ch might not be in the bus_candidates
            bus_candidates.remove(ch_id)

        bus_table.values(ch_id)['sub_cluster_members'].add(veh_id)

    stand_alone.remove(veh_id)
    zone_stand_alone[veh_table.values(veh_id)['macro_zone']].remove(veh_id)


    return (bus_table, veh_table,
            stand_alone, zone_stand_alone)


def remove_member(mem, ch_id, veh_table, bus_table, config,
                  stand_alone, zone_stand_alone,
                  ch_stays=True, mem_stays=True):
    """
    This function would remove a cluster member from the cluster
    :param ch_stays:
    :type
    :param mem:
    :param ch_id:
    :param veh_table:
    :param bus_table:
    :param config:
    :param stand_alone:
    :param zone_stand_alone:
    :param mem_stays:
    :return:
    """
    # Use actual immediate parent pointers. The older sub_cluster_members
    # representation may include grandchildren but is not authoritative for
    # arbitrary-depth MMZCA trees.
    temp_sub_mem = {
        node_id for node_id in veh_table.ids() if node_id != mem and
        (veh_table.values(node_id).get('parent_node') or
         veh_table.values(node_id).get('parent_id') or
         veh_table.values(node_id).get('secondary_ch')) == mem
    }
    for s_m in temp_sub_mem:
        (veh_table, bus_table,
         stand_alone, zone_stand_alone) = remove_sub_member(s_m, mem, ch_id, veh_table, bus_table, config,
                                                            stand_alone, zone_stand_alone,
                                                            ch_stays=ch_stays)
    veh_table.values(mem)['counter'] = config.counter
    if ch_stays is True:
        veh_table.values(mem)['priority_ch'] = ch_id
    else:
        veh_table.values(mem)['priority_ch'] = None
    veh_table.values(mem)['priority_counter'] = config.priority_counter
    if 'bus' in ch_id:
        bus_table.values(ch_id)['cluster_members'].discard(mem)
        bus_table.values(ch_id).setdefault('direct_children', set()).discard(mem)
    else:
        veh_table.values(ch_id)['cluster_members'].discard(mem)
        veh_table.values(ch_id).setdefault('direct_children', set()).discard(mem)
    if mem_stays is True:
        stand_alone.add(mem)
        zone_stand_alone[veh_table.values(mem)['macro_zone']].add(mem)
    veh_table.values(mem)['primary_ch'] = None
    veh_table.values(mem)['secondary_ch'] = None
    veh_table.values(mem)['root_ch'] = None
    veh_table.values(mem)['parent_id'] = None
    veh_table.values(mem)['parent_node'] = None
    veh_table.values(mem)['hop_count'] = None
    veh_table.values(mem)['path_bottleneck'] = None
    veh_table.values(mem)['committed_link_score'] = None
    veh_table.values(mem)['root_connection_valid'] = False
    veh_table.values(mem)['direct_children'] = set()
    veh_table.values(mem)['sub_cluster_members'] = set()
    veh_table.values(mem)['kernel_descriptor'] = None
    veh_table.values(mem)['role'] = 'SA'
    veh_table.values(mem)['current_root_cost'] = None
    veh_table.values(mem)['current_parent_cost'] = None
    veh_table.values(mem)['cluster_record'].append(None, {'is_ch': False, 'secondary_ch': list(), 'start_time': None,
                                                          'ef': None, 'timer': None, 'interrupt': list()})

    return (veh_table, bus_table,
            stand_alone, zone_stand_alone)


def remove_sub_member(sub_mem_id, sub_ch_id, ch_id, veh_table, bus_table, config,
                      stand_alone, zone_stand_alone, ch_stays=True, sub_mem_stays=True,
                      _visited=None):
    """
    This function would remove a sub_cluster member from the cluster
    :type
    :param sub_mem_id:
    :param sub_ch_id:
    :param ch_id:
    :param veh_table:
    :param bus_table:
    :param config:
    :param net_graph:
    :param stand_alone:
    :param zone_stand_alone:
    :param ch_stays:
    :param sub_mem_stays:
    :return:
    """
    if _visited is None:
        _visited = set()
    if sub_mem_id in _visited:
        return veh_table, bus_table, stand_alone, zone_stand_alone
    _visited.add(sub_mem_id)

    # Detach descendants before their parent to avoid broken intermediate
    # paths during recursive MMZCA removal.
    descendants = {
        node_id for node_id in veh_table.ids() if node_id != sub_mem_id and
        (veh_table.values(node_id).get('parent_node') or
         veh_table.values(node_id).get('parent_id') or
         veh_table.values(node_id).get('secondary_ch')) == sub_mem_id
    }
    for descendant_id in descendants:
        veh_table, bus_table, stand_alone, zone_stand_alone = remove_sub_member(
            descendant_id, sub_mem_id, ch_id, veh_table, bus_table, config,
            stand_alone, zone_stand_alone, ch_stays=ch_stays,
            _visited=_visited
        )

    veh_table.values(sub_mem_id)['counter'] = config.counter
    if ch_stays is True:
        veh_table.values(sub_mem_id)['priority_ch'] = ch_id
    else:
        veh_table.values(sub_mem_id)['priority_ch'] = None
    veh_table.values(sub_mem_id)['priority_counter'] = config.priority_counter
    if sub_mem_stays is True:
        stand_alone.add(sub_mem_id)
        zone_stand_alone[veh_table.values(sub_mem_id)['macro_zone']].add(sub_mem_id)
    parent = veh_table.values(sub_ch_id)
    parent['sub_cluster_members'].discard(sub_mem_id)
    parent.setdefault('direct_children', set()).discard(sub_mem_id)
    if not parent['direct_children'] and parent.get('role') == 'RN':
        parent['role'] = 'CM'
    if 'bus' in ch_id:
        bus_table.values(ch_id)['sub_cluster_members'].discard(sub_mem_id)
    else:
        veh_table.values(ch_id)['sub_cluster_members'].discard(sub_mem_id)
    veh_table.values(sub_mem_id)['primary_ch'] = None
    veh_table.values(sub_mem_id)['secondary_ch'] = None
    veh_table.values(sub_mem_id)['root_ch'] = None
    veh_table.values(sub_mem_id)['parent_id'] = None
    veh_table.values(sub_mem_id)['parent_node'] = None
    veh_table.values(sub_mem_id)['hop_count'] = None
    veh_table.values(sub_mem_id)['path_bottleneck'] = None
    veh_table.values(sub_mem_id)['committed_link_score'] = None
    veh_table.values(sub_mem_id)['root_connection_valid'] = False
    veh_table.values(sub_mem_id)['direct_children'] = set()
    veh_table.values(sub_mem_id)['sub_cluster_members'] = set()
    veh_table.values(sub_mem_id)['kernel_descriptor'] = None
    veh_table.values(sub_mem_id)['role'] = 'SA'
    veh_table.values(sub_mem_id)['current_root_cost'] = None
    veh_table.values(sub_mem_id)['current_parent_cost'] = None
    veh_table.values(sub_mem_id)['cluster_record'].append(None,
                                                          {'is_ch': False, 'secondary_ch': list(), 'start_time': None,
                                                           'ef': None,  'timer': None, 'interrupt': list()})

    return (veh_table, bus_table,
            stand_alone, zone_stand_alone)


def set_ch(veh_id, veh_table, all_chs, stand_alone,
           zone_stand_alone, zone_ch, config, its_sa_clustering=False):
    """
    This function would update the information of a vehicle turning into a CH
    :param veh_id:
    :param veh_table:
    :param all_chs:
    :param stand_alone:
    :param zone_stand_alone:
    :param zone_ch:
    :param config:
    :param its_sa_clustering:
    :return:
    """
    veh_table.values(veh_id)['cluster_head'] = True
    veh_table.values(veh_id)['role'] = 'CH'
    veh_table.values(veh_id)['root_ch'] = veh_id
    veh_table.values(veh_id)['parent_id'] = None
    veh_table.values(veh_id)['parent_node'] = None
    veh_table.values(veh_id)['hop_count'] = 0
    veh_table.values(veh_id)['path_bottleneck'] = 0.0
    veh_table.values(veh_id)['committed_link_score'] = None
    veh_table.values(veh_id)['root_connection_valid'] = True
    veh_table.values(veh_id)['direct_children'] = set()
    veh_table.values(veh_id)['secondary_ch'] = None
    veh_table.values(veh_id)['cluster_record'].tail.value['is_ch'] = True
    veh_table.values(veh_id)['start_ch_zone'] = veh_table.values(veh_id)['macro_zone']
    all_chs.add(veh_id)
    zone_ch[veh_table.values(veh_id)['macro_zone']].add(veh_id)
    veh_table.values(veh_id)['counter'] = config.counter
    veh_table.values(veh_id)['priority_counter'] = config.priority_counter
    veh_table.values(veh_id)['priority_ch'] = None
    if its_sa_clustering is False:
        stand_alone.remove(veh_id)
        zone_stand_alone[veh_table.values(veh_id)['macro_zone']].remove(veh_id)
    else:
        try:
            stand_alone.remove(veh_id)
            zone_stand_alone[veh_table.values(veh_id)['macro_zone']].remove(veh_id)
        except KeyError:
            pass

    return (veh_table, all_chs, stand_alone,
            zone_stand_alone, zone_ch)


def set_ch_to_veh(veh_id, veh_table, zone_ch,
                  all_chs, stand_alone, zone_stand_alone):
    """
    This function would set a CH to a stand_alone vehicle
    :param veh_id:
    :param veh_table:
    :param zone_ch:
    :param all_chs:
    :param stand_alone:
    :param zone_stand_alone:
    :return:
    """
    veh_table.values(veh_id)['cluster_members'] = set()
    veh_table.values(veh_id)['sub_cluster_members'] = set()
    veh_table.values(veh_id)['cluster_head'] = False
    veh_table.values(veh_id)['role'] = 'SA'
    veh_table.values(veh_id)['root_ch'] = None
    veh_table.values(veh_id)['parent_id'] = None
    veh_table.values(veh_id)['parent_node'] = None
    veh_table.values(veh_id)['hop_count'] = None
    veh_table.values(veh_id)['path_bottleneck'] = None
    veh_table.values(veh_id)['committed_link_score'] = None
    veh_table.values(veh_id)['root_connection_valid'] = False
    veh_table.values(veh_id)['direct_children'] = set()
    veh_table.values(veh_id)['kernel_descriptor'] = None
    veh_table.values(veh_id)['secondary_ch'] = None
    veh_table.values(veh_id)['primary_ch'] = None
    veh_table.values(veh_id)['current_root_cost'] = None
    veh_table.values(veh_id)['current_parent_cost'] = None
    veh_table.values(veh_id)['cluster_record'].append(None, {'is_ch': False, 'secondary_ch': list(), 'start_time': None,
                                                 'ef': None,  'timer': None, 'interrupt': list()})
    veh_table.values(veh_id)['start_ch_zone'] = None
    zone_ch[veh_table.values(veh_id)['macro_zone']].remove(veh_id)
    all_chs.remove(veh_id)
    stand_alone.add(veh_id)
    zone_stand_alone[veh_table.values(veh_id)['macro_zone']].add(veh_id)
    return (veh_table, zone_ch, all_chs,
            stand_alone, zone_stand_alone)


def mac_address():
    """
    This function is used in the Main.py file
    :return: this function will return a generated random mac Address for vehicles
    """
    mac = [152, 237, 92,
           random.randint(0x00, 0x7f),
           random.randint(0x00, 0xff),
           random.randint(0x00, 0xff)]
    return ':'.join(map(lambda x: "%02x" % x, mac))


def middle_zone(u_row: object, u_col: object,
                l_row: object, l_col: object,
                n_cols: object) -> object:
    """
    This function is used in the zonex.py file
    :param u_row: The upper row id +1
    :param l_row: The lower row id +1
    :param u_col: The upper col id +1
    :param l_col: The lower col id +1
    :param n_cols: The maximum number of columns in the whole area
    :return: The zone in the middle of the inputs
    """
    # the almost centre zone id will be obtained here
    middle_row = int(np.floor((u_row + l_row) / 2))
    middle_col = int(np.floor((u_col + l_col) / 2))
    middle_zone_id = ((middle_row - 1) * n_cols) + middle_col - 1
    middle_row -= 1  # because the formal numbering has been considered form 0 not 1
    middle_col -= 1  # because the formal numbering has been considered form 0 not 1
    return 'zone' + str(middle_zone_id), middle_row, middle_col


def presence(area_cord, veh_cord):
    """
    This function will determine if the vehicle or bus is in the understudied area (padded area)
    :param area_cord: the coordination of the understudied area (padded area)
    :param veh_cord: the coordination of the vehicle or bus
    :return: it returns True or False
    """
    if (
            (float(veh_cord.getAttribute('x')) < area_cord['max_long']) and
            (float(veh_cord.getAttribute('x')) > area_cord['min_long']) and
            (float(veh_cord.getAttribute('y')) < area_cord['max_lat']) and
            (float(veh_cord.getAttribute('y')) > area_cord['min_lat'])
    ):
        return True
    else:
        return False


def det_near_ch(veh_id, veh_table, bus_table, zone_buses, zone_vehicles):
    """Existing reciprocal-range neighbor search, deduplicated and indexed."""
    src = veh_table.values(veh_id)
    if src is None:
        return set(), set(), set(), set()
    src_range = src['trans_range']
    buses, vehicles = set(), set()
    for zone in src['neighbor_zones']:
        buses.update(zone_buses[zone])
        vehicles.update(zone_vehicles[zone])
    bus_candidates, ch_candidates, cm_candidates, other_vehs = set(), set(), set(), set()
    for j in buses:
        dst = bus_table.values(j)
        if dst is not None and det_dist(veh_id, veh_table, j, bus_table) <= min(src_range, dst['trans_range']):
            bus_candidates.add(j)
    for j in vehicles:
        dst = veh_table.values(j)
        if dst is None:
            continue
        if det_dist(veh_id, veh_table, j, veh_table) > min(src_range, dst['trans_range']):
            continue
        if dst['cluster_head'] is True:
            ch_candidates.add(j)
        elif dst['primary_ch'] is not None:
            cm_candidates.add(j)
            other_vehs.add(j)
        else:
            other_vehs.add(j)
    return bus_candidates, ch_candidates, cm_candidates, other_vehs


def det_buses_other_ch(bus_id, veh_table, bus_table,
                       zone_buses, zone_chs):
    """

    :param bus_id:
    :param veh_table:
    :param bus_table:
    :param zone_buses:
    :param zone_chs:
    :return: the CHs in transmission range of bus_id
    """
    all_near_chs = set()
    all_chs = set()
    for zone in bus_table.values(bus_id)['neighbor_zones']:
        all_chs.update(all_chs.union(zone_chs[zone]))
        # all_chs.update(all_chs.union(zone_buses[zone]))
    for ch in all_chs:
        if ch == bus_id:
            continue
        elif 'bus' in ch:
            ch_table = bus_table
        elif 'veh' in ch:
            ch_table = veh_table
        else:
            continue
        euclidian_dist = det_dist(bus_id, bus_table, ch, ch_table)

        if euclidian_dist <= min(bus_table.values(bus_id)['trans_range'], ch_table.values(ch)['trans_range']):
            all_near_chs.add(ch)
    return all_near_chs


def choose_ch(table, veh_table_i, area_zones, candidates, config):
    """
    This function will be used to choose a ch among all other candidates or a ch from other chs nearby as the vehicle's
    primary_ch. The Factors are [proposed similarity factor, speed similarity, distance].
    """

    # Latitude and longitude of the centre of the previous zone that the vehicle was in
    prev_veh_lat = (area_zones.zone_hash.values(veh_table_i['prev_macro_zone'])['max_lat'] +
                    area_zones.zone_hash.values(veh_table_i['prev_macro_zone'])['min_lat']) / 2
    prev_veh_long = (area_zones.zone_hash.values(veh_table_i['prev_macro_zone'])['max_long'] +
                     area_zones.zone_hash.values(veh_table_i['prev_macro_zone'])['min_long']) / 2

    euclidean_distance = hs.haversine((prev_veh_lat, prev_veh_long),
                                      (veh_table_i['lat'], veh_table_i['long']),
                                      unit=hs.Unit.METERS)

    veh_alpha = np.arctan2((veh_table_i['long'] - prev_veh_long), (veh_table_i['lat'] - prev_veh_lat))

    veh_vector_x = np.multiply(euclidean_distance, np.cos(veh_alpha))
    veh_vector_y = np.multiply(euclidean_distance, np.sin(veh_alpha))

    min_ef = float('inf')
    nominee = None

    for j in candidates:
        # Latitude and longitude of the centre of the previous zone that the candidate was in
        prev_ch_lat = (area_zones.zone_hash.values(table.values(j)['prev_macro_zone'])['max_lat'] +
                       area_zones.zone_hash.values(table.values(j)['prev_macro_zone'])['min_lat']) / 2
        prev_ch_long = (area_zones.zone_hash.values(table.values(j)['prev_macro_zone'])['max_long'] +
                        area_zones.zone_hash.values(table.values(j)['prev_macro_zone'])['min_long']) / 2

        euclidean_distance_ch = hs.haversine((prev_ch_lat, prev_ch_long),
                                             (table.values(j)['lat'], table.values(j)['long']),
                                             unit=hs.Unit.METERS)

        ch_alpha = np.arctan2((table.values(j)['long'] - prev_ch_long), (table.values(j)['lat'] - prev_ch_lat))

        ch_vector_x = np.multiply(euclidean_distance_ch, np.cos(ch_alpha))
        ch_vector_y = np.multiply(euclidean_distance_ch, np.sin(ch_alpha))

        # Calculate cosine similarity
        cos_sim = 1 - spatial.distance.cosine([veh_vector_x, veh_vector_y], [ch_vector_x, ch_vector_y])
        theta_sim = np.arccos(np.clip(cos_sim, -1.0, 1.0)) / (2 * np.pi)  # Ensure cos_sim is within valid range

        theta_dist = euclidean_distance / min(table.values(j)['trans_range'], veh_table_i['trans_range'])

        with np.errstate(divide='ignore', invalid='ignore'):
            speed_sim = np.divide(np.abs(table.values(j)['speed'] - veh_table_i['speed']) + 0.001,
                                  max(np.abs(table.values(j)['speed']), veh_table_i['speed']) + 0.001)

        # Calculate the Eligibility Factor (EF) for candidates
        weights = np.divide(config.weights_s, sum(config.weights_s))  # Normalize the weights
        ef = np.matmul(np.transpose(weights), np.array([theta_sim, speed_sim, theta_dist]))

        if ef < min_ef:
            min_ef = ef
            nominee = j

    return nominee, min_ef


def choose_multihop_ch(veh_id, veh_table, bus_table, bus_candidates,
                       ch_candidates, sub_ch_candidates, area_zones, config, beta=None):
    beta_ch, beta_bus = det_beta(bus_candidates, ch_candidates, sub_ch_candidates)

    # latitude of the centre of previous zone that vehicle were in
    prev_veh_lat = (area_zones.zone_hash.values(veh_table.values(veh_id)['prev_macro_zone'])['max_lat'] +
                    area_zones.zone_hash.values(veh_table.values(veh_id)['prev_macro_zone'])['min_lat']) / 2
    # longitude of the centre of previous zone that vehicle were in
    prev_veh_long = (area_zones.zone_hash.values(veh_table.values(veh_id)['prev_macro_zone'])['max_long'] +
                     area_zones.zone_hash.values(veh_table.values(veh_id)['prev_macro_zone'])['min_long']) / 2

    euclidian_distance = hs.haversine((prev_veh_lat, prev_veh_long),
                                      (veh_table.values(veh_id)['lat'], veh_table.values(veh_id)['long']),
                                      unit=hs.Unit.METERS)

    veh_alpha = np.arctan2((prev_veh_long - veh_table.values(veh_id)['long']),
                          (prev_veh_lat - veh_table.values(veh_id)['lat']))

    veh_vector_x = np.multiply(euclidian_distance, np.cos(veh_alpha))
    veh_vector_y = np.multiply(euclidian_distance, np.sin(veh_alpha))

    min_ef = 1000000
    nominee = None
    for j in bus_candidates.union(ch_candidates, sub_ch_candidates):
        beta, table = (beta_bus, bus_table) if 'bus' in j else (beta_ch, veh_table)

        # latitude of the centre of previous zone that ch were in
        prev_ch_lat = (area_zones.zone_hash.values(table.values(j)['prev_macro_zone'])['max_lat'] +
                       area_zones.zone_hash.values(table.values(j)['prev_macro_zone'])['min_lat']) / 2
        # latitude of the centre of previous zone that ch were in
        prev_ch_long = (area_zones.zone_hash.values(table.values(j)['prev_macro_zone'])['max_long'] +
                        area_zones.zone_hash.values(table.values(j)['prev_macro_zone'])['min_long']) / 2

        euclidian_distance = hs.haversine((prev_ch_lat, prev_ch_long),
                                          (table.values(j)['lat'], table.values(j)['long']),
                                          unit=hs.Unit.METERS)

        ch_alpha = np.arctan2((prev_veh_long - veh_table.values(veh_id)['long']),
                             (prev_veh_lat - veh_table.values(veh_id)['lat']))

        ch_vector_x = np.multiply(euclidian_distance, np.cos(ch_alpha))
        ch_vector_y = np.multiply(euclidian_distance, np.sin(ch_alpha))

        cos_sim = 1 - spatial.distance.cosine([veh_vector_x, veh_vector_y], [ch_vector_x, ch_vector_y])
        theta_sim = np.arccos(np.clip(cos_sim, -1.0, 1.0)) / (2 * np.pi)

        theta_dist = euclidian_distance / min(table.values(j)['trans_range'], veh_table.values(veh_id)['trans_range'])
        # since it might return RuntimeWarning regarding the division, the warning will be ignored
        with np.errstate(divide='ignore', invalid='ignore'):
            speed_sim = np.divide(np.abs(table.values(j)['speed'] - veh_table.values(veh_id)['speed']),
                                  max(np.abs(table.values(j)['speed']), veh_table.values(veh_id)['speed']))

        # calculate the Eligibility Factor (EF) for chs
        weights = np.divide(config.weights_m, sum(config.weights_m))  # normalizing the weights
        ef = np.matmul(np.transpose(weights),
                       np.array([theta_sim, speed_sim, theta_dist]))
        if ('veh' in j) and (table.values(j)['primary_ch'] is not None)\
                and (table.values(j)['secondary_ch'] is None):
            ef = np.average([ef, table.values(j)['cluster_record'].tail.value['ef']])

        ef *= beta  # the way beta is working is that if both chs, sub_chs, and buses are nearby, the beta
        # related to the buses would be multiplied if the connection would be single-hop not multi-hop. Hence, for a
        # sub_ch connected to a buss the beta would be same as beta for chs
        if ef < min_ef:
            min_ef = ef
            nominee = j
    return nominee, min_ef


def priority_clusters(veh_id, veh_table, bus_candidates,
                      ch_candidates, sub_ch_candidates):
    """
    This function would just consider the sub_chs that are belong to the previous cluster that veh_id used to be in it
    and disregard the rest of the chs and buses nearby
    :param veh_id:
    :param veh_table:
    :param bus_candidates:
    :param ch_candidates:
    :param sub_ch_candidates:
    :return:
    """
    if 'veh' in veh_table.values(veh_id)['priority_ch']:
        for prior in ch_candidates:
            if prior == veh_table.values(veh_id)['priority_ch']:
                ch_candidates = set()
                ch_candidates.add(prior)
                bus_candidates = set()

        temp_sub_ch_candidates = set()
        for prior in sub_ch_candidates:
            if veh_table.values(prior)['primary_ch'] == veh_table.values(veh_id)['priority_ch']:
                temp_sub_ch_candidates.add(prior)
        if len(temp_sub_ch_candidates) != 0:
            sub_ch_candidates = temp_sub_ch_candidates.copy()
            bus_candidates = set()

    elif 'bus' in veh_table.values(veh_id)['priority_ch']:
        for prior in bus_candidates:
            if prior == veh_table.values(veh_id)['priority_ch']:
                bus_candidates = set()
                bus_candidates.add(prior)
                ch_candidates = set()

        temp_sub_ch_candidates = set()
        for prior in sub_ch_candidates:
            if veh_table.values(prior)['primary_ch'] == veh_table.values(veh_id)['priority_ch']:
                temp_sub_ch_candidates.add(prior)
        if len(temp_sub_ch_candidates) != 0:
            sub_ch_candidates = temp_sub_ch_candidates.copy()
            ch_candidates = set()

    return bus_candidates, ch_candidates, sub_ch_candidates


def det_beta(bus_candidates, ch_candidates,
             sub_ch_candidates):
    """
    This function would determine the beta coefficient that gives priority to the buses and other chs (or sub_chs)
    based on the ratio of number of buses and number of chs(or sub_chs)
    :param bus_candidates:
    :param ch_candidates:
    :param sub_ch_candidates:
    :return:
    """
    num_bus: int = len(bus_candidates)
    num_ch_sub_ch: int = len(ch_candidates) + len(sub_ch_candidates)

    beta_ch = 1
    beta_bus = 1
    beta_bus = 0.70 if (((num_ch_sub_ch+1)/(num_bus+1)) <= 1) else beta_bus
    beta_bus = 0.80 if (1 < (num_ch_sub_ch + 1) / (num_bus + 1) < 2.5) and (num_bus != 0) else beta_bus
    beta_bus = 0.95 if (2.5 <= (num_ch_sub_ch + 1) / (num_bus + 1)) and (num_bus != 0) else beta_bus

    return beta_ch, beta_bus


def update_bus_table(veh, bus_table, micro_zone_id, meso_zone_id, macro_zone_id, understudied_area, zones, config, zone_buses, zone_ch, current_time):
    """
    this function updates the bus_tabel and zone_buses from main.py
    :param current_time: "self.time" in the data_cluster.py
    :param zone_ch: the self.zone_ch dictionary
    :param veh: it's the veh from .xml file
    :param bus_table: its bus_table
    :param micro_zone_id: the zon_id f the vehicle
    :param meso_zone_id: the zon_id f the vehicle
    :param macro_zone_id: the zon_id f the vehicle
    :param understudied_area: the un_padded area
    :param zones: the area_zones.zones() or zones table from the DataTable class in the main.py
    :param config:
    :param zone_buses: zone_buses from the DataTable class in the main.py
    :return: updated bus_table and zone_buses
    """
    veh_id = veh.getAttribute('id')
    if veh_id in bus_table.ids():
        veh_bus_table = bus_table.values(veh_id)
        veh_bus_table['meso_zone_changed_this_tick'] = (
            veh_bus_table['meso_zone'] != meso_zone_id
        )
        veh_bus_table['macro_zone_changed_this_tick'] = (
            veh_bus_table['macro_zone'] != macro_zone_id
        )
        if veh_bus_table['micro_zone'] != micro_zone_id:
            veh_bus_table['prev_micro_zone'] = \
                veh_bus_table['micro_zone']  # update prev_micro_zone

        if veh_bus_table['meso_zone'] != meso_zone_id:
            veh_bus_table['prev_meso_zone'] = \
                veh_bus_table['meso_zone']  # update prev_meso_zone

        if veh_bus_table['macro_zone'] != macro_zone_id:
            veh_bus_table['prev_macro_zone'] = \
                veh_bus_table['macro_zone']  # update prev_macro_zone

        zone_buses[veh_bus_table['macro_zone']]. \
            remove(veh_id)  # This will remove the vehicle from its previous zone_buses
        zone_ch[veh_bus_table['macro_zone']]. \
            remove(veh_id)
        if veh_bus_table['lane']['id'] == veh.getAttribute('lane'):
            veh_bus_table['lane']['timer'] += 1
        else:
            veh_bus_table['lane']['id'] = veh.getAttribute('lane')
            veh_bus_table['lane']['timer'] = 0
        veh_bus_table['long'] = float(veh.getAttribute('x'))
        veh_bus_table['lat'] = float(veh.getAttribute('y'))
        veh_bus_table['angle'] = float(veh.getAttribute('angle'))
        veh_bus_table['speed'] = float(veh.getAttribute('speed')) + 0.01
        veh_bus_table['pos'] = float(veh.getAttribute('pos'))
        veh_bus_table['micro_zone'] = micro_zone_id
        veh_bus_table['meso_zone'] = meso_zone_id
        veh_bus_table['macro_zone'] = macro_zone_id
        veh_bus_table['in_area'] = presence(understudied_area, veh)
        veh_bus_table['neighbor_zones'] = zones.neighbor_zones(macro_zone_id)
        veh_bus_table['gate_chs'] = set()
        veh_bus_table['gates'] = dict()
        veh_bus_table['other_chs'] = set()
        veh_bus_table['other_vehs'] = set()
        zone_buses[macro_zone_id].add(veh_id)
        zone_ch[macro_zone_id].add(veh_id)
    else:
        bus_table.set_item(veh_id, initiate_new_bus(veh, zones, micro_zone_id, meso_zone_id, macro_zone_id,
                                                                    config, understudied_area))
        bus_table.values(veh_id)['arrive_time'] = current_time
        zone_buses[macro_zone_id].add(veh_id)
        zone_ch[macro_zone_id].add(veh_id)

    return bus_table, zone_buses, zone_ch


def update_veh_table(veh, veh_table, micro_zone_id, meso_zone_id, macro_zone_id, understudied_area, zones, config,
                     zone_vehicles, zone_ch, stand_alone, zone_stand_alone, current_time):
    """
    this function updates the veh_tabel and zone_vehicles from main.py
    :param current_time: self.time in data_cluster.py
    :param zone_stand_alone: its the self.zone_stand_alone
    :param stand_alone: its the self.stand_alone
    :param zone_ch: the self.zone_ch dictionary
    :param veh: it's the veh from .xml file
    :param veh_table: its veh_table
    :param micro_zone_id: the zon_id f the vehicle
    :param meso_zone_id: the zon_id f the vehicle
    :param macro_zone_id: the zon_id f the vehicle
    :param understudied_area: the un_padded area
    :param zones: the area_zones.zones() or zones table from the DataTable class in the main.py
    :param config:
    :param zone_vehicles: zone_vehicles from the DataTable class in the main.py
    :return: updated veh_table, zone_vehicles, zone_ch, stand_alone, zone_stand_alone
    """
    veh_id = veh.getAttribute('id')
    if veh_id in veh_table.ids():
        veh_veh_table = veh_table.values(veh_id)
        old_macro_zone = veh_veh_table['macro_zone']
        veh_veh_table['meso_zone_changed_this_tick'] = (
            veh_veh_table['meso_zone'] != meso_zone_id
        )
        veh_veh_table['macro_zone_changed_this_tick'] = (
            veh_veh_table['macro_zone'] != macro_zone_id
        )
        if veh_veh_table['micro_zone'] != micro_zone_id:
            veh_veh_table['prev_micro_zone'] = \
                veh_veh_table['micro_zone']  # update prev_micro_zone

        if veh_veh_table['meso_zone'] != meso_zone_id:
            veh_veh_table['prev_meso_zone'] = \
                veh_veh_table['meso_zone']  # update prev_meso_zone

        if veh_veh_table['macro_zone'] !=  macro_zone_id:
            veh_veh_table['prev_macro_zone'] = \
                veh_veh_table['macro_zone']  # update prev_macro_zone

        zone_vehicles[veh_veh_table['macro_zone']]. \
            remove(veh_id)  # remove the vehicle from its previous zone_vehicles
        if veh_veh_table['cluster_head'] is True:
            zone_ch[veh_veh_table['macro_zone']]. \
                remove(veh_id)
        if veh_veh_table['lane']['id'] == veh.getAttribute('lane'):
            veh_veh_table['lane']['timer'] += 1
        else:
            lane_id = veh.getAttribute('lane')
            if ":" not in lane_id:
                pattern = re.compile(f"^(.*?){re.escape('_')}")
                match = pattern.search(lane_id)
                if match:
                    lane_id = match.group(1)
            veh_veh_table['lane']['id'] = lane_id
            veh_veh_table['lane']['timer'] = 0
        veh_veh_table['long'] = float(veh.getAttribute('x'))
        veh_veh_table['lat'] = float(veh.getAttribute('y'))
        veh_veh_table['angle'] = float(veh.getAttribute('angle'))
        veh_veh_table['speed'] = float(veh.getAttribute('speed')) + 0.01
        veh_veh_table['pos'] = float(veh.getAttribute('pos'))
        veh_veh_table['micro_zone'] = micro_zone_id
        veh_veh_table['meso_zone'] = meso_zone_id
        veh_veh_table['macro_zone'] = macro_zone_id
        veh_veh_table['in_area'] = presence(understudied_area, veh)
        veh_veh_table['neighbor_zones'] = zones.neighbor_zones(macro_zone_id)
        veh_veh_table['gate_chs'] = set()
        veh_veh_table['gates'] = dict()
        veh_veh_table['other_chs'] = set()
        veh_veh_table['other_vehs'] = set()
        zone_vehicles[macro_zone_id].add(veh_id)
        if veh_veh_table['cluster_head'] is True:
            zone_ch[macro_zone_id].add(veh_id)
        elif (veh_veh_table['cluster_head'] is False) and \
                (veh_veh_table['primary_ch'] is None):
            stand_alone.add(veh_id)
            zone_stand_alone[old_macro_zone].discard(veh_id)

            zone_stand_alone[macro_zone_id].add(veh_id)

    else:
        veh_table.set_item(veh_id, initiate_new_veh(veh, zones,  micro_zone_id, meso_zone_id, macro_zone_id,
                                                                    config, understudied_area))
        veh_table.values(veh_id)['arrive_time'] = current_time
        zone_vehicles[macro_zone_id].add(veh_id)
        stand_alone.add(veh_id)
        zone_stand_alone[veh_table.values(veh_id)['macro_zone']].add(veh_id)

    return veh_table, zone_vehicles, zone_ch, stand_alone, zone_stand_alone


def det_near_sa(veh_id, veh_table, stand_alone, zone_stand_alone):
    """Return feasible SA neighbors once each, preserving reciprocal-range rule."""
    src = veh_table.values(veh_id)
    if src is None:
        return set()
    others = set()
    for zone in src['neighbor_zones']:
        others.update(zone_stand_alone[zone])
    others.discard(veh_id)
    src_range = src['trans_range']
    result = set()
    for j in others:
        dst = veh_table.values(j)
        if dst is not None and det_dist(veh_id, veh_table, j, veh_table) <= min(src_range, dst['trans_range']):
            result.add(j)
    return result


def update_sa_net_graph(veh_table, k, near_sa, net_graph):
    """
    this function is used to determine chs among the stand-alones to k which is ch too
    :param veh_table:
    :param k:
    :param near_sa:
    :param net_graph:
    :return: chs among the stand-alones to k which is ch too
    """
    for j in near_sa[k]:
        if veh_table.values(k)['cluster_head'] + veh_table.values(j)['cluster_head'] > 0:
            dist = det_dist(k, veh_table, j, veh_table)

            if dist < min(veh_table.values(k)['cluster_head'],
                          veh_table.values(j)['cluster_head']):
                if veh_table.values(k)['cluster_head'] + veh_table.values(j)['cluster_head'] == 2:
                    veh_table.values(k)['other_chs'].add(j)
                    veh_table.values(j)['other_chs'].add(k)
                    net_graph.add_edges_from([(k, j),
                                        (j, k)])

                elif (veh_table.values(k)['cluster_head'] is True) and \
                        (veh_table.values(j)['cluster_head'] is False):

                    if veh_table.values(j)['primary_ch'] != k:
                        veh_table.values(j)['other_chs'].add(k)
                        veh_table.values(veh_table.values(j)['primary_ch'])['gates'][j].add(k)
                        veh_table.values(veh_table.values(j)['primary_ch'])['gate_chs'].add(k)

                elif (veh_table.values(j)['cluster_head'] is True) and \
                        (veh_table.values(k)['cluster_head'] is False):

                    if veh_table.values(k)['primary_ch'] != j:
                        veh_table.values(k)['other_chs'].add(j)
                        veh_table.values(veh_table.values(k)['primary_ch'])['gates'][k].add(j)
                        veh_table.values(veh_table.values(k)['primary_ch'])['gate_chs'].add(j)

    return veh_table, net_graph


def det_dist(id1, table1, id2, table2):
    """Original haversine metric with symmetric tick-local memoization.

    Coordinates are part of the key, so same-tick mobility mutations do not
    produce stale results in tests or during simulation.
    """
    a, b = table1.values(id1), table2.values(id2)
    a_lat, a_lon = a['lat'], a['long']
    b_lat, b_lon = b['lat'], b['long']
    cache = globals().get('_MMZCA_DISTANCE_CACHE')
    if cache is None:
        return hs.haversine((a_lat, a_lon), (b_lat, b_lon), unit=hs.Unit.METERS)
    left, right = (id(a), a_lat, a_lon), (id(b), b_lat, b_lon)
    key = (left, right) if left[0] <= right[0] else (right, left)
    try:
        return cache[key]
    except KeyError:
        value = hs.haversine((a_lat, a_lon), (b_lat, b_lon), unit=hs.Unit.METERS)
        cache[key] = value
        return value


def det_pot_ch(veh_id, near_sa, n_near_sa):
    """

    :param veh_id:
    :param near_sa: dictionary shows the SAVs and their nearby SAVs
    :param n_near_sa: number of stand_alone vehicles to veh_id
    :return: potential SAVs nearby veh_id that can be CH
    """
    pot_ch = veh_id
    for mem in near_sa[veh_id]:
        if mem in near_sa.keys():
            if n_near_sa[mem] > n_near_sa[pot_ch]:
                pot_ch = mem
            else:
                continue
        else:
            continue
    return pot_ch


def det_pot_ch_dsca(veh_id, near_sa, n_near_sa, sf_factor):
    """

    :param veh_id:
    :param near_sa: dictionary shows the SAVs and their nearby SAVs
    :param n_near_sa: number of stand_alone vehicles to veh_id
    :return: potential SAVs nearby veh_id that can be CH
    """
    pot_ch = veh_id
    for mem in near_sa[veh_id]:
        if mem in near_sa.keys():
            if sf_factor[mem] > sf_factor[pot_ch]:
                pot_ch = mem
            else:
                continue
        else:
            continue
    return pot_ch


def save_img(m, zoom_out_value, name):
    """
    Saves a screenshot of a map to an image file.

    :param m: The map object to save as an image.
    :param zoom_out_value: The amount to zoom out on the map.
    :param name: The name of the image file to save.
    """
    # Save the map as an HTML file
    map_file = "/Users/pouyafirouzmakan/Desktop/slideshow/map.html"
    m.save(map_file)

    # Setup the Firefox WebDriver with headless option
    service = Service(executable_path='/usr/local/bin/geckodriver')
    options = Options()
    options.headless = True

    # Initialize the Firefox webdriver
    driver = webdriver.Firefox(service=service, options=options)

    # Open the map HTML file using the correct file URL format
    driver.get(f'file://{map_file}')

    # Change the zoom level of the map using JavaScript
    script = f"document.getElementsByClassName('leaflet-control')[0].style.transform = 'scale({1 / zoom_out_value})';"
    driver.execute_script(script)

    # Wait for the map to load
    time.sleep(1)

    # Take a screenshot of the entire webpage (including the map)
    screenshot = driver.get_screenshot_as_png()

    # Save the screenshot as an image file with good resolution
    img = Image.open(BytesIO(screenshot))
    img.save(name + '.png', 'PNG', quality=95)

    # Close the Firefox window and quit the webdriver instance
    driver.quit()

    # Delete the temporary HTML file
    os.remove(map_file)

def image_num(filename):
    """

    :param filename: name of the image
    :return: returns the number of image in the input folder to crearte the slideshow
    """
    # Extract the numeric part of the filename (assuming "Graph<number>.jpg" format)
    return int(''.join(filter(str.isdigit, filename)))


def make_slideshow(image_folder, output_path, fps):
    """
    :param image_folder: the images directory
    :param output_path: the output directory
    :param fps: play-back speed
    :return: it will create a mp4 video of the map_saved_imgs
    """
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

    if not image_files:
        print("No images found in the specified folder.")
        return

    # Sort the image files based on the numeric part of the filename
    image_files.sort(key=image_num)

    # Determine the width and height of the images from the first image in the folder
    image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(image_path)
    height, width, _ = first_image.shape

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' on Windows
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)

        # Add the frame to the video
        out.write(frame)

    # Release the VideoWriter and close the video file
    out.release()

    print("Video creation complete.")


def sumo_net_info(sumo_edge, sumo_node):
    """
    In this function, the information related to the sumo and its net are going to be put inside two dictionaries to
    clarify and use them for creating the papers we are going to make comparison with them
    :param sumo_edge: xml.dom.minidom.parse(sumo_edge_path) which is related to roads segments info
    :param sumo_node: xml.dom.minidom.parse(sumo_node_path) which is related to nodes info
    :return: edge_info and node_info
    """
    edge_info = dict()
    node_info = dict()
    for edge in sumo_edge.documentElement.getElementsByTagName('edge'):
        # just "from" is enough to define the T_leave
        if ':' in edge.getAttribute('id'):
            edge_info[edge.getAttribute('id')] = {'from': None}
        else:
            edge_info[edge.getAttribute('id')] = {'from': edge.getAttribute('from')}
            edge_info[edge.getAttribute('id')]['length'] = float(edge.getElementsByTagName('lane')[0].
                                                                 getAttribute('length'))

    for node in sumo_node.documentElement.getElementsByTagName('node'):
        # Note that in the osm_bbox.osm.xml file, the attributes are "lat" and "lon" and is not "long"
        node_info[node.getAttribute('id')] = {'lat': float(node.getAttribute('lat')),
                                              'long': float(node.getAttribute('lon'))
                                              }
    return edge_info, node_info


def other_connections_update(veh_table, bus_table, zone_ch,
                             zone_buses, zone_vehicles):
    """
    In this funtion, the other_chs, other_vehs, gate_chs, and gates would be updated after
    :param veh_table:
    :param bus_table:
    :param zone_ch:
    :param zone_buses:
    :param zone_vehicles:
    :return:
    """
    for veh_id in veh_table.ids():
        (bus_candidates, ch_candidates,
         sub_ch_candidates, other_vehs) = det_near_ch(veh_id, veh_table, bus_table, zone_buses, zone_vehicles)

        if veh_table.values(veh_id)['primary_ch'] is not None:
            ch = veh_table.values(veh_id)['primary_ch']
            secondary_ch = veh_table.values(veh_id)['secondary_ch']
            veh_table.values(veh_id)['other_chs'] = bus_candidates.union(ch_candidates) - {ch}
            veh_table.values(veh_id)['other_vehs'] = other_vehs - {secondary_ch} - {veh_id}
            if 'veh' in ch:
                veh_table.values(ch)['gates'][veh_id] = veh_table.values(veh_id)['other_chs']
                veh_table.values(ch)['gate_chs'].update(veh_table.values(ch)['gate_chs'].
                                                           union(veh_table.values(veh_id)['other_chs']))
            else:
                bus_table.values(ch)['gates'][veh_id] = veh_table.values(veh_id)['other_chs']
                bus_table.values(ch)['gate_chs'].update(bus_table.values(ch)['gate_chs'].
                                                           union(veh_table.values(veh_id)['other_chs']))

        elif veh_table.values(veh_id)['cluster_head'] is True:
            veh_table.values(veh_id)['other_chs'] = bus_candidates.union(ch_candidates)
            veh_table.values(veh_id)['other_vehs'] = (other_vehs - veh_table.values(veh_id)['cluster_members'] -
                                                      veh_table.values(veh_id)['sub_cluster_members']) - {veh_id}

        else:
            veh_table.values(veh_id)['other_vehs'] = other_vehs - {veh_id}

    for bus in bus_table.ids():
        bus_other_vehs = set()
        neigh_veh = list()

        for neigh_z in bus_table.values(bus)['neighbor_zones']:
            neigh_veh += zone_vehicles[neigh_z]

        for j in neigh_veh:
            euclidian_dist = det_dist(bus, bus_table, j, veh_table)

            if euclidian_dist <= min(bus_table.values(bus)['trans_range'],
                                     veh_table.values(j)['trans_range']):
                if veh_table.values(j)['primary_ch'] != bus:
                    bus_other_vehs.add(j)
        bus_table.values(bus)['other_vehs'] = (bus_other_vehs - bus_table.values(bus)['cluster_members'] -
                                               bus_table.values(bus)['sub_cluster_members'])

        bus_table.values(bus)['other_chs'] = det_buses_other_ch(bus, veh_table,bus_table, zone_buses, zone_ch)

    return veh_table, bus_table


"""MMZCA kernel utilities to append to the existing ``utils/util.py``.

No replacement data store is introduced: every node and kernel descriptor lives
in the existing ``veh_table`` or ``bus_table`` HashTable record.

Call ``mmzca_kernel_tick`` once per simulation tick AFTER cluster maintenance.
It models subtree-histogram updates and sketch distribution as one-hop messages
delivered at the next beacon tick. The caller owns the message queue and the
root-incarnation registry, e.g. as attributes of its existing DataTable.
"""

import math


def _mmzca_node(node_id, veh_table, bus_table):
    """Return (record, containing HashTable), or (None, None)."""
    record = veh_table.values(node_id)
    if record is not None:
        return record, veh_table
    record = bus_table.values(node_id)
    return (record, bus_table) if record is not None else (None, None)


def _mmzca_is_root(node_id, veh_table, bus_table):
    record, table = _mmzca_node(node_id, veh_table, bus_table)
    return record is not None and (table is bus_table or record.get('cluster_head') is True)


def _mmzca_root_id(node_id, record, veh_table, bus_table):
    if _mmzca_is_root(node_id, veh_table, bus_table):
        return node_id
    return record.get('root_ch') or record.get('primary_ch')


def _mmzca_parent_id(record):
    # Generalized fields first; retain legacy SMZCA fields as fallback.
    return (record.get('parent_node') or record.get('parent_id') or
            record.get('secondary_ch') or record.get('primary_ch'))


def _mmzca_micro_rc(zone_id, micro_zones):
    """Existing zone IDs are 'mic_zoneN', with zero-based row-major N."""
    if not isinstance(zone_id, str) or not zone_id.startswith('mic_zone'):
        return None
    try:
        index = int(zone_id[len('mic_zone'):])
        if not 0 <= index < micro_zones.n_rows * micro_zones.n_cols:
            return None
        return divmod(index, micro_zones.n_cols)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def mmzca_kernel_radius(config):
    """r_K in MICRO-cell units; alpha_micro is in km, ranges are in m.

    ``kernel_max_edge_range_m`` must be an actual upper bound on *all*
    committed parent-link lengths. Using the largest configured vehicle/bus
    transmission range gives a conservative bound for reciprocal links.
    """
    h_max = int(config.h_max)
    alpha_micro_km = float(config.alpha_micro)
    max_range_m = float(config.kernel_max_edge_range_m)
    if h_max < 1 or alpha_micro_km <= 0 or max_range_m <= 0:
        raise ValueError('Invalid H_max, alpha_micro, or kernel edge-length bound')
    return int(math.ceil((h_max - 1) * max_range_m / (1000.0 * alpha_micro_km)))


def _mmzca_midpoint_quantiles(weighted_values, count):
    """M inverse-empirical-CDF samples at q_l=(l+1/2)/M.

    weighted_values is an iterable of (coordinate, positive multiplicity).
    Counts preserve the member-weighted occupancy distribution without
    expanding every histogram entry into an individual observation.
    """
    items = sorted((float(v), int(w)) for v, w in weighted_values if int(w) > 0)
    total = sum(w for _, w in items)
    if total == 0 or count <= 0:
        return []
    samples = []
    cumulative = 0
    idx = 0
    for ell in range(count):
        target = (ell + 0.5) * total / count
        while idx < len(items) - 1 and cumulative + items[idx][1] < target:
            cumulative += items[idx][1]
            idx += 1
        samples.append(items[idx][0])
    return samples


def mmzca_build_kernel(root_id, incarnation, snapshot_version, now,
                       root_micro_zone, absolute_histogram, micro_zones,
                       config):
    """Build K_h from ACCEPTED sparse subtree reports at root h.

    absolute_histogram maps micro-zone ID to the COUNT of eligible CMs.
    It must exclude the root and depth-H_max members. This function does not
    scan all members or reconstruct a global cluster-membership list.
    """
    anchor = _mmzca_micro_rc(root_micro_zone, micro_zones)
    if anchor is None:
        return None
    r_k = mmzca_kernel_radius(config)
    M = int(config.kernel_quantile_samples)
    if M < 1:
        raise ValueError('kernel_quantile_samples must be positive')

    rho_weights = []
    sigma_weights = []
    for zone_id, count in absolute_histogram.items():
        rc = _mmzca_micro_rc(zone_id, micro_zones)
        if rc is None or int(count) <= 0:
            continue
        u, v = rc[0] - anchor[0], rc[1] - anchor[1]
        rho_weights.append((u + v, count))
        sigma_weights.append((u - v, count))

    # For H_max=1, no CM is eligible to act as a parent.
    rho_hat = _mmzca_midpoint_quantiles(rho_weights, M) if r_k > 0 else []
    sigma_hat = _mmzca_midpoint_quantiles(sigma_weights, M) if r_k > 0 else []
    return {
        'root_id': root_id,
        'incarnation': incarnation,
        'snapshot_version': snapshot_version,
        'generation_time': now,
        'anchor_rc': anchor,
        'k': 2 * r_k + 1,
        'M': M,
        'rho_samples': rho_hat,
        'sigma_samples': sigma_hat,
    }


def mmzca_kernel_is_usable(kernel, root_id, incarnation, now, t_ker):
    """Check the root, incarnation, nonempty sketch, and timestamp bounds."""
    if not isinstance(kernel, dict) or incarnation is None:
        return False
    if kernel.get('root_id') != root_id or kernel.get('incarnation') != incarnation:
        return False
    try:
        age = float(now) - float(kernel['generation_time'])
        k, M = int(kernel['k']), int(kernel['M'])
        rho, sigma = kernel['rho_samples'], kernel['sigma_samples']
        anchor = kernel['anchor_rc']
        return (
            math.isfinite(age) and 0 <= age <= float(t_ker)
            and k >= 3 and k % 2 == 1 and M > 0
            and len(rho) == M and len(sigma) == M
            and len(anchor) == 2
            and all(math.isfinite(float(x)) for x in rho)
            and all(math.isfinite(float(x)) for x in sigma)
        )
    except (ValueError, TypeError, KeyError, OverflowError):
        return False


def mmzca_affinity_from_kernel(veh_id, root_id, veh_table, micro_zones,
                               kernel, incarnation, now, t_ker):
    """Return (A_tilde, usable) for the versioned diagonal sketch.

    Per the requested simulation convention, an unusable kernel has numeric
    A_tilde=0. The boolean MUST be retained when implementing the paper's
    mixed-freshness parent-selection/fallback rule.
    """
    A_hat, usable = mmzca_raw_affinity_from_kernel(
        veh_id, root_id, veh_table, micro_zones,
        kernel, incarnation, now, t_ker
    )
    if not usable:
        return 0.0, False
    try:
        r_k = (int(kernel['k']) - 1) // 2
        return min(1.0, max(0.0, A_hat / r_k)), True
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return 0.0, False


def mmzca_raw_affinity_from_kernel(veh_id, root_id, veh_table, micro_zones,
                                   kernel, incarnation, now, t_ker):
    """Return (A_hat, usable), where A_hat is in MICRO-zone grid units."""
    if not mmzca_kernel_is_usable(kernel, root_id, incarnation, now, t_ker):
        return 0.0, False
    record = veh_table.values(veh_id)
    if record is None:
        return 0.0, False
    current_rc = _mmzca_micro_rc(record.get('micro_zone'), micro_zones)
    if current_rc is None:
        return 0.0, False
    try:
        root_row, root_col = kernel['anchor_rc']  # NEVER the root's new position
        u, v = current_rc[0] - root_row, current_rc[1] - root_col
        rho_i, sigma_i = u + v, u - v
        M = int(kernel['M'])
        rho_sum = sum(abs(rho_i - x) for x in kernel['rho_samples'])
        sigma_sum = sum(abs(sigma_i - x) for x in kernel['sigma_samples'])
        A_hat = (rho_sum + sigma_sum) / (2.0 * M)
        return (A_hat, True) if math.isfinite(A_hat) else (0.0, False)
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return 0.0, False


def det_normalized_micro_affinity(veh_id, parent_id, veh_table, bus_table,
                                  micro_zones, current_time, t_ker):
    """Return A_tilde(i|root(j)), with user's zero rules for CH/stale.

    A root-CH candidate j -> 0.0. Missing/expired/invalid kernel -> 0.0.
    Otherwise use the latest K_h that candidate parent j can advertise.
    """
    if _mmzca_is_root(parent_id, veh_table, bus_table):
        return 0.0
    parent, _ = _mmzca_node(parent_id, veh_table, bus_table)
    if parent is None:
        return 0.0
    root_id = _mmzca_root_id(parent_id, parent, veh_table, bus_table)
    score, _ = mmzca_affinity_from_kernel(
        veh_id, root_id, veh_table, micro_zones,
        parent.get('kernel_descriptor'), parent.get('kernel_root_incarnation'),
        current_time, t_ker
    )
    return score


def det_micro_affinity(veh_id, parent_id, veh_table, bus_table,
                       micro_zones, current_time, t_ker):
    """Return unnormalized A_hat(i|h) with the same CH/stale -> 0 rule."""
    if _mmzca_is_root(parent_id, veh_table, bus_table):
        return 0.0
    parent, _ = _mmzca_node(parent_id, veh_table, bus_table)
    if parent is None:
        return 0.0
    root_id = _mmzca_root_id(parent_id, parent, veh_table, bus_table)
    score, _ = mmzca_raw_affinity_from_kernel(
        veh_id, root_id, veh_table, micro_zones,
        parent.get('kernel_descriptor'), parent.get('kernel_root_incarnation'),
        current_time, t_ker
    )
    return score


def mmzca_affinity_status(veh_id, parent_id, veh_table, bus_table,
                          micro_zones, current_time, t_ker):
    """Companion (value, status) for selection: valid/root_ch/unavailable.

    Even though root CH and stale cases have A_tilde=0 by convention, they
    should not be confused with a measured perfect spatial match.
    """
    if _mmzca_is_root(parent_id, veh_table, bus_table):
        return 0.0, 'root_ch'
    parent, _ = _mmzca_node(parent_id, veh_table, bus_table)
    if parent is None:
        return 0.0, 'unavailable'
    root_id = _mmzca_root_id(parent_id, parent, veh_table, bus_table)
    score, valid = mmzca_affinity_from_kernel(
        veh_id, root_id, veh_table, micro_zones,
        parent.get('kernel_descriptor'), parent.get('kernel_root_incarnation'),
        current_time, t_ker
    )
    return score, 'valid' if valid else 'unavailable'


def mmzca_latest_root_kernel(root_id, parent_ids, veh_table, bus_table,
                             now, t_ker):
    """SA's latest *one-hop-advertised* valid descriptor for root h.

    Compare multiple parents of the SAME root against the same snapshot when
    possible, as required by the same-root affinity property. No remote CH
    state or global cluster-membership list is consulted.
    """
    selected = None
    for parent_id in parent_ids:
        if _mmzca_is_root(parent_id, veh_table, bus_table):
            continue
        parent, _ = _mmzca_node(parent_id, veh_table, bus_table)
        if parent is None or _mmzca_root_id(parent_id, parent, veh_table, bus_table) != root_id:
            continue
        descriptor = parent.get('kernel_descriptor')
        if not mmzca_kernel_is_usable(
                descriptor, root_id, parent.get('kernel_root_incarnation'), now, t_ker):
            continue
        if selected is None or (
            descriptor['incarnation'], descriptor['snapshot_version'], descriptor['generation_time']
        ) > (
            selected['incarnation'], selected['snapshot_version'], selected['generation_time']
        ):
            selected = descriptor
    return selected


def _mmzca_init_kernel_fields(record):
    """Add kernel state to an existing veh_table/bus_table dictionary in place."""
    record.setdefault('kernel_root_incarnation', None)
    record.setdefault('kernel_local_root_id', None)
    record.setdefault('kernel_descriptor', None)
    record.setdefault('kernel_subtree_hist', {})
    record.setdefault('kernel_subtree_version', 0)
    record.setdefault('kernel_child_reports', {})
    record.setdefault('kernel_last_signature', None)
    record.setdefault('kernel_known_children', set())
    record.setdefault('kernel_snapshot_version', 0)
    record.setdefault('kernel_last_snapshot_hist', None)
    record.setdefault('kernel_last_snapshot_anchor', None)


def mmzca_note_ch_promotion(root_id, root_registry):
    """Call on ordinary-vehicle SA->CH promotion (including same-tick reuse).

    This event hook prevents an unobserved CH->SA->CH transition within one
    simulation tick from reusing a previous incarnation identifier.
    """
    state = root_registry.setdefault(root_id, {'epoch': 0, 'active': False})
    state['epoch'] += 1
    state['active'] = True
    return state['epoch']


def _mmzca_valid_affiliations(veh_table, bus_table, root_epochs, h_max,
                               distance_fn):
    """Reference simulator validation of locally committed parent chains.

    The resulting map is used for message delivery and admission to the
    kernel protocol, not as a remote CH list exposed to an SA.
    """
    all_ids = veh_table.ids() | bus_table.ids()
    memo = {}
    visiting = set()

    def validate(node_id):
        if node_id in memo:
            return memo[node_id]
        if node_id in visiting or node_id not in all_ids:
            return None
        visiting.add(node_id)
        node, table = _mmzca_node(node_id, veh_table, bus_table)
        result = None
        if _mmzca_is_root(node_id, veh_table, bus_table):
            if node_id in root_epochs:
                result = (node_id, None, 0, root_epochs[node_id]['epoch'])
        else:
            root_id = _mmzca_root_id(node_id, node, veh_table, bus_table)
            parent_id = _mmzca_parent_id(node)
            depth = node.get('hop_count')
            if root_id and parent_id and parent_id != node_id:
                parent_info = validate(parent_id)
                if parent_info is not None:
                    parent_root, _, parent_depth, epoch = parent_info
                    if (root_id == parent_root and isinstance(depth, int)
                            and depth == parent_depth + 1 and depth <= h_max):
                        parent, parent_table = _mmzca_node(parent_id, veh_table, bus_table)
                        try:
                            d = distance_fn(node_id, table, parent_id, parent_table)
                            feasible = (math.isfinite(d) and d <= min(
                                float(node['trans_range']), float(parent['trans_range'])))
                        except (TypeError, ValueError, KeyError):
                            feasible = False
                        if feasible:
                            result = (root_id, parent_id, depth, epoch)
        visiting.remove(node_id)
        memo[node_id] = result
        return result

    for node_id in all_ids:
        validate(node_id)
    return {node_id: info for node_id, info in memo.items() if info is not None}


def mmzca_kernel_tick(veh_table, bus_table, micro_zones, config, now,
                      pending_messages, root_registry, stats=None,
                      distance_fn=None):
    """Advance versioned histograms and propagated kernels one beacon tick.

    Mutates ONLY node dictionaries in the existing two hash tables, the
    caller-owned root_registry, and stats. Returns the next pending-message
    list. Every queued message has a one-hop delivery delay of one tick.

    Call after DataTable.update_cluster and stand_alones_cluster. Parent
    selection then reads the previous delivered snapshot from candidate j.
    """
    if distance_fn is None:
        distance_fn = det_dist  # existing function in the user's util.py
    if stats is None:
        stats = {}
    stats.setdefault('kernel_messages_sent', 0)
    stats.setdefault('kernel_control_bytes_est', 0)

    active_ids = veh_table.ids() | bus_table.ids()
    active_roots = {n for n in active_ids if _mmzca_is_root(n, veh_table, bus_table)}
    for root_id, state in root_registry.items():
        if root_id not in active_roots:
            state['active'] = False
    for root_id in active_roots:
        state = root_registry.setdefault(root_id, {'epoch': 0, 'active': False})
        # Epoch advances only when a node newly becomes a CH. The prior
        # inactive state is preserved across ticks and departures.
        if state['epoch'] == 0 or not state['active']:
            state['epoch'] += 1
        state['active'] = True

    for node_id in active_ids:
        node, _ = _mmzca_node(node_id, veh_table, bus_table)
        _mmzca_init_kernel_fields(node)

    info = _mmzca_valid_affiliations(
        veh_table, bus_table, root_registry, int(config.h_max), distance_fn
    )
    children = {node_id: set() for node_id in info}
    for node_id, (_, parent_id, _, _) in info.items():
        if parent_id is not None:
            children[parent_id].add(node_id)

    # Assign only simulation-level incarnation metadata to committed nodes.
    for node_id in active_ids:
        node, _ = _mmzca_node(node_id, veh_table, bus_table)
        expected_epoch = info[node_id][3] if node_id in info else None
        expected_root_id = info[node_id][0] if node_id in info else None
        if (node.get('kernel_root_incarnation') != expected_epoch
                or node.get('kernel_local_root_id') != expected_root_id):
            node['kernel_descriptor'] = None
            node['kernel_child_reports'] = {}
            node['kernel_last_signature'] = None
            node['kernel_known_children'] = set()
        node['kernel_root_incarnation'] = expected_epoch
        node['kernel_local_root_id'] = expected_root_id

    next_messages = []

    message_delay = int(config.kernel_message_delay_ticks)
    if message_delay < 1:
        raise ValueError('kernel_message_delay_ticks must be at least 1')

    def send(kind, sender, receiver, payload):
        msg = {'deliver_at': now + message_delay, 'kind': kind, 'sender': sender,
               'receiver': receiver, 'payload': payload}
        next_messages.append(msg)
        stats['kernel_messages_sent'] += 1
        # Deterministic payload-size proxy; it is not a PHY/MAC byte model.
        stats['kernel_control_bytes_est'] += len(repr(msg).encode('utf-8'))

    # Receive only previously queued, physically feasible one-hop messages.
    for msg in pending_messages:
        if msg['deliver_at'] > now:
            next_messages.append(msg)
            continue
        sender, receiver = msg['sender'], msg['receiver']
        if sender not in info or receiver not in info:
            continue
        payload = msg['payload']
        receiver_rec, _ = _mmzca_node(receiver, veh_table, bus_table)
        if msg['kind'] == 'hist':
            if info[sender][1] != receiver or info[sender][0] != info[receiver][0]:
                continue
            if payload['root_id'] != info[receiver][0] or payload['epoch'] != info[receiver][3]:
                continue
            cache = receiver_rec['kernel_child_reports']
            old = cache.get(sender)
            if old is None or payload['version'] > old['version']:
                cache[sender] = {'version': payload['version'],
                                 'hist': dict(payload['hist'])}
        elif msg['kind'] == 'kernel':
            if info[receiver][1] != sender or info[receiver][0] != info[sender][0]:
                continue
            if payload is None:
                continue
            if (payload.get('root_id') != info[receiver][0]
                    or payload.get('incarnation') != info[receiver][3]):
                continue
            old = receiver_rec.get('kernel_descriptor')
            newer = (old is None or old.get('incarnation') != payload['incarnation']
                     or old.get('snapshot_version', -1) < payload['snapshot_version'])
            if newer:
                receiver_rec['kernel_descriptor'] = payload
                for child in children.get(receiver, ()):
                    send('kernel', receiver, child, payload)

    # Discard reports for no-longer-committed immediate children.
    for node_id in info:
        node, _ = _mmzca_node(node_id, veh_table, bus_table)
        cache = node['kernel_child_reports']
        for old_child in list(cache):
            if old_child not in children[node_id]:
                del cache[old_child]

    # A node includes itself only when it is an eligible CM. Accepted child
    # reports are REPLACED by version, never added repeatedly.
    for node_id, (root_id, parent_id, depth, epoch) in info.items():
        node, _ = _mmzca_node(node_id, veh_table, bus_table)
        own_zone = node.get('micro_zone')
        eligible = 1 <= depth <= int(config.h_max) - 1
        own = {own_zone: 1} if eligible and _mmzca_micro_rc(own_zone, micro_zones) else {}
        hist = dict(own)
        for child_report in node['kernel_child_reports'].values():
            for zone_id, count in child_report['hist'].items():
                hist[zone_id] = hist.get(zone_id, 0) + count
        signature = (root_id, parent_id, depth, epoch, own_zone, tuple(sorted(hist.items())))
        if signature != node['kernel_last_signature']:
            node['kernel_subtree_hist'] = hist
            node['kernel_subtree_version'] += 1
            node['kernel_last_signature'] = signature
            if parent_id is not None:
                send('hist', node_id, parent_id, {
                    'root_id': root_id, 'epoch': epoch,
                    'version': node['kernel_subtree_version'], 'hist': hist
                })

    # Root CH constructs a timestamped snapshot only from received reports.
    for root_id in active_roots:
        if root_id not in info:
            continue
        root, _ = _mmzca_node(root_id, veh_table, bus_table)
        histogram = root['kernel_subtree_hist']
        anchor_zone = root.get('micro_zone')
        refresh_period = max(1, int(config.kernel_refresh_period))
        old = root.get('kernel_descriptor')
        must_refresh = (
            old is None
            or old.get('incarnation') != info[root_id][3]
            or root['kernel_last_snapshot_hist'] != histogram
            or root['kernel_last_snapshot_anchor'] != anchor_zone
            or now - old['generation_time'] >= refresh_period
        )
        if must_refresh:
            root['kernel_snapshot_version'] += 1
            descriptor = mmzca_build_kernel(
                root_id, info[root_id][3], root['kernel_snapshot_version'],
                now, anchor_zone, histogram, micro_zones, config
            )
            root['kernel_descriptor'] = descriptor
            root['kernel_last_snapshot_hist'] = dict(histogram)
            root['kernel_last_snapshot_anchor'] = anchor_zone
            for child in children[root_id]:
                send('kernel', root_id, child, descriptor)

    # Existing descriptors also reach children which joined after generation.
    for node_id in info:
        node, _ = _mmzca_node(node_id, veh_table, bus_table)
        new_children = children[node_id] - node['kernel_known_children']
        descriptor = node.get('kernel_descriptor')
        if descriptor is not None:
            for child in new_children:
                send('kernel', node_id, child, descriptor)
        node['kernel_known_children'] = children[node_id].copy()

    return next_messages


# ===== MMZCA PHYSICAL LINK AND PARENT SELECTION =====

"""Smaller MMZCA parent-selection additions for the EXISTING utils/util.py.

Append AFTER the MMZCA kernel helpers. In particular, this intentionally
reuses their _mmzca_node(), _mmzca_is_root(), _mmzca_root_id(), and
_mmzca_parent_id(), plus your existing det_dist(id, table, id, table).

This file REPLACES the earlier choose_mmzca_parent() addition. Do not append
both versions. It also exposes mmzca_physical_link() for PCH election.
"""

import math
import random


def _mmzca_normalized_weights(weights, count, name):
    values = [float(x) for x in weights]
    if (len(values) != count or not all(math.isfinite(v) and v >= 0 for v in values)
            or sum(values) <= 0):
        raise ValueError(name + ' requires nonnegative finite weights with positive sum')
    total = sum(values)
    return [v / total for v in values]


def _mmzca_heading(record, zones, zone_level, lon_scale):
    """Local motion unit vector using last distinct zone; memoized per state."""
    previous = record.get('prev_' + zone_level + '_zone')
    if previous is None or previous == record.get(zone_level + '_zone'):
        return None
    cache = globals().get('_MMZCA_HEADING_CACHE')
    key = (id(record), id(zones), previous, zone_level,
           record['long'], record['lat'], lon_scale)
    if cache is not None and key in cache:
        return cache[key]
    cell = zones.zone_hash.values(previous)
    if cell is None:
        value = None
    else:
        cx = (float(cell['min_long']) + float(cell['max_long'])) / 2
        cy = (float(cell['min_lat']) + float(cell['max_lat'])) / 2
        dx = (float(record['long']) - cx) * lon_scale
        dy = float(record['lat']) - cy
        length = math.hypot(dx, dy)
        value = ((dx / length, dy / length)
                 if length > 1e-15 and math.isfinite(length) else None)
    if cache is not None:
        cache[key] = value
    return value


def _mmzca_angular_difference(a, b, zones, level, lon_scale):
    av = _mmzca_heading(a, zones, level, lon_scale)
    bv = _mmzca_heading(b, zones, level, lon_scale)
    if av is None or bv is None:
        return math.pi, True
    similarity = max(-1.0, min(1.0, av[0] * bv[0] + av[1] * bv[1]))
    return math.acos(similarity), False


def mmzca_physical_link(i_id, j_id, i_table, j_table,
                        meso_zones, macro_zones, config):
    """Physical L_ij and MZOTD using existing util.det_dist in METERS.

    Return None for a non-feasible reciprocal one-hop physical link.
    This common helper is usable for both PCH SA-SA evaluation and
    the CH/CM parent selector; no duplicated haversine/projection distance.
    """
    i, j = i_table.values(i_id), j_table.values(j_id)
    if i is None or j is None:
        return None
    try:
        distance_m = float(det_dist(i_id, i_table, j_id, j_table))
        radius_m = min(float(i['trans_range']), float(j['trans_range']))
        if (not math.isfinite(distance_m) or not math.isfinite(radius_m)
                or radius_m <= 0 or distance_m > radius_m):
            return None

        v_i, v_j = abs(float(i['speed'])), abs(float(j['speed']))
        speed = abs(v_i - v_j) / max(v_i, v_j, float(config.v0))
        if not math.isfinite(speed) or float(config.v0) <= 0:
            return None

        w = _mmzca_normalized_weights(config.weights_m, 3, 'weights_m')
        omega = _mmzca_normalized_weights(
            (config.omega_meso, config.omega_macro), 2, 'MZOTD weights'
        )
        latitude = (float(config.area['min_lat']) + float(config.area['max_lat'])) / 2
        lon_scale = math.cos(math.radians(latitude))
        theta_m, missing_m = _mmzca_angular_difference(
            i, j, meso_zones, 'meso', lon_scale
        )
        theta_M, missing_M = _mmzca_angular_difference(
            i, j, macro_zones, 'macro', lon_scale
        )
        mzotd = omega[0] * theta_m + omega[1] * theta_M
        L = w[0] * mzotd / math.pi + w[1] * speed + w[2] * distance_m / radius_m
        return {
            'L': L, 'distance_m': distance_m, 'MZOTD_rad': mzotd,
            'meso_angle_missing': missing_m, 'macro_angle_missing': missing_M,
            'speed_term': speed,
        } if math.isfinite(L) else None
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None


def _mmzca_candidate_chain(j_id, joining_id, veh_table, bus_table, h_max):
    """Validate parent j's committed root path and reciprocal physical links.

    Returns (root_id, depth_j) or None; excludes paths through joining_id.
    """
    j, _ = _mmzca_node(j_id, veh_table, bus_table)
    if j is None or j.get('root_connection_valid') is False:
        return None
    root_id = _mmzca_root_id(j_id, j, veh_table, bus_table)
    root, _ = _mmzca_node(root_id, veh_table, bus_table)
    if root is None or not _mmzca_is_root(root_id, veh_table, bus_table):
        return None
    root_epoch = root.get('kernel_root_incarnation')
    if _mmzca_is_root(j_id, veh_table, bus_table):
        return j_id, 0

    try:
        depth_j = int(j['hop_count'])
        if not 1 <= depth_j <= h_max - 1:
            return None
    except (ValueError, TypeError, KeyError):
        return None

    seen = {joining_id}
    node_id, expected_depth = j_id, depth_j
    while node_id not in seen:
        seen.add(node_id)
        node, node_table = _mmzca_node(node_id, veh_table, bus_table)
        if node is None or node.get('root_connection_valid') is False:
            return None
        if _mmzca_root_id(node_id, node, veh_table, bus_table) != root_id:
            return None
        node_epoch = node.get('kernel_root_incarnation')
        if (root_epoch is not None and node_epoch is not None
                and root_epoch != node_epoch):
            return None
        if _mmzca_is_root(node_id, veh_table, bus_table):
            return (root_id, depth_j) if node_id == root_id and expected_depth == 0 else None
        if node.get('hop_count') != expected_depth or expected_depth <= 0:
            return None

        parent_id = _mmzca_parent_id(node)
        parent, parent_table = _mmzca_node(parent_id, veh_table, bus_table)
        if parent is None or parent_id in seen:
            return None
        try:
            d = det_dist(node_id, node_table, parent_id, parent_table)
            if d > min(float(node['trans_range']), float(parent['trans_range'])):
                return None
        except (KeyError, ValueError, TypeError):
            return None
        node_id, expected_depth = parent_id, expected_depth - 1
    return None


def choose_mmzca_parent(veh_id, veh_table, bus_table,
                        bus_candidates, ch_candidates, sub_ch_candidates,
                        micro_zones, meso_zones, macro_zones,
                        config, current_time, return_details=False, rng=None,
                        allowed_root_id=None, extra_descendant_hops=0):
    """Unified direct-CH/CM MMZCA selection, without private distance code.

    Requires the MMZCA kernel utilities in the SAME existing util.py.
    Numerical affinity: zero for CH, stale or missing kernel; status retained.
    """
    if rng is None:
        rng = random
    empty = (None, math.inf, {}) if return_details else (None, math.inf)
    i = veh_table.values(veh_id)
    if i is None:
        return empty

    h_max = int(config.h_max)
    if h_max < 1:
        raise ValueError('h_max must be >= 1')
    wG = _mmzca_normalized_weights(config.weights_G, 3, 'weights_G')
    iota, tol, delta_bus = (float(config.iota), float(config.selection_tolerance),
                            float(config.delta_bus))
    if not 0 <= iota <= 1 or tol < 0 or delta_bus < 0:
        raise ValueError('Invalid iota, selection_tolerance or delta_bus')

    feasible = {}
    for j in set(bus_candidates) | set(ch_candidates) | set(sub_ch_candidates):
        if j == veh_id:
            continue
        record, table = _mmzca_node(j, veh_table, bus_table)
        # A full parent may no longer admit NEW children, but an attached
        # member must still be able to score its existing (grandfathered) link.
        # Otherwise there is no comparable incumbent for hysteresis.
        incumbent_parent = _mmzca_parent_id(i)
        if record is None or (record.get('admit_children') is False
                              and j != incumbent_parent):
            continue
        chain = _mmzca_candidate_chain(j, veh_id, veh_table, bus_table, h_max)
        if chain is None:
            continue
        root_id, depth_j = chain
        # For an incumbent CM, preserve its root and fit its entire subtree.
        if allowed_root_id is not None and root_id != allowed_root_id:
            continue
        if depth_j + 1 + int(extra_descendant_hops) > h_max:
            continue
        link = mmzca_physical_link(veh_id, j, veh_table, table,
                                   meso_zones, macro_zones, config)
        if link is None:
            continue

        children = record.get('direct_children')
        capacity = config.max_ch_members if depth_j == 0 else config.max_rn_members
        if children is not None and len(children) >= int(capacity) and veh_id not in children:
            continue
        if depth_j == 0:
            Q_j = 0.0
        else:
            try:
                Q_j = float(record['path_bottleneck'])
                if not math.isfinite(Q_j) or not 0 <= Q_j <= 1:
                    continue
            except (KeyError, ValueError, TypeError):
                continue

        Q = max(Q_j, link['L'])
        depth = depth_j + 1
        G0 = wG[0] * Q + wG[1] * link['L'] + wG[2] * depth / h_max
        feasible[j] = dict(link, root_id=root_id, depth=depth,
                           Q_candidate=Q, G0=G0, is_root_parent=(depth_j == 0),
                           is_bus_rooted=bus_table.values(root_id) is not None)
    if not feasible:
        return empty

    # The most recent eligible sketch is shared among one-hop CMs of one root.
    kernels_by_root = {}
    for j, f in feasible.items():
        kernels_by_root.setdefault(f['root_id'], set()).add(j)
    affinity_by_root = {}
    if iota < 1:
        for root_id, parents in kernels_by_root.items():
            descriptor = mmzca_latest_root_kernel(
                root_id, parents, veh_table, bus_table, current_time, config.kernel_ttl
            )
            affinity_by_root[root_id] = (
                mmzca_affinity_from_kernel(
                    veh_id, root_id, veh_table, micro_zones, descriptor,
                    descriptor['incarnation'], current_time, config.kernel_ttl
                ) if descriptor is not None else (0.0, False)
            )

    for f in feasible.values():
        if f['is_root_parent']:
            A, status = 0.0, 'root_ch'
        elif iota == 1:
            A, status = 0.0, 'not_used'
        else:
            A, known = affinity_by_root[f['root_id']]
            status = 'valid' if known else 'unavailable'
        f['A_tilde'], f['kernel_status'] = A, status
        f['L_aff'] = iota * f['L'] + (1 - iota) * A
        f['Gaff'] = (wG[0] * f['Q_candidate'] + wG[1] * f['L_aff']
                     + wG[2] * f['depth'] / h_max)
        f['Glb'] = (wG[0] * f['Q_candidate'] + wG[1] * iota * f['L']
                    + wG[2] * f['depth'] / h_max)

    incumbent = _mmzca_parent_id(i)

    def best(ids, key):
        cost = min(feasible[j][key] for j in ids)
        tied = [j for j in ids if abs(feasible[j][key] - cost) <= tol]
        minimum_link = min(feasible[j]['L'] for j in tied)
        tied = [j for j in tied if abs(feasible[j]['L'] - minimum_link) <= tol]
        min_depth = min(feasible[j]['depth'] for j in tied)
        tied = [j for j in tied if feasible[j]['depth'] == min_depth]
        return incumbent if incumbent in tied else rng.choice(tied)

    known = {j for j, f in feasible.items() if f['kernel_status'] != 'unavailable'}
    unknown = set(feasible) - known
    if iota == 1 or not known:
        mode, key, allowed = 'physical', 'G0', set(feasible)
    elif not unknown:
        mode, key, allowed = 'affinity', 'Gaff', known
    else:
        leader = best(known, 'Gaff')
        lb = min(feasible[j]['Glb'] for j in unknown)
        if feasible[leader]['Gaff'] < lb - tol:
            mode, key, allowed = 'guarded_affinity', 'Gaff', known
        else:
            mode, key, allowed = 'physical', 'G0', set(feasible)
    chosen = best(allowed, key)

    buses = {j for j in allowed if feasible[j]['is_bus_rooted'] and
             feasible[j][key] <= feasible[chosen][key] + delta_bus + tol}
    if mode == 'guarded_affinity':
        lb = min(feasible[j]['Glb'] for j in unknown)
        buses = {j for j in buses if feasible[j]['Gaff'] < lb - tol}
    if buses:
        chosen = best(buses, key)

    if incumbent in allowed and incumbent != chosen:
        improvement = feasible[incumbent][key] - feasible[chosen][key]
        # An incumbent is retained unless improvement STRICTLY exceeds Delta_P.
        if improvement <= float(config.parent_hysteresis_delta) + tol:
            chosen = incumbent
    score = feasible[chosen][key]
    details = dict(feasible[chosen], selection_mode=mode, all_candidates=feasible)
    return (chosen, score, details) if return_details else (chosen, score)
