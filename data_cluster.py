"""

This Module is coded for extracting data from XML file related to SUMO and putting them to a Hash table.
There are methods in the main DataTable class to initiate and update the vehicles and buses coming to the
understudied area and creating and updating the clusters using recursion.

"""
__author__: str = "Pouya 'Adrian' Firouzmakan"

import numpy as np
import pandas as pd
import networkx as nx
import folium
from folium.plugins import MarkerCluster
import webbrowser
import sys

from graph import Graph
import utils.util as util
import utils.util_pmc as util_pmc
import hash


class DataTable:
    # This class is determined for defining the hash_table, updating data, routing messages,
    # and defining ip addresses by using trace (which is sumo_trace)

    def __init__(self, config, zones):
        """

        :param config: look at options.py and config.py
        :param zones: all the zones of the area
        """

        self.map = None
        self.bus_table = hash.HashTable(config.n_cars * 100)
        self.veh_table = hash.HashTable(config.n_cars * 100)

        self.zone_vehicles = dict(zip(zones.zone_hash.ids(),
                                      [set() for j in range(len(zones.zone_hash.ids()))]
                                      )
                                  )
        self.zone_buses = dict(zip(zones.zone_hash.ids(),
                                   [set() for j in range(len(zones.zone_hash.ids()))]
                                   )
                               )
        self.zone_ch = dict(zip(zones.zone_hash.ids(),
                                [set() for j in range(len(zones.zone_hash.ids()))]
                                )
                            )
        self.zone_stand_alone = dict(zip(zones.zone_hash.ids(),
                                         [set() for j in range(len(zones.zone_hash.ids()))]
                                         )
                                     )
        self.stand_alone = set()
        self.all_chs = set()
        self.left_veh = dict()
        self.left_bus = dict()
        self.time = config.start_time
        self.understudied_area = zones.understudied_area()
        self.init_count = 0  # this counter is just for defining the self.net_graph for the very first time
        self.edge_color = ''
        self.sumo_edges, self.sumo_nodes = util.sumo_net_info(config.sumo_edge, config.sumo_node)
        for veh in config.sumo_trace.documentElement.getElementsByTagName('timestep')[self.time].childNodes[
                   1::2]:
            self.init_count += 1
            zone_id = zones.det_zone(float(veh.getAttribute('y')),  # determine the zone_id of the car (bus | veh)
                                     float(veh.getAttribute('x'))
                                     )
            # the bus_table will be initiated here for the very first time
            if 'bus' in veh.getAttribute('id'):
                self.bus_table.set_item(veh.getAttribute('id'), util.initiate_new_bus(veh, zones, zone_id, config,
                                                                                      self.understudied_area))
                self.bus_table.values(veh.getAttribute('id'))['arrive_time'] = self.time
                # Here the buses will be added to zone_buses
                self.zone_buses[zone_id].add(veh.getAttribute('id'))
                self.zone_ch[zone_id].add(veh.getAttribute('id'))
                self.all_chs.add(veh.getAttribute('id'))

                # the veh_table will be initiated here for the very first time self.understudied_area
            else:
                self.veh_table.set_item(veh.getAttribute('id'), util.initiate_new_veh(veh, zones, zone_id, config,
                                                                                      self.understudied_area))
                self.veh_table.values(veh.getAttribute('id'))['arrive_time'] = self.time
                # Here the vehicles will be added to zone_vehicles
                self.zone_vehicles[zone_id].add(veh.getAttribute('id'))
                self.stand_alone.add(veh.getAttribute('id'))
                self.zone_stand_alone[self.veh_table.values(veh.getAttribute('id'))['zone']].add(veh.getAttribute('id'))

            # create the self.net_graph or add the new vertex
            if self.init_count == 1:
                self.net_graph = Graph(veh.getAttribute('id'), (float(veh.getAttribute('y')),
                                                                float(veh.getAttribute('x'))
                                                                )
                                       )
            else:
                self.net_graph.add_vertex(veh.getAttribute('id'), (float(veh.getAttribute('y')),
                                                                   float(veh.getAttribute('x'))
                                                                   )
                                          )

    def update(self, config, zones):
        """
        this method updates the bus_table and veh_table values for the current interval.
        Attention: The properties related to clusters and ip addresses are going to be updated here
        :return:
        """
        self.time += 1
        bus_ids = set()
        veh_ids = set()
        for veh in config.sumo_trace.documentElement.getElementsByTagName('timestep')[self.time].childNodes[
                   1::2]:
            zone_id = zones.det_zone(float(veh.getAttribute('y')),  # determine the zone_id of the car (bus | veh)
                                     float(veh.getAttribute('x'))
                                     )

            # update the bus_table for the time step
            if 'bus' in veh.getAttribute('id'):
                bus_ids.add(veh.getAttribute('id'))
                self.bus_table, self.zone_buses, self.zone_ch = util.update_bus_table(veh, self.bus_table, zone_id,
                                                                                      self.understudied_area, zones,
                                                                                      config, self.zone_buses,
                                                                                      self.zone_ch, self.time)
                self.all_chs.add(veh.getAttribute('id'))

            else:
                veh_ids.add(veh.getAttribute('id'))
                self.veh_table, self.zone_vehicles, self.zone_ch, self.stand_alone, \
                    self.zone_stand_alone = util.update_veh_table(veh, self.veh_table, zone_id, self.understudied_area,
                                                                  zones, config, self.zone_vehicles, self.zone_ch,
                                                                  self.stand_alone, self.zone_stand_alone, self.time)
                if self.veh_table.values(veh.getAttribute('id'))['cluster_head'] is True:
                    self.all_chs.add(veh.getAttribute('id'))
            # add the vertex to the graph
            try:
                self.net_graph.adj_list[veh.getAttribute('id')]['pos'] = (float(veh.getAttribute('y')),
                                                                          float(veh.getAttribute('x'))
                                                                          )
                if 'bus' in veh.getAttribute('id'):
                    self.net_graph.adj_list[veh.getAttribute('id')]['edges'] = list(self.bus_table. \
                                                                                    values(veh.getAttribute('id'))[
                                                                                        'cluster_members'])
                else:
                    self.net_graph.adj_list[veh.getAttribute('id')]['edges'] = list(self.veh_table. \
                                                                                    values(veh.getAttribute('id'))[
                                                                                        'cluster_members'])

            except KeyError:

                self.net_graph.add_vertex(veh.getAttribute('id'), (float(veh.getAttribute('y')),
                                                                   float(veh.getAttribute('x'))
                                                                   )
                                          )
        # removing the buses, that have left the understudied area, from self.bus_table and self.zone_buses
        for k in (self.bus_table.ids() - bus_ids):
            temp_cluster_members = self.bus_table.values(k)['cluster_members'].copy()
            for m in temp_cluster_members:
                if m in self.bus_table.values(k)['cluster_members']:  # because m is might be removed
                    # from the cluster_members if it was a sum_member and its secondary_ch is already removed
                    # from the cluster:

                    (self.veh_table, self.bus_table, self.net_graph,
                     self.stand_alone, self.zone_stand_alone) = util.remove_member(m, k, self.veh_table,
                                                                                   self.bus_table, config,
                                                                                   self.net_graph, self.stand_alone,
                                                                                   self.zone_stand_alone,
                                                                                   ch_stays=False)
                    # since k is not inside the area anymore, the priority_ch must be None
                    self.veh_table.values(m)['priority_ch'] = None
                    self.veh_table.values(m)['priority_counter'] = config.priority_counter

            self.zone_buses[self.bus_table.values(k)['zone']].remove(k)
            self.zone_ch[self.bus_table.values(k)['zone']].remove(k)
            self.all_chs.remove(k)
            self.bus_table.values(k)['depart_time'] = self.time - 1
            self.left_bus[k] = self.bus_table.values(k)

            self.bus_table.remove(k)
            self.net_graph.remove_vertex(k)

        # removing the vehicles, that have left the understudied area, from self.veh_table and self.zone_vehicles

        for k in (self.veh_table.ids() - veh_ids):
            if self.veh_table.values(k)['cluster_head'] is True:
                temp_cluster_members = self.veh_table.values(k)['cluster_members'].copy()
                for m in temp_cluster_members:
                    # if m in veh_ids:  # this must be veh_ids not self.veh_table.ids()
                    if m in self.veh_table.values(k)['cluster_members']:  # because m is might be removed
                        # from the cluster_members if it was a sum_member and its secondary_ch is already removed
                        # from the cluster
                        (self.veh_table, self.bus_table, self.net_graph,
                         self.stand_alone, self.zone_stand_alone) = util.remove_member(m, k, self.veh_table,
                                                                                       self.bus_table, config,
                                                                                       self.net_graph, self.stand_alone,
                                                                                       self.zone_stand_alone,
                                                                                       ch_stays=False)

                self.zone_ch[self.veh_table.values(k)['zone']].remove(k)
                self.all_chs.remove(k)

            elif self.veh_table.values(k)['primary_ch'] is not None:
                k_ch = self.veh_table.values(k)['primary_ch']
                if self.veh_table.values(k)['secondary_ch'] is not None:
                    secondary_ch = self.veh_table.values(k)['secondary_ch']
                    (self.veh_table, self.bus_table, self.net_graph,
                     self.stand_alone, self.zone_stand_alone) = util.remove_sub_member(k, secondary_ch, k_ch,
                                                                                       self.veh_table,
                                                                                       self.bus_table, config,
                                                                                       self.net_graph,
                                                                                       self.stand_alone,
                                                                                       self.zone_stand_alone,
                                                                                       sub_mem_stays=False)

                else:
                    (self.veh_table, self.bus_table, self.net_graph,
                     self.stand_alone, self.zone_stand_alone) = util.remove_member(k, k_ch, self.veh_table,
                                                                                   self.bus_table, config,
                                                                                   self.net_graph, self.stand_alone,
                                                                                   self.zone_stand_alone,
                                                                                   mem_stays=False)

            elif k in self.stand_alone:
                self.stand_alone.remove(k)
                self.zone_stand_alone[self.veh_table.values(k)['zone']].remove(k)

            self.zone_vehicles[self.veh_table.values(k)['zone']].remove(k)
            self.veh_table.values(k)['depart_time'] = self.time - 1
            self.left_veh[k] = self.veh_table.values(k)

            self.veh_table.remove(k)
            self.net_graph.remove_vertex(k)

    def update_cluster(self, veh_ids, config, zones):

        """
        This method is designed for finding a cluster for veh_id
        :return: cluster heads and connection between them including through the gate_chs
        """
        for veh_id in self.veh_table.ids():
            self.veh_table.values(veh_id)['other_chs'] = set()
            self.veh_table.values(veh_id)['gates'] = dict()
            self.veh_table.values(veh_id)['gate_chs'] = set()
            self.veh_table.values(veh_id)['other_vehs'] = set()

            # determining the buses and cluster_head in neighbor zones
            (bus_candidates, ch_candidates,
             sub_ch_candidates, other_vehs) = util.det_near_ch(veh_id, self.veh_table, self.bus_table,
                                                               self.zone_buses, self.zone_vehicles)
            if ((len(bus_candidates) + len(ch_candidates) + len(sub_ch_candidates) == 0) and
                    (self.veh_table.values(veh_id)['in_area'] is True) and
                    (self.veh_table.values(veh_id)['primary_ch'] is None) and
                    (self.veh_table.values(veh_id)['cluster_head'] is False)):
                if self.veh_table.values(veh_id)['counter'] > 1:
                    self.veh_table.values(veh_id)['counter'] -= 1
                    # here priority_counter is getting modified

                    if self.veh_table.values(veh_id)['priority_ch'] is None:
                        self.veh_table.values(veh_id)['priority_counter'] = config.priority_counter
                    else:
                        self.veh_table.values(veh_id)['priority_counter'] = (
                                self.veh_table.values(veh_id)['priority_counter'] - 1)
                    # here priority_counter is getting reset if it is 0. Hence, the priority_ch would be None

                    if self.veh_table.values(veh_id)['priority_counter'] == 0:
                        self.veh_table.values(veh_id)['priority_counter'] = config.priority_counter
                        self.veh_table.values(veh_id)['priority_ch'] = None
                    self.stand_alone.add(veh_id)
                    self.veh_table.values(veh_id)['other_vehs'] = other_vehs
                    self.zone_stand_alone[self.veh_table.values(veh_id)['zone']].add(veh_id)
                    continue
                else:
                    (self.veh_table, self.all_chs, self.stand_alone,
                     self.zone_stand_alone, self.zone_ch) = util.set_ch(veh_id, self.veh_table, self.all_chs,
                                                                        self.stand_alone, self.zone_stand_alone,
                                                                        self.zone_ch, config)
                    continue

            elif (self.veh_table.values(veh_id)['in_area'] is True) and \
                    (self.veh_table.values(veh_id)['primary_ch'] is None) and \
                    (self.veh_table.values(veh_id)['cluster_head'] is True):

                temp_cluster_members = self.veh_table.values(veh_id)['cluster_members'].copy()
                for m in temp_cluster_members:
                    dist = util.det_dist(veh_id, self.veh_table, m, self.veh_table)
                    if dist > min(self.veh_table.values(veh_id)['trans_range'],
                                  self.veh_table.values(m)['trans_range']):
                        # remove 'm' from veh_id which is a CH  as 'm' is away of veh_id
                        if m in self.veh_table.values(veh_id)['cluster_members']:
                            (self.veh_table, self.bus_table, self.net_graph,
                             self.stand_alone, self.zone_stand_alone) = (
                                util.remove_member(m, veh_id, self.veh_table, self.bus_table, config,
                                                   self.net_graph, self.stand_alone, self.zone_stand_alone))

                # if the veh_id is a ch and does not have any member, after changing its zone, it won't remain as a ch
                # unless get selected by another vehicles or can't find a cluster head after the counter
                if (len(self.veh_table.values(veh_id)['cluster_members']) == 0) and \
                        ((self.veh_table.values(veh_id)['start_ch_zone'] != self.veh_table.values(veh_id)['zone']) and
                         (self.veh_table.values(veh_id)['prev_zone'] != self.veh_table.values(veh_id)['zone'])):

                    (self.veh_table, self.zone_ch, self.all_chs,
                     self.stand_alone, self.zone_stand_alone) = util.set_ch_to_veh(veh_id, self.veh_table, self.zone_ch,
                                                                                   self.all_chs, self.stand_alone,
                                                                                   self.zone_stand_alone)
                    self.update_cluster([veh_id, ], config, zones)

                else:
                    ch_candidates.remove(veh_id)
                    self.veh_table.values(veh_id)['other_chs'].update(self.veh_table.values(veh_id)['other_chs'].
                                                                      union(bus_candidates))
                    self.veh_table.values(veh_id)['other_chs'].update(self.veh_table.values(veh_id)['other_chs'].
                                                                      union(ch_candidates))
                    self.zone_ch[self.veh_table.values(veh_id)['zone']].add(veh_id)
                    self.all_chs.add(veh_id)
                for other_ch in self.veh_table.values(veh_id)['other_chs']:
                    self.net_graph.add_edge(veh_id, other_ch)
                continue
            # checking if the vehicle is understudied-area and still in transmission range of its current primary_ch
            # or is not in its transmission_range anymore
            elif (self.veh_table.values(veh_id)['in_area'] is True) and \
                    (self.veh_table.values(veh_id)['cluster_head'] is False) and \
                    (self.veh_table.values(veh_id)['primary_ch'] is not None):

                ch_id = self.veh_table.values(veh_id)['primary_ch']
                dist_to_primarych = float()
                dist_to_secondarych = float()
                if 'bus' in ch_id:
                    temp_table = self.bus_table
                else:
                    temp_table = self.veh_table
                if self.veh_table.values(veh_id)['secondary_ch'] is None:
                    secondary_ch = None
                    dist_to_primarych = util.det_dist(veh_id, self.veh_table, ch_id, temp_table)
                elif self.veh_table.values(veh_id)['secondary_ch'] is not None:
                    secondary_ch = self.veh_table.values(veh_id)['secondary_ch']
                    dist_to_secondarych = util.det_dist(veh_id, self.veh_table, secondary_ch, self.veh_table)

                if (((self.veh_table.values(veh_id)['secondary_ch'] is None) and
                     (dist_to_primarych <= min(self.veh_table.values(veh_id)['trans_range'],
                                               temp_table.values(ch_id)['trans_range']))) or
                        ((self.veh_table.values(veh_id)['secondary_ch'] is not None) and
                         (dist_to_secondarych <= min(self.veh_table.values(veh_id)['trans_range'],
                                                     self.veh_table.values(secondary_ch)['trans_range'])))):

                    self.veh_table.values(veh_id)['other_chs'].update(self.veh_table.values(veh_id)['other_chs'].
                                                                      union(bus_candidates))
                    self.veh_table.values(veh_id)['other_chs'].update(self.veh_table.values(veh_id)['other_chs'].
                                                                      union(ch_candidates))
                    # assert self.veh_table.values(veh_id)['primary_ch'] in self.veh_table.values(veh_id)['other_chs']
                    if ch_id in self.veh_table.values(veh_id)['other_chs']:
                        self.veh_table.values(veh_id)['other_chs'].remove(ch_id)
                    # the following conditional is for making sure that self.update_cluster called inside
                    # self.stand_alones_cluster would not add 1 to timer of cluster_record
                    if self.veh_table.values(veh_id)['cluster_record'].tail.value['start_time'] + \
                            self.veh_table.values(veh_id)['cluster_record'].tail.value['timer'] - 1 != self.time:
                        self.veh_table.values(veh_id)['cluster_record'].tail.value['timer'] += 1
                    # updating 'gates' and 'gate_chs' considering if the primary_ch is bus or vehicle-ch
                    if 'bus' in ch_id:
                        self.bus_table.values(ch_id)['gates'][veh_id] = \
                            self.veh_table.values(veh_id)['other_chs']
                        self.bus_table.values(ch_id)['gate_chs']. \
                            update(self.bus_table.values(ch_id)['gate_chs'].
                                   union(self.veh_table.values(veh_id)['other_chs']))
                        self.veh_table.values(veh_id)['other_vehs'] = other_vehs
                    else:
                        self.veh_table.values(ch_id)['gates'][veh_id] = \
                            self.veh_table.values(veh_id)['other_chs']
                        self.veh_table.values(ch_id)['gate_chs']. \
                            update(self.veh_table.values(ch_id)['gate_chs'].
                                   union(self.veh_table.values(veh_id)['other_chs']))
                        self.veh_table.values(veh_id)['other_vehs'] = other_vehs
                    self.net_graph.add_edge(ch_id, veh_id)
                    for other_ch in self.veh_table.values(veh_id)['other_chs']:
                        self.net_graph.add_edge(veh_id, other_ch)
                    continue
                # here the 'primary_ch' will be changed to None and recursion is applied
                elif (((self.veh_table.values(veh_id)['secondary_ch'] is None) and
                       (dist_to_primarych > min(self.veh_table.values(veh_id)['trans_range'],
                                                temp_table.values(ch_id)['trans_range']))) or
                      ((self.veh_table.values(veh_id)['secondary_ch'] is not None) and
                       (dist_to_secondarych > min(self.veh_table.values(veh_id)['trans_range'],
                                                  self.veh_table.values(secondary_ch)['trans_range'])))):
                    if self.veh_table.values(veh_id)['secondary_ch'] is None:
                        (self.veh_table, self.bus_table, self.net_graph,
                         self.stand_alone, self.zone_stand_alone) = (util.remove_member(veh_id, ch_id,
                                                                                        self.veh_table, self.bus_table,
                                                                                        config, self.net_graph,
                                                                                        self.stand_alone,
                                                                                        self.zone_stand_alone))
                    else:
                        sec_ch = self.veh_table.values(veh_id)['secondary_ch']
                        ch_id = self.veh_table.values(veh_id)['primary_ch']
                        # print(veh_id in veh_ids)
                        (self.veh_table, self.bus_table, self.net_graph,
                         self.stand_alone, self.zone_stand_alone) = util.remove_sub_member(veh_id, sec_ch, ch_id,
                                                                                           self.veh_table,
                                                                                           self.bus_table, config,
                                                                                           self.net_graph,
                                                                                           self.stand_alone,
                                                                                           self.zone_stand_alone)

                else:
                    continue
                    # now the updates related to sub_cluster_members should be applied
            self.check_general_framework(veh_id)

        temp_stand_alone = self.stand_alone.copy()
        for veh_id in temp_stand_alone:
            self.veh_table.values(veh_id)['other_chs'] = set()
            self.veh_table.values(veh_id)['gates'] = dict()
            self.veh_table.values(veh_id)['gate_chs'] = set()
            self.veh_table.values(veh_id)['other_vehs'] = set()

            # determining the buses and cluster_head in neighbor zones
            (bus_candidates, ch_candidates,
             sub_ch_candidates, other_vehs) = util.det_near_ch(veh_id, self.veh_table, self.bus_table,
                                                               self.zone_buses, self.zone_vehicles)

            if (len(bus_candidates) + len(ch_candidates) + len(sub_ch_candidates)) != 0:

                self.pmc(veh_id, config, bus_candidates,
                         ch_candidates, sub_ch_candidates, other_vehs)

            self.check_general_framework(veh_id)

        # finding buses' other_chs
        for bus in self.bus_table.ids():
            self.bus_table.values(bus)['other_chs'] = set()
        for bus in self.bus_table.ids():
            nearby_chs = util.det_buses_other_ch(bus, self.veh_table, self.bus_table,
                                                 self.zone_buses, self.zone_ch)
            self.bus_table.values(bus)['other_chs'].update(self.bus_table.values(bus)['other_chs'].union(nearby_chs))
            for node in self.bus_table.values(bus)['other_chs']:
                self.net_graph.add_edge(bus, node)

        # Here the other_vehs must be updated again. Otherwise, the graph would face with some conflicts
        for veh_id in veh_ids:
            if self.veh_table.values(veh_id)['primary_ch'] is not None:
                ch = self.veh_table.values(veh_id)['primary_ch']
                if 'bus' in ch:
                    table = self.bus_table
                else:
                    table = self.veh_table
                self.veh_table.values(veh_id)['other_vehs'] = (self.veh_table.values(veh_id)['other_vehs'] -
                                                               table.values(ch)['cluster_members'])
                for other_veh in self.veh_table.values(veh_id)['other_vehs']:
                    self.net_graph.add_edge(other_veh, veh_id)

    def pmc(self, veh_id, config, bus_candidates,
                  ch_candidates, sub_ch_candidates, other_vehs):

        all_vehs = set()
        all_vehs = all_vehs.union(bus_candidates, ch_candidates, sub_ch_candidates)
        ch, ef = util_pmc.choose_ch(veh_id, self.veh_table, self.bus_table, all_vehs,
                                    self.zone_buses, self.zone_vehicles, config.weights_pcm)

        if (('veh' in ch) and (self.veh_table.values(ch)['cluster_head'] is True)) or ('bus' in ch):
            (self.bus_table, self.veh_table,
             self.stand_alone, self.zone_stand_alone,
             self.net_graph) = util.add_member(ch, self.bus_table, veh_id, self.veh_table,
                                               config, ef, self.time, bus_candidates,
                                               ch_candidates, self.stand_alone,
                                               self.zone_stand_alone, self.net_graph,
                                               other_vehs)

        if ('veh' in ch) and (self.veh_table.values(ch)['cluster_head'] is False):
            sub_ch = ch
            veh_ch = self.veh_table.values(sub_ch)['primary_ch']

            (self.bus_table, self.veh_table,
             self.stand_alone, self.zone_stand_alone,
             self.net_graph) = util.add_sub_member(veh_ch, self.bus_table, veh_id, sub_ch, self.veh_table,
                                                   config, ef, self.time, bus_candidates,
                                                   ch_candidates, self.stand_alone,
                                                   self.zone_stand_alone, self.net_graph,
                                                   other_vehs)
        self.check_general_framework(veh_id)

    def stand_alones_cluster(self, configs, zones):
        near_sa = dict()
        speed_rel = dict()
        data = dict()
        data['veh_id'] = list()
        data['n_near_sa'] = list()
        data['speed_rel'] = list()
        for veh_id in self.stand_alone:
            # self.stand_alone_test(veh_id)
            near_sa[veh_id], speed_rel[veh_id] = util.det_near_sa(veh_id, self.veh_table,
                                               self.stand_alone, self.zone_stand_alone
                                               )
            data['veh_id'].append(veh_id)
            data['speed_rel'].append(speed_rel[veh_id])
            data['n_near_sa'].append(len(near_sa[veh_id]))

        df = pd.dataframe(data=data)
        n_to_be_ch = int(np.ceil(df.shape[0]/5))
        df = df.sort_values(['n_near_sa', 'speed_rel'], ascending=[False, True])
        new_chs = df.head(n_to_be_ch)['veh_id']

        for id in new_chs:
            util.set_ch(id, self.veh_table, self.all_chs, self.stand_alone,
           self.zone_stand_alone, self.zone_ch, configs, its_sa_clustering=True)

        
        # Determining the updating self.veh_tale and self.net_graph
        for k in near_sa.keys():
            self.veh_table, self.net_graph = util.update_sa_net_graph(self.veh_table, k, near_sa, self.net_graph)

        self.update_cluster(self.veh_table.ids(), configs, zones)

    def eval_cluster(self, configs):
        total_clusters = 0
        n_sav_ch = 0  # number of vehicles that are allways ch or stand-alone (never experiences being a cm)
        for i in self.veh_table.ids():
            if self.veh_table.values(i)['depart_time'] is None:
                self.veh_table.values(i)['depart_time'] = configs.start_time + configs.iter
            in_area_time = self.veh_table.values(i)["depart_time"] - self.veh_table.values(i)["arrive_time"] + 1
            total_length = self.veh_table.values(i)['cluster_record'].length
            if (total_length == 1) and (self.veh_table.values(i)['cluster_record'].head.value['timer'] is None):
                n_sav_ch += 1
                continue
            if in_area_time == 0:
                in_area_time += 1

            one_veh = 0
            temp = self.veh_table.values(i)['cluster_record'].head
            summing = 0
            while temp:
                if temp.value['timer'] is not None:
                    summing += temp.value['timer']  # temp.length acs as penalty
                temp = temp.next
            one_veh += np.divide(summing, total_length*in_area_time)
            total_clusters += one_veh

        for i in self.left_veh.keys():
            total_length = self.left_veh[i]['cluster_record'].length
            if (total_length == 1) and (self.left_veh[i]['cluster_record'].head.key is None):
                n_sav_ch += 1
                continue
            one_veh = 0
            temp = self.left_veh[i]['cluster_record'].head
            summing = 0
            in_area_time = self.left_veh[i]['depart_time'] - self.left_veh[i]['arrive_time']
            while temp:
                if temp.value['timer'] is not None:
                    summing += np.divide(temp.value['timer'], (total_length*in_area_time))  # temp.length acs as penalty
                temp = temp.next
            one_veh += np.divide(summing, total_length * in_area_time)
            total_clusters += one_veh
        return np.divide(total_clusters, len(self.veh_table.ids()) + len(self.left_veh) - n_sav_ch)

    def show_graph(self, configs):
        """
        this function will illustrate the self.net_graph
        :return: Graph
        """
        G = nx.Graph()
        # Add nodes and edges with coordinates to the networkx graph
        for vertex, data in self.net_graph.adj_list.items():
            G.add_node(vertex, pos=data['pos'])
            for edge in list(set(data['edges'])):
                G.add_edge(vertex, edge)

        # Extract positions from node attributes
        pos = nx.get_node_attributes(G, 'pos')

        # Create a folium map centered around the first node
        self.map = folium.Map(location=configs.center_loc, zoom_start=configs.map_zoom, tiles='cartodbpositron',
                              attr='Google', name='Google Maps', prefer_canvas=True)

        # Create a MarkerCluster group for the networkx graph nodes
        marker_cluster = MarkerCluster(name='VANET')

        # Add nodes to the MarkerCluster group
        for node, node_pos in pos.items():
            if 'bus' in node:
                if 'rsu' in node:
                    marker = folium.CircleMarker(location=node_pos, radius=10, color='darkpurple', fill=True,
                                                 fill_color='red')
                else:
                    marker = folium.CircleMarker(location=node_pos, radius=10, color='red', fill=True,
                                                 fill_color='red')
            else:
                if self.veh_table.values(node)['cluster_head'] is True:
                    marker = folium.CircleMarker(location=node_pos, radius=10, color='red', fill=True,
                                                 fill_color='red')
                else:
                    marker = folium.CircleMarker(location=node_pos, radius=5, color='lightblue', fill=True,
                                                 fill_color='lightblue')
            marker.add_to(marker_cluster)

        # Add the MarkerCluster group to the map
        marker_cluster.add_to(self.map)

        # Create a feature group for the networkx graph edges
        edge_group = folium.FeatureGroup(name='Graph Edges')

        # Add edges to the feature group
        for edge in G.edges():
            start_pos = pos[edge[0]]
            end_pos = pos[edge[1]]
            locations = [start_pos, end_pos]
            # determine the edge colors
            if ('bus' in edge[0]) and ('bus' in edge[1]):
                self.edge_color = 'pink'
            elif ('veh' in edge[0]) and ('bus' in edge[1]):
                if self.veh_table.values(edge[0])['cluster_head'] is True:
                    self.edge_color = 'pink'
                else:
                    if self.veh_table.values(edge[0])['primary_ch'] == edge[1]:
                        self.edge_color = 'green'
                    else:
                        self.edge_color = 'gray'
            elif ('bus' in edge[0]) and ('veh' in edge[1]):
                if self.veh_table.values(edge[1])['cluster_head'] is True:
                    self.edge_color = 'pink'
                else:
                    if self.veh_table.values(edge[1])['primary_ch'] == edge[0]:
                        self.edge_color = 'green'
                    else:
                        self.edge_color = 'gray'
            elif ('veh' in edge[0]) and ('veh' in edge[1]):
                if self.veh_table.values(edge[0])['cluster_head'] is True:
                    if self.veh_table.values(edge[1])['cluster_head'] is True:
                        self.edge_color = 'pink'
                    elif (self.veh_table.values(edge[1])['cluster_head'] is False) and \
                            (self.veh_table.values(edge[1])['primary_ch'] == edge[0]):
                        self.edge_color = 'green'
                    else:
                        self.edge_color = 'gray'
                else:
                    if self.veh_table.values(edge[1])['cluster_head'] is True:
                        if self.veh_table.values(edge[0])['primary_ch'] == edge[1]:
                            self.edge_color = 'green'
                        else:
                            self.edge_color = 'gray'
                    else:
                        self.edge_color = 'lightblue'

            folium.PolyLine(locations=locations, color=self.edge_color).add_to(edge_group)

        # Create a feature group for the networkx graph nodes
        node_group = folium.FeatureGroup(name='Graph Nodes')

        # Add nodes to the feature group
        for node, node_pos in pos.items():
            folium.Marker(location=node_pos,
                          icon=folium.DivIcon(html=f'<div style="font-size: 10pt; color: blue;">{node}</div>')).add_to(
                node_group)

        # Add the feature group to the map
        node_group.add_to(self.map)
        # Add the edge group to the map
        edge_group.add_to(self.map)

        # Add the map layer control
        folium.LayerControl().add_to(self.map)

        # Save the map as an HTML file
        self.map.save("graph_map.html")

        # Open the HTML file in a web browser
        webbrowser.open("graph_map.html")

        # save the map as image

    def save_map_img(self, zoom, name):
        util.save_img(self.map, zoom, name)

    def print_table(self):
        self.bus_table.print_hash_table()
        self.veh_table.print_hash_table()

    def check_general_framework(self, veh_id):
        """
        this test is to check if the veh_table is a cluster_head and inside another class at a same time
        :param veh_id:
        :return:
        """
        try:
            assert (
                    ((self.veh_table.values(veh_id)['cluster_head'] is True) and
                     (self.veh_table.values(veh_id)['primary_ch'] is None)) or
                    ((self.veh_table.values(veh_id)['cluster_head'] is False) and
                     (self.veh_table.values(veh_id)['primary_ch'] is not None)) or
                    ((self.veh_table.values(veh_id)['cluster_head'] is False) and
                     (self.veh_table.values(veh_id)['primary_ch'] is None) and
                     (veh_id in self.stand_alone))
            )
        except AssertionError:
            print(f'the error happens for {veh_id} at {self.time} \n'
                  f'this test is to check if the veh_table is a cluster_head and inside another class at a same time \n'
                  f'{self.veh_table.values(veh_id)}')

            sys.exit(1)

    def stand_alone_test(self, veh_id):
        """
        this test is to check if a stand_alone vehicle is not in a cluster or is a cluster_head
        :param veh_id:
        :return:
        """
        try:
            assert ((self.veh_table.values(veh_id)['cluster_head'] is False) and
                    ((veh_id in self.stand_alone) and
                     (self.veh_table.values(veh_id)['primary_ch'] is None)))
        except AssertionError:
            print(f'the error happens for {veh_id} at {self.time} \n '
                  f'this test is to check if a stand_alone vehicle is not in a cluster or is a cluster_head \n'
                  f'{self.veh_table.values(veh_id)}')
            sys.exit(1)
