"""

This Module is coded for extracting data from XML file related to SUMO and putting them to a Hash table.
There are methods in the main DataTable class to initiate and update the vehicles and buses coming to the
understudied area and creating and updating the clusters using recursion.

"""
__author__: str = "Pouya 'Adrian' Firouzmakan"

import numpy as np
import networkx as nx
import folium
from folium.plugins import MarkerCluster
import webbrowser
import sys

from graph import Graph
import utils.util as util
import utils.util_helper as uhelp
import utils.util_graph as util_graph
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
        self.micro_zones = zones['micro']
        self.meso_zones = zones['meso']
        self.macro_zones = zones['macro']

        self.zone_vehicles = dict(zip(self.macro_zones.zone_hash.ids(),
                                      [set() for j in range(len(self.macro_zones.zone_hash.ids()))]
                                      )
                                  )
        self.zone_buses = dict(zip(self.macro_zones.zone_hash.ids(),
                                   [set() for j in range(len(self.macro_zones.zone_hash.ids()))]
                                   )
                               )
        self.zone_ch = dict(zip(self.macro_zones.zone_hash.ids(),
                                [set() for j in range(len(self.macro_zones.zone_hash.ids()))]
                                )
                            )
        self.zone_stand_alone = dict(zip(self.macro_zones.zone_hash.ids(),
                                         [set() for j in range(len(self.macro_zones.zone_hash.ids()))]
                                         )
                                     )
        self.stand_alone = set()
        self.all_chs = set()
        self.left_veh = dict()
        self.left_bus = dict()
        self.time = config.start_time
        self.understudied_area = self.macro_zones.understudied_area()
        self.init_count = 0  # this counter is just for defining the self.net_graph for the very first time
        self.edge_color = ''
        self.sumo_edges, self.sumo_nodes = util.sumo_net_info(config.sumo_edge, config.sumo_node)
        self.ch_net = None
        for veh in config.sumo_trace.documentElement.getElementsByTagName('timestep')[self.time].childNodes[
                   1::2]:
            self.init_count += 1
            micro_zone_id = self.micro_zones.det_zone(float(veh.getAttribute('y')),
                                                      # determine the micro_zone_id of the car (bus | veh)
                                                      float(veh.getAttribute('x'))
                                                      )
            meso_zone_id = self.meso_zones.det_zone(float(veh.getAttribute('y')),
                                                      # determine the meso_zone_id of the car (bus | veh)
                                                      float(veh.getAttribute('x'))
                                                      )
            macro_zone_id = self.macro_zones.det_zone(float(veh.getAttribute('y')),  # determine the macro_zone_id of the car (bus | veh)
                                     float(veh.getAttribute('x'))
                                     )
            # the bus_table will be initiated here for the very first time
            veh_id = veh.getAttribute('id')
            if 'bus' in veh_id:
                self.bus_table.set_item(veh_id, util.initiate_new_bus(veh, self.macro_zones, micro_zone_id, meso_zone_id, macro_zone_id, config,
                                                                                      self.understudied_area))
                self.bus_table.values(veh_id)['arrive_time'] = self.time
                # Here the buses will be added to zone_buses
                self.zone_buses[macro_zone_id].add(veh_id)
                self.zone_ch[macro_zone_id].add(veh_id)
                self.all_chs.add(veh_id)

                # the veh_table will be initiated here for the very first time self.understudied_area
            else:
                self.veh_table.set_item(veh_id, util.initiate_new_veh(veh, self.macro_zones, micro_zone_id, meso_zone_id, macro_zone_id, config,
                                                                                      self.understudied_area))
                self.veh_table.values(veh_id)['arrive_time'] = self.time
                # Here the vehicles will be added to zone_vehicles
                self.zone_vehicles[macro_zone_id].add(veh_id)
                self.stand_alone.add(veh_id)
                self.zone_stand_alone[self.veh_table.values(veh_id)['macro_zone']].add(veh_id)

            # create the self.net_graph or add the new vertex
            if self.init_count == 1:
                self.net_graph = nx.Graph()
                self.net_graph.add_node(veh_id, pos=(float(veh.getAttribute('y')),
                                                                float(veh.getAttribute('x'))
                                                                     )
                                        )
            else:
                self.net_graph.add_node(veh_id, pos=(float(veh.getAttribute('y')),
                                                                   float(veh.getAttribute('x'))
                                                                   )
                                        )

    def update(self, config):
        """
        Update bus_table and veh_table for the next timestep.

        This version is compatible with the generalized MMZCA state:
          - micro / meso / macro zones
          - root_ch / parent_node / hop_count
          - legacy primary_ch / secondary_ch compatibility
        """
        self.time += 1
        bus_ids = set()
        veh_ids = set()
        self.net_graph.remove_edges_from(self.net_graph.edges())

        # -------- update all active nodes from SUMO --------
        for veh in config.sumo_trace.documentElement.getElementsByTagName('timestep')[self.time].childNodes[1::2]:
            micro_zone_id = self.micro_zones.det_zone(
                float(veh.getAttribute('y')),
                float(veh.getAttribute('x'))
            )
            meso_zone_id = self.meso_zones.det_zone(
                float(veh.getAttribute('y')),
                float(veh.getAttribute('x'))
            )
            macro_zone_id = self.macro_zones.det_zone(
                float(veh.getAttribute('y')),
                float(veh.getAttribute('x'))
            )

            veh_id = veh.getAttribute('id')

            if 'bus' in veh_id:
                bus_ids.add(veh_id)

                self.bus_table, self.zone_buses, self.zone_ch = util.update_bus_table(
                    veh, self.bus_table,
                    micro_zone_id, meso_zone_id, macro_zone_id,
                    self.understudied_area, self.macro_zones,
                    config, self.zone_buses, self.zone_ch, self.time
                )

                # generalized state sync for buses
                self.bus_table.values(veh_id)['root_ch'] = veh_id
                self.bus_table.values(veh_id)['parent_node'] = None
                self.bus_table.values(veh_id)['hop_count'] = 0
                self.bus_table.values(veh_id)['current_root_cost'] = 0.0
                self.bus_table.values(veh_id)['current_parent_cost'] = 0.0

                self.all_chs.add(veh_id)

            else:
                veh_ids.add(veh_id)

                self.veh_table, self.zone_vehicles, self.zone_ch, self.stand_alone, self.zone_stand_alone = \
                    util.update_veh_table(
                        veh, self.veh_table,
                        micro_zone_id, meso_zone_id, macro_zone_id,
                        self.understudied_area, self.macro_zones,
                        config, self.zone_vehicles, self.zone_ch,
                        self.stand_alone, self.zone_stand_alone, self.time
                    )

                # generalized state sync for vehicles
                uhelp.sync_node_cluster_fields(veh_id, self.veh_table, self.bus_table)

                if self.veh_table.values(veh_id)['cluster_head'] is True:
                    self.all_chs.add(veh_id)

            # graph node update
            try:
                self.net_graph.nodes[veh_id]['pos'] = (
                    float(veh.getAttribute('y')),
                    float(veh.getAttribute('x'))
                )
            except KeyError:
                self.net_graph.add_node(
                    veh_id,
                    pos=(float(veh.getAttribute('y')), float(veh.getAttribute('x')))
                )

        # -------- remove buses that left the area --------
        temp_left_buses = self.bus_table.ids() - bus_ids
        for k in temp_left_buses:
            temp_cluster_members = self.bus_table.values(k)['cluster_members'].copy()

            for m in temp_cluster_members:
                if m in self.bus_table.values(k)['cluster_members']:
                    self.veh_table, self.bus_table, self.stand_alone, self.zone_stand_alone = \
                        util.remove_member(
                            m, k, self.veh_table, self.bus_table, config,
                            self.stand_alone, self.zone_stand_alone,
                            ch_stays=False
                        )

                    if m in self.veh_table.ids():
                        self.veh_table.values(m)['priority_ch'] = None
                        self.veh_table.values(m)['priority_counter'] = config.priority_counter
                        uhelp.reset_cluster_fields(m, self.veh_table, self.bus_table)

            self.zone_buses[self.bus_table.values(k)['macro_zone']].remove(k)
            self.zone_ch[self.bus_table.values(k)['macro_zone']].remove(k)
            self.all_chs.remove(k)

            self.bus_table.values(k)['depart_time'] = self.time - 1
            self.left_bus[k] = self.bus_table.values(k)

            self.bus_table.remove(k)
            self.net_graph.remove_node(k)

        # -------- remove vehicles that left the area --------
        temp_left_vehs = self.veh_table.ids() - veh_ids
        for k in temp_left_vehs:
            if self.veh_table.values(k)['cluster_head'] is True:
                temp_cluster_members = self.veh_table.values(k)['cluster_members'].copy()

                for m in temp_cluster_members:
                    if m in self.veh_table.values(k)['cluster_members']:
                        self.veh_table, self.bus_table, self.stand_alone, self.zone_stand_alone = \
                            util.remove_member(
                                m, k, self.veh_table, self.bus_table, config,
                                self.stand_alone, self.zone_stand_alone,
                                ch_stays=False
                            )
                        if m in self.veh_table.ids():
                            uhelp.reset_cluster_fields(m, self.veh_table, self.bus_table)

                self.zone_ch[self.veh_table.values(k)['macro_zone']].remove(k)
                self.all_chs.remove(k)

            elif self.veh_table.values(k)['primary_ch'] is not None:
                k_root = self.veh_table.values(k)['primary_ch']
                k_parent = self.veh_table.values(k)['secondary_ch']

                if k_parent is not None:
                    self.veh_table, self.bus_table, self.stand_alone, self.zone_stand_alone = \
                        util.remove_sub_member(
                            k, k_parent, k_root,
                            self.veh_table, self.bus_table, config,
                            self.stand_alone, self.zone_stand_alone,
                            sub_mem_stays=False
                        )
                else:
                    self.veh_table, self.bus_table, self.stand_alone, self.zone_stand_alone = \
                        util.remove_member(
                            k, k_root, self.veh_table, self.bus_table, config,
                            self.stand_alone, self.zone_stand_alone,
                            mem_stays=False
                        )

            elif k in self.stand_alone:
                self.stand_alone.remove(k)
                self.zone_stand_alone[self.veh_table.values(k)['macro_zone']].remove(k)

            self.zone_vehicles[self.veh_table.values(k)['macro_zone']].remove(k)
            self.veh_table.values(k)['depart_time'] = self.time - 1
            self.left_veh[k] = self.veh_table.values(k)

            self.veh_table.remove(k)
            self.net_graph.remove_node(k)

        # -------- final consistency pass for active vehicles --------
        for veh_id in self.veh_table.ids():
            uhelp.sync_node_cluster_fields(veh_id, self.veh_table, self.bus_table)

    def update_cluster(self, veh_ids, config):
        """
        New MMZCA update logic:
          - maintain clustered nodes if feasible
          - recover previous root during priority window
          - unresolved SAs try root-first, parent-second joining
          - do not self-elect CH here immediately; leave that to stand_alones_cluster
        """
        # ---- pass 1: clear per-tick local info and sync fields
        for veh_id in self.veh_table.ids():
            self.veh_table.values(veh_id)['other_chs'] = set()
            self.veh_table.values(veh_id)['gates'] = dict()
            self.veh_table.values(veh_id)['gate_chs'] = set()
            self.veh_table.values(veh_id)['other_vehs'] = set()
            uhelp.sync_node_cluster_fields(veh_id, self.veh_table, self.bus_table)

        # ---- pass 2: maintain CHs and currently attached vehicles
        for veh_id in list(self.veh_table.ids()):
            st = self.veh_table.values(veh_id)

            # ---- CH maintenance
            if st['cluster_head'] is True:
                temp_cluster_members = st['cluster_members'].copy()
                for m in temp_cluster_members:
                    dist = util.det_dist(veh_id, self.veh_table, m, self.veh_table)
                    if dist > min(st['trans_range'], self.veh_table.values(m)['trans_range']):
                        if m in st['cluster_members']:
                            self.veh_table, self.bus_table, self.stand_alone, self.zone_stand_alone = \
                                util.remove_member(
                                    m, veh_id, self.veh_table, self.bus_table, config,
                                    self.stand_alone, self.zone_stand_alone
                                )

                # CH with no members becomes SA after macro-zone change
                if (len(st['cluster_members']) == 0 and
                        (st['start_ch_zone'] != st['macro_zone']) and
                        (st['prev_macro_zone'] != st['macro_zone'])):
                    self.veh_table, self.zone_ch, self.all_chs, self.stand_alone, self.zone_stand_alone = \
                        util.set_ch_to_veh(
                            veh_id, self.veh_table, self.zone_ch,
                            self.all_chs, self.stand_alone, self.zone_stand_alone
                        )
                    # uhelp.reset_cluster_fields(veh_id, self.veh_table, self.bus_table)
                    # uhelp.sync_node_cluster_fields(veh_id, self.veh_table, self.bus_table)
                # else:
                #     self.zone_ch[self.veh_table.values(veh_id)['macro_zone']].add(veh_id)
                    # self.all_chs.add(veh_id)
                continue

            # ---- attached vehicle maintenance
            if st['primary_ch'] is not None:
                parent_id = uhelp.get_parent(veh_id, self.veh_table, self.bus_table)
                root_id = uhelp.get_root(veh_id, self.veh_table, self.bus_table)

                # if parent_id is None or root_id is None:
                #     self.veh_table, self.bus_table, self.stand_alone, self.zone_stand_alone = \
                #         uhelp.detach_from_current_cluster(
                #             veh_id, self.veh_table, self.bus_table, config,
                #             self.stand_alone, self.zone_stand_alone
                #         )
                #     continue

                still_feasible = uhelp.feasible_parent_for_root(
                    veh_id, parent_id, root_id,
                    self.veh_table, self.bus_table,
                    self.micro_zones, self.meso_zones, self.macro_zones,
                    config
                )

                if still_feasible:
                    # refresh timer as before
                    rec = self.veh_table.values(veh_id)['cluster_record'].tail.value
                    if rec['start_time'] is not None and rec['start_time'] + rec['timer'] - 1 != self.time:
                        rec['timer'] += 1
                    continue
                else:
                    if parent_id == root_id:
                        self.veh_table, self.bus_table, self.stand_alone, self.zone_stand_alone = \
                            util.remove_member(
                                veh_id, root_id, self.veh_table, self.bus_table, config,
                                self.stand_alone, self.zone_stand_alone
                            )
                    else:
                        self.veh_table, self.bus_table, self.stand_alone, self.zone_stand_alone = \
                            util.remove_sub_member(
                                veh_id, parent_id, root_id, self.veh_table, self.bus_table, config,
                                self.stand_alone, self.zone_stand_alone
                            )

                    # self.veh_table, self.bus_table, self.stand_alone, self.zone_stand_alone = \
                    #     uhelp.detach_from_current_cluster(
                    #         veh_id, self.veh_table, self.bus_table, config,
                    #         self.stand_alone, self.zone_stand_alone
                    #     )
                    # after detaching, it becomes SA with previous-root priority preserved by legacy fields
                    continue

        # ---- pass 3: unresolved stand-alone joining logic
        temp_stand_alone = list(self.stand_alone.copy())
        for veh_id in temp_stand_alone:
            if veh_id not in self.veh_table.ids():
                continue

            st = self.veh_table.values(veh_id)

            # if st['cluster_head'] is True or st['primary_ch'] is not None:
            #     continue
            # if st['in_area'] is not True:
            #     continue

            st['other_chs'] = set()
            st['gates'] = dict()
            st['gate_chs'] = set()
            st['other_vehs'] = set()

            bus_candidates, ch_candidates, sub_ch_candidates, other_vehs = util.det_near_ch(
                veh_id, self.veh_table, self.bus_table,
                self.zone_buses, self.zone_vehicles
            )

            if len(sub_ch_candidates) == 0:
                self.single_hop(veh_id, config, bus_candidates, ch_candidates, other_vehs)
            else:
                self.multi_hop(veh_id, config, bus_candidates, ch_candidates, sub_ch_candidates, other_vehs)

    def multi_hop(self, veh_id, config, bus_candidates,
                  ch_candidates, sub_ch_candidates, other_vehs):
        if veh_id not in self.veh_table.ids():
            return

        rn_candidates = set()

        for cand in sub_ch_candidates:
            if cand in self.veh_table.ids():
                if (self.veh_table.values(cand)['cluster_head'] is False and
                        self.veh_table.values(cand)['primary_ch'] is not None):
                    rn_candidates.add(cand)

        bus_candidates, ch_candidates, rn_candidates = uhelp.priority_visible_candidates(
            veh_id, bus_candidates, ch_candidates, rn_candidates, self.veh_table
        )

        best_cand, best_cost = uhelp.choose_best_visible_candidate(
            veh_id,
            bus_candidates, ch_candidates, rn_candidates,
            self.veh_table, self.bus_table,
            self.micro_zones, self.meso_zones, self.macro_zones,
            config
        )

        if best_cand is None:
            st = self.veh_table.values(veh_id)

            if st['counter'] > 0:
                st['counter'] -= 1

            # priority window countdown
            if st['priority_ch'] is not None:
                st['priority_counter'] -= 1
                if st['priority_counter'] <= 0:
                    st['priority_ch'] = None
                    st['priority_counter'] = config.priority_counter
            else:
                st['priority_counter'] = config.priority_counter

            self.stand_alone.add(veh_id)
            self.zone_stand_alone[st['macro_zone']].add(veh_id)
            return

        st = self.veh_table.values(veh_id)

        # switching only inside same cluster
        if st['primary_ch'] is not None:
            if not uhelp.intra_cluster_switch_allowed(veh_id, best_cand, self.veh_table, self.bus_table):
                return

        if uhelp.candidate_type(best_cand, self.veh_table, self.bus_table) == 'root':
            root_id = best_cand
            parent_id = best_cand
        else:
            root_id = uhelp.get_root(best_cand, self.veh_table, self.bus_table)
            parent_id = best_cand

        self.bus_table, self.veh_table, self.stand_alone, self.zone_stand_alone = \
            uhelp.attach_to_parent(
                root_id, parent_id, veh_id,
                self.veh_table, self.bus_table, config, self.time,
                bus_candidates, ch_candidates,
                self.stand_alone, self.zone_stand_alone, other_vehs,
                root_cost_value=best_cost if parent_id == root_id else None,
                parent_cost_value=best_cost
            )

        uhelp.sync_node_cluster_fields(veh_id, self.veh_table, self.bus_table)

    def single_hop(self, veh_id, config,
                   bus_candidates, ch_candidates, other_vehs):
        if veh_id not in self.veh_table.ids():
            return

        rn_candidates = set()

        # use priority window
        bus_candidates, ch_candidates, rn_candidates = uhelp.priority_visible_candidates(
            veh_id, bus_candidates, ch_candidates, rn_candidates, self.veh_table
        )

        best_cand, best_cost = uhelp.choose_best_visible_candidate(
            veh_id,
            bus_candidates, ch_candidates, rn_candidates,
            self.veh_table, self.bus_table,
            self.micro_zones, self.meso_zones, self.macro_zones,
            config
        )

        if best_cand is None:
            st = self.veh_table.values(veh_id)

            if st['counter'] > 0:
                st['counter'] -= 1

            # priority window countdown
            if st['priority_ch'] is not None:
                st['priority_counter'] -= 1
                if st['priority_counter'] <= 0:
                    # st['priority_ch'] = None
                    # st['priority_counter'] = config.priority_counter
                    self.veh_table, self.all_chs, self.stand_alone, self.zone_stand_alone, self.zone_ch = \
                        util.set_ch(
                            veh_id, self.veh_table, self.all_chs,
                            self.stand_alone, self.zone_stand_alone,
                            self.zone_ch, config, its_sa_clustering=False
                        )
            # else:
            #     st['priority_counter'] = config.priority_counter

            # self.stand_alone.add(veh_id)
            # self.zone_stand_alone[st['macro_zone']].add(veh_id)
            return

        st = self.veh_table.values(veh_id)

        # switching only inside same cluster
        if st['primary_ch'] is not None:
            if not uhelp.intra_cluster_switch_allowed(veh_id, best_cand, self.veh_table, self.bus_table):
                return

        if uhelp.candidate_type(best_cand, self.veh_table, self.bus_table) == 'root':
            root_id = best_cand
            parent_id = best_cand
        else:
            root_id = uhelp.get_root(best_cand, self.veh_table, self.bus_table)
            parent_id = best_cand

        self.bus_table, self.veh_table, self.stand_alone, self.zone_stand_alone = \
            uhelp.attach_to_parent(
                root_id, parent_id, veh_id,
                self.veh_table, self.bus_table, config, self.time,
                bus_candidates, ch_candidates,
                self.stand_alone, self.zone_stand_alone, other_vehs,
                root_cost_value=best_cost if parent_id == root_id else None,
                parent_cost_value=best_cost
            )

        uhelp.sync_node_cluster_fields(veh_id, self.veh_table, self.bus_table)

    def stand_alones_cluster(self, configs):
        """
        New SA handling:
          1. build nearby unresolved SA sets
          2. choose better PCH/anchor instead of pure neighbor count
          3. remaining SAs try anchor-assisted joining
          4. only unresolved and expired-counter SAs become CH
        """
        near_sa = {}
        pch_choice = {}
        processed = set()

        temp_stand_alone = sorted(list(self.stand_alone), reverse=True)

        # Step 1: nearby unresolved SAs
        for veh_id in temp_stand_alone:
            if veh_id not in self.veh_table.ids():
                continue
            near_sa[veh_id] = util.det_near_sa(
                veh_id, self.veh_table,
                self.stand_alone, self.zone_stand_alone
            )

        # Step 2: better PCH / anchor consensus
        for veh_id in temp_stand_alone:
            if veh_id not in self.veh_table.ids():
                continue
            if len(near_sa.get(veh_id, set())) == 0:
                continue

            best_anchor, best_score = uhelp.best_pch_for_sa(
                veh_id, near_sa,
                self.veh_table,
                self.micro_zones, self.meso_zones,
                configs
            )
            if best_anchor is not None:
                pch_choice[veh_id] = best_anchor

        # Step 3: unresolved SAs use anchor-assisted joining
        for veh_id in temp_stand_alone:
            if veh_id not in self.veh_table.ids():
                continue
            if veh_id in processed:
                continue
            if self.veh_table.values(veh_id)['cluster_head'] is True:
                continue
            if self.veh_table.values(veh_id)['primary_ch'] is not None:
                continue

            bus_candidates, ch_candidates, rn_candidates, other_vehs = util.det_near_ch(
                veh_id, self.veh_table, self.bus_table,
                self.zone_buses, self.zone_vehicles
            )

            # add chosen anchor as an RN candidate if it is attached
            anchor = pch_choice.get(veh_id)
            if anchor is not None and anchor in self.veh_table.ids():
                if (self.veh_table.values(anchor)['cluster_head'] is False and
                        self.veh_table.values(anchor)['primary_ch'] is not None):
                    rn_candidates.add(anchor)

            root_to_parents = uhelp.feasible_roots_and_parents(
                veh_id,
                bus_candidates, ch_candidates, rn_candidates,
                self.veh_table, self.bus_table,
                self.micro_zones, self.meso_zones, self.macro_zones,
                configs
            )

            if len(root_to_parents) > 0:
                best_root, best_root_cost = uhelp.choose_best_root(
                    veh_id, root_to_parents,
                    self.veh_table, self.bus_table,
                    self.meso_zones, self.macro_zones,
                    configs
                )

                if best_root is not None:
                    best_parent, best_parent_cost = uhelp.choose_best_parent_for_root(
                        veh_id, best_root, root_to_parents[best_root],
                        self.veh_table, self.bus_table,
                        self.micro_zones,
                        configs
                    )

                    if best_parent is not None:
                        self.bus_table, self.veh_table, self.stand_alone, self.zone_stand_alone = \
                            uhelp.attach_to_parent(
                                best_root, best_parent, veh_id,
                                self.veh_table, self.bus_table, configs, self.time,
                                bus_candidates, ch_candidates,
                                self.stand_alone, self.zone_stand_alone, other_vehs,
                                root_cost_value=best_root_cost,
                                parent_cost_value=best_parent_cost
                            )
                        uhelp.sync_node_cluster_fields(veh_id, self.veh_table, self.bus_table)
                        processed.add(veh_id)
                        continue

            # Step 4: only unresolved and expired-counter SAs become CH
            if (self.veh_table.values(veh_id)['primary_ch'] is None and
                    self.veh_table.values(veh_id)['cluster_head'] is False and
                    self.veh_table.values(veh_id)['counter'] <= 0):
                self.veh_table, self.all_chs, self.stand_alone, self.zone_stand_alone, self.zone_ch = \
                    util.set_ch(
                        veh_id, self.veh_table, self.all_chs,
                        self.stand_alone, self.zone_stand_alone,
                        self.zone_ch, configs, its_sa_clustering=True
                    )

                self.veh_table.values(veh_id)['root_ch'] = veh_id
                self.veh_table.values(veh_id)['parent_node'] = None
                self.veh_table.values(veh_id)['hop_count'] = 0
                self.veh_table.values(veh_id)['current_root_cost'] = 0.0
                self.veh_table.values(veh_id)['current_parent_cost'] = 0.0

        # Final consistency pass
        for veh_id in self.veh_table.ids():
            uhelp.sync_node_cluster_fields(veh_id, self.veh_table, self.bus_table)

    def update_other_connections(self):
        # finding buses' other_chs
        # Here the other_vehs must be updated again. Otherwise, the graph would face with some conflicts
        self.veh_table, self.bus_table = util.other_connections_update(self.veh_table, self.bus_table,
                                                                       self.zone_ch, self.zone_buses,
                                                                       self.zone_vehicles)

    def form_net_graph(self):
        for veh_id in self.veh_table.ids():
            if self.veh_table.values(veh_id)['cluster_head'] is False:
                self.net_graph = util_graph.veh_add_edges(veh_id, self.veh_table, self.net_graph)
            else:
                self.net_graph = util_graph.ch_add_edges(veh_id, self.veh_table, self.net_graph)

        for bus_id in self.bus_table.ids():
            self.net_graph = util_graph.bus_add_edges(bus_id, self.bus_table, self.net_graph)


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

    def vcsm(self, configs):
        """
        Evaluates VCSM using root-cluster consistency for MMZCA.

            VCSM = (1 / n_vm) * sum_{i in V_m} [ (sum_k t_{i,k}) / (gamma_i * T_i) ]

        where:
          - V_m: vehicles that were attached to a root cluster at least once as non-CH members
          - gamma_i: number of root-cluster residence segments for vehicle i
          - T_i: total time vehicle i remained in the area
          - t_{i,k}: duration of the k-th root-cluster residence segment

        Notes:
          - CH-only or always-SA vehicles are excluded.
          - Parent changes that preserve the same root cluster should not create a new segment.
          - This function assumes cluster_record stores root CH IDs as keys.
        """

        def _veh_vcsm_one(cluster_record, arrive_time, depart_time):
            # Total time in area
            T_i = depart_time - arrive_time
            if T_i <= 0:
                T_i = 1

            total_clustered_time = 0.0
            gamma_i = 0

            temp = cluster_record.head
            while temp:
                if hasattr(temp, "key") and hasattr(temp, "value"):
                    root_id = temp.key
                    timer = temp.value.get('timer', None)
                    is_ch = temp.value.get('is_ch', False)

                    # Only count non-CH membership segments with valid root and timer
                    if (root_id is not None) and (timer is not None) and (is_ch is False):
                        total_clustered_time += timer
                        gamma_i += 1
                temp = temp.next

            # Vehicle was never clustered as member
            if gamma_i == 0:
                return None

            return total_clustered_time / (gamma_i * T_i)

        total_vcsm = 0.0
        n_vm = 0

        # Active vehicles
        for vid in self.veh_table.ids():
            v = self.veh_table.values(vid)

            depart_time = v.get('depart_time', None)
            if depart_time is None:
                depart_time = self.time

            vcsm_i = _veh_vcsm_one(
                cluster_record=v['cluster_record'],
                arrive_time=v['arrive_time'],
                depart_time=depart_time
            )

            if vcsm_i is not None:
                total_vcsm += vcsm_i
                n_vm += 1

        # Vehicles that left
        for vid, v in self.left_veh.items():
            vcsm_i = _veh_vcsm_one(
                cluster_record=v['cluster_record'],
                arrive_time=v['arrive_time'],
                depart_time=v['depart_time']
            )

            if vcsm_i is not None:
                total_vcsm += vcsm_i
                n_vm += 1

        if n_vm == 0:
            return 0.0

        return total_vcsm / n_vm

    def connected_components(self):
        n = 0  # this would return the minimum number of path needed to connect all the clusters
        investigated = set()
        self.ch_net = nx.Graph()
        self.ch_net.add_nodes_from(list(self.all_chs))

        for i in self.all_chs:
            investigated.add(i)
            for j in (self.all_chs - investigated):
                try:
                    nx.shortest_path(self.net_graph, source=i, target=j)
                    self.ch_net.add_edge(i,j)
                except nx.exception.NetworkXNoPath:
                    n += 1

        conn_comp = list(nx.connected_components(self.ch_net))

        return len(conn_comp)

    def show_graph(self, configs):
        """
        this function will illustrate the self.net_graph
        :return: Graph
        """

        # Extract positions from node attributes
        pos = nx.get_node_attributes(self.net_graph, 'pos')

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
        for edge in self.net_graph.edges():
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
                        if (self.veh_table.values(edge[1])['secondary_ch'] == edge[0]) \
                            or (self.veh_table.values(edge[0])['secondary_ch'] == edge[1]):
                            self.edge_color = 'lightgreen'
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