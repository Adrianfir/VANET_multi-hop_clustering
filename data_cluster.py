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

        self.tvct_history = []  # stores TVCT(t) at each tick
        self.avg_tvct_history = []  # stores running average up to each tick
        self.tvct_cumulative = 0.0

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


    def _pmc_clear_local_fields(self, veh_id):
        st = self.veh_table.values(veh_id)
        st['other_chs'] = set()
        st['gates'] = dict()
        st['gate_chs'] = set()
        st['other_vehs'] = set()

    def _pmc_node_state(self, node_id):
        if 'bus' in node_id:
            return self.bus_table.values(node_id)
        return self.veh_table.values(node_id)

    def _pmc_params(self, config):
        alpha = getattr(config, 'pmc_alpha', 1.0/3.0)
        beta = getattr(config, 'pmc_beta', 1.0/3.0)
        gamma = getattr(config, 'pmc_gamma', 1.0/3.0)
        max_member = getattr(config, 'max_member', getattr(config, 'max_rn_members', 10))
        h_max = getattr(config, 'h_max', 2)
        return alpha, beta, gamma, max_member, h_max

    def _pmc_following_degree(self, node_id):
        st = self._pmc_node_state(node_id)
        followers = len(st.get('cluster_members', set())) + len(st.get('sub_cluster_members', set()))
        same_lane = 0
        if 'veh' in node_id and node_id in self.veh_table.ids():
            lane = st['lane']['id']
            neigh = self.macro_zones.neighbor_zones(st['macro_zone'])
            for j in self.veh_table.ids():
                if j == node_id:
                    continue
                sj = self.veh_table.values(j)
                if sj['macro_zone'] not in neigh:
                    continue
                if sj['lane']['id'] != lane:
                    continue
                try:
                    d = util.det_dist(node_id, self.veh_table, j, self.veh_table)
                except Exception:
                    continue
                if d <= min(st['trans_range'], sj['trans_range']):
                    same_lane += 1
        return same_lane + followers

    def _pmc_relative_mobility(self, id1, id2):
        s1 = self._pmc_node_state(id1)
        s2 = self._pmc_node_state(id2)
        speed_term = abs(s1['speed'] - s2['speed']) / max(abs(s1['speed']), abs(s2['speed']), 1e-6)
        angle_diff = abs(s1['angle'] - s2['angle'])
        while angle_diff > 180:
            angle_diff -= 360
        angle_term = abs(angle_diff) / 180.0
        return 0.7 * speed_term + 0.3 * angle_term

    def _pmc_avg_relative_mobility(self, veh_id):
        if veh_id not in self.veh_table.ids():
            return float('inf')
        st = self.veh_table.values(veh_id)
        neigh = []
        for j in self.veh_table.ids():
            if j == veh_id:
                continue
            sj = self.veh_table.values(j)
            if sj['macro_zone'] not in st['neighbor_zones']:
                continue
            try:
                d = util.det_dist(veh_id, self.veh_table, j, self.veh_table)
            except Exception:
                continue
            if d <= min(st['trans_range'], sj['trans_range']):
                neigh.append(j)
        if not neigh:
            return float('inf')
        vals = [self._pmc_relative_mobility(veh_id, j) for j in neigh]
        return float(np.mean(vals))

    def _pmc_etx(self, id1, id2):
        s1 = self._pmc_node_state(id1)
        s2 = self._pmc_node_state(id2)
        table2 = self.bus_table if 'bus' in id2 else self.veh_table
        d = util.det_dist(id1, self.veh_table if 'veh' in id1 else self.bus_table, id2, table2)
        q = max(0.0, 1.0 - d / max(min(s1['trans_range'], s2['trans_range']), 1.0))
        q = max(q*q, 1e-6)
        return 1.0 / q

    def _pmc_llt(self, id1, id2):
        s1 = self._pmc_node_state(id1)
        s2 = self._pmc_node_state(id2)
        dx = (s2['long'] - s1['long']) * 85000.0
        dy = (s2['lat'] - s1['lat']) * 111000.0
        a1 = np.deg2rad(s1['angle'])
        a2 = np.deg2rad(s2['angle'])
        v1x, v1y = s1['speed'] * np.cos(a1), s1['speed'] * np.sin(a1)
        v2x, v2y = s2['speed'] * np.cos(a2), s2['speed'] * np.sin(a2)
        dvx, dvy = v1x - v2x, v1y - v2y
        denom = dvx*dvx + dvy*dvy
        if denom <= 1e-6:
            return 1e6
        r = min(s1['trans_range'], s2['trans_range'])
        cross = dx*dvy - dy*dvx
        inside = r*r*denom - cross*cross
        if inside < 0:
            inside = 0.0
        llt = (np.sqrt(inside) - (dx*dvx + dy*dvy)) / denom
        return max(float(llt), 1e-6)

    def _pmc_priority(self, veh_id, cand_id, config):
        alpha, beta, gamma, _, _ = self._pmc_params(config)
        nf = max(self._pmc_following_degree(cand_id), 1)
        etx = self._pmc_etx(veh_id, cand_id)
        llt = self._pmc_llt(veh_id, cand_id)
        return alpha * (1.0 / nf) + beta * etx + gamma * (1.0 / llt)

    def _pmc_bech(self, x, y):
        fx = self._pmc_following_degree(x)
        fy = self._pmc_following_degree(y)
        rx = self._pmc_avg_relative_mobility(x)
        ry = self._pmc_avg_relative_mobility(y)
        return (fx > fy) or ((fx == fy) and (rx < ry))

    def _pmc_can_accept_parent(self, parent_id, config):
        _, _, _, max_member, h_max = self._pmc_params(config)
        if 'bus' in parent_id:
            return True
        st = self.veh_table.values(parent_id)
        if st['cluster_head']:
            return len(st.get('cluster_members', set())) < max_member
        if st['primary_ch'] is not None:
            return (st.get('hop_count', 1) < h_max and len(st.get('sub_cluster_members', set())) < max_member)
        return False

    def _pmc_root_of(self, node_id):
        if 'bus' in node_id:
            return node_id
        st = self.veh_table.values(node_id)
        if st['cluster_head']:
            return node_id
        return st.get('primary_ch')

    def _pmc_attach(self, veh_id, parent_id, config, bus_candidates, ch_candidates, other_vehs):
        root_id = self._pmc_root_of(parent_id)
        if root_id is None:
            root_id = parent_id
        if parent_id == root_id:
            self.bus_table, self.veh_table, self.stand_alone, self.zone_stand_alone = util.add_member(
                root_id, self.bus_table, veh_id, self.veh_table, config, 0.0, self.time,
                bus_candidates, ch_candidates, self.stand_alone, self.zone_stand_alone, other_vehs
            )
        else:
            self.bus_table, self.veh_table, self.stand_alone, self.zone_stand_alone = util.add_sub_member(
                root_id, self.bus_table, veh_id, parent_id, self.veh_table, config, 0.0, self.time,
                bus_candidates, ch_candidates, self.stand_alone, self.zone_stand_alone, other_vehs
            )
        uhelp.sync_node_cluster_fields(veh_id, self.veh_table, self.bus_table)

    def _pmc_detach(self, veh_id, config):
        st = self.veh_table.values(veh_id)
        root_id = st.get('primary_ch')
        parent_id = st.get('secondary_ch')
        if root_id is None:
            return
        if parent_id is None:
            self.veh_table, self.bus_table, self.stand_alone, self.zone_stand_alone = util.remove_member(
                veh_id, root_id, self.veh_table, self.bus_table, config,
                self.stand_alone, self.zone_stand_alone
            )
        else:
            self.veh_table, self.bus_table, self.stand_alone, self.zone_stand_alone = util.remove_sub_member(
                veh_id, parent_id, root_id, self.veh_table, self.bus_table, config,
                self.stand_alone, self.zone_stand_alone
            )
        uhelp.sync_node_cluster_fields(veh_id, self.veh_table, self.bus_table)

    def _pmc_best_existing_parent(self, veh_id, config):
        bus_candidates, ch_candidates, sub_ch_candidates, other_vehs = util.det_near_ch(
            veh_id, self.veh_table, self.bus_table, self.zone_buses, self.zone_vehicles
        )
        best_parent = None
        best_pri = float('inf')
        # prefer CH/bus when available
        for cand in list(bus_candidates) + list(ch_candidates):
            if self._pmc_can_accept_parent(cand, config):
                pri = 0.0 if 'bus' in cand else self._pmc_priority(veh_id, cand, config)
                if pri < best_pri:
                    best_pri = pri
                    best_parent = cand
        if best_parent is None:
            for cand in sub_ch_candidates:
                if self._pmc_can_accept_parent(cand, config):
                    pri = self._pmc_priority(veh_id, cand, config)
                    if pri < best_pri:
                        best_pri = pri
                        best_parent = cand
        return best_parent, bus_candidates, ch_candidates, sub_ch_candidates, other_vehs

    def _pmc_try_merge_ch(self, veh_id, config):
        if veh_id not in self.veh_table.ids():
            return
        st = self.veh_table.values(veh_id)
        if not st['cluster_head']:
            return
        _, _, _, max_member, h_max = self._pmc_params(config)
        _, near_chs = set(), set()
        bus_candidates, ch_candidates, _, _ = util.det_near_ch(veh_id, self.veh_table, self.bus_table, self.zone_buses, self.zone_vehicles)
        best_target = None
        for ch in ch_candidates:
            if ch == veh_id:
                continue
            if not self._pmc_bech(ch, veh_id):
                continue
            # rough same direction condition
            if abs(self.veh_table.values(ch)['angle'] - st['angle']) > 45 and abs(self.veh_table.values(ch)['angle'] - st['angle']) < 315:
                continue
            best_target = ch
            break
        if best_target is None:
            return
        if len(self.veh_table.values(best_target).get('cluster_members', set())) >= max_member:
            return
        # demote current CH and attach directly to better CH
        self.veh_table, self.zone_ch, self.all_chs, self.stand_alone, self.zone_stand_alone = util.set_ch_to_veh(
            veh_id, self.veh_table, self.zone_ch, self.all_chs, self.stand_alone, self.zone_stand_alone
        )
        # after set_ch_to_veh, attach as direct member
        self.bus_table, self.veh_table, self.stand_alone, self.zone_stand_alone = util.add_member(
            best_target, self.bus_table, veh_id, self.veh_table, config, 0.0, self.time,
            set(), {best_target}, self.stand_alone, self.zone_stand_alone, set()
        )
        uhelp.sync_node_cluster_fields(veh_id, self.veh_table, self.bus_table)


    def update_cluster(self, veh_ids, config):
        """
        PMC-style clustering update:
          - maintain existing CHs and members
          - try CH merging among neighboring CHs
          - unresolved vehicles first try to join existing CHs, then CMs within MAX_HOP
          - unresolved SAs are left for passive cluster formation in stand_alones_cluster
        """
        # pass 1: clear local tick fields
        for veh_id in list(self.veh_table.ids()):
            self._pmc_clear_local_fields(veh_id)
            uhelp.sync_node_cluster_fields(veh_id, self.veh_table, self.bus_table)

        # pass 2: maintain CHs
        for veh_id in list(self.veh_table.ids()):
            st = self.veh_table.values(veh_id)
            if st['cluster_head'] is not True:
                continue

            temp_cluster_members = st['cluster_members'].copy()
            for m in temp_cluster_members:
                if m not in self.veh_table.ids():
                    continue
                dist = util.det_dist(veh_id, self.veh_table, m, self.veh_table)
                if dist > min(st['trans_range'], self.veh_table.values(m)['trans_range']):
                    if m in st['cluster_members']:
                        self.veh_table, self.bus_table, self.stand_alone, self.zone_stand_alone = util.remove_member(
                            m, veh_id, self.veh_table, self.bus_table, config,
                            self.stand_alone, self.zone_stand_alone
                        )

            if (len(st['cluster_members']) == 0 and len(st.get('sub_cluster_members', set())) == 0 and
                    (st['start_ch_zone'] != st['macro_zone']) and
                    (st['prev_macro_zone'] != st['macro_zone'])):
                self.veh_table, self.zone_ch, self.all_chs, self.stand_alone, self.zone_stand_alone = util.set_ch_to_veh(
                    veh_id, self.veh_table, self.zone_ch, self.all_chs, self.stand_alone, self.zone_stand_alone
                )
            else:
                self._pmc_try_merge_ch(veh_id, config)

        # pass 3: maintain already attached members
        for veh_id in list(self.veh_table.ids()):
            st = self.veh_table.values(veh_id)
            if st['cluster_head'] is True or st['primary_ch'] is None:
                continue
            parent_id = st['secondary_ch'] if st['secondary_ch'] is not None else st['primary_ch']
            root_id = st['primary_ch']
            if parent_id is None or root_id is None:
                self._pmc_detach(veh_id, config)
                continue
            try:
                parent_table = self.bus_table if 'bus' in parent_id else self.veh_table
                dist = util.det_dist(veh_id, self.veh_table, parent_id, parent_table)
                still_feasible = dist <= min(st['trans_range'], parent_table.values(parent_id)['trans_range'])
                if st.get('hop_count', 1) > getattr(config, 'h_max', 2):
                    still_feasible = False
            except Exception:
                still_feasible = False
            if still_feasible:
                rec = st['cluster_record'].tail.value
                if rec['start_time'] is not None and rec['start_time'] + rec['timer'] - 1 != self.time:
                    rec['timer'] += 1
            else:
                self._pmc_detach(veh_id, config)

        # pass 4: unresolved vehicles try joining existing clusters
        for veh_id in list(self.stand_alone.copy()):
            if veh_id not in self.veh_table.ids():
                continue
            st = self.veh_table.values(veh_id)
            if st['cluster_head'] or st['primary_ch'] is not None or st['in_area'] is not True:
                continue

            parent, bus_candidates, ch_candidates, sub_ch_candidates, other_vehs = self._pmc_best_existing_parent(veh_id, config)
            if parent is not None:
                self._pmc_attach(veh_id, parent, config, bus_candidates, ch_candidates, other_vehs)
            else:
                # remain SA for passive election stage
                self.stand_alone.add(veh_id)
                self.zone_stand_alone[st['macro_zone']].add(veh_id)



    def single_hop(self, veh_id, config,
                   bus_candidates, ch_candidates, other_vehs):
        """PMC branch keeps joining logic inside update_cluster; retained for compatibility."""
        return

    def multi_hop(self, veh_id, config, bus_candidates,
                  ch_candidates, sub_ch_candidates, other_vehs):
        """PMC branch keeps joining logic inside update_cluster; retained for compatibility."""
        return

    def stand_alones_cluster(self, configs):
        """
        PMC-style passive cluster formation among unresolved SAs.
        Vehicles follow the best higher-priority one-hop SA; roots become CHs passively.
        Remaining unresolved nodes decrement counter and may become isolated CHs.
        """
        _, _, _, max_member, h_max = self._pmc_params(configs)
        sa_ids = [vid for vid in list(self.stand_alone) if vid in self.veh_table.ids() and self.veh_table.values(vid)['primary_ch'] is None and self.veh_table.values(vid)['cluster_head'] is False]
        parent_map = {}

        # each SA follows the best more-stable one-hop SA by priority
        for veh_id in sa_ids:
            st = self.veh_table.values(veh_id)
            neigh = util.det_near_sa(veh_id, self.veh_table, self.stand_alone, self.zone_stand_alone)
            candidates = []
            for j in neigh:
                if j not in sa_ids:
                    continue
                if self._pmc_bech(j, veh_id):
                    candidates.append(j)
            if candidates:
                best = min(candidates, key=lambda j: self._pmc_priority(veh_id, j, configs))
                parent_map[veh_id] = best
            else:
                parent_map[veh_id] = None

        # detect passive roots
        roots = set()
        for veh_id in sa_ids:
            seen = []
            cur = veh_id
            while cur is not None and cur not in seen:
                seen.append(cur)
                cur = parent_map.get(cur)
            if cur is None:
                roots.add(seen[-1])
            else:
                cyc = seen[seen.index(cur):]
                best = cyc[0]
                for x in cyc[1:]:
                    if self._pmc_bech(x, best):
                        best = x
                roots.add(best)
                for x in cyc:
                    if x != best and parent_map.get(x) == best:
                        continue

        # set roots as CHs if they have neighbors/followers, else let counter decide later
        for r in list(roots):
            if r not in self.veh_table.ids():
                continue
            if self.veh_table.values(r)['cluster_head'] is False:
                self.veh_table, self.all_chs, self.stand_alone, self.zone_stand_alone, self.zone_ch = util.set_ch(
                    r, self.veh_table, self.all_chs, self.stand_alone, self.zone_stand_alone, self.zone_ch, configs, its_sa_clustering=True
                )
                uhelp.sync_node_cluster_fields(r, self.veh_table, self.bus_table)

        # attach non-roots along parent chains up to h_max
        progress = True
        while progress:
            progress = False
            for veh_id in sa_ids:
                if veh_id not in self.veh_table.ids():
                    continue
                st = self.veh_table.values(veh_id)
                if st['cluster_head'] or st['primary_ch'] is not None:
                    continue
                parent = parent_map.get(veh_id)
                if parent is None or parent not in self.veh_table.ids():
                    continue
                pst = self.veh_table.values(parent)
                if pst['cluster_head'] is True:
                    if self._pmc_can_accept_parent(parent, configs):
                        self._pmc_attach(veh_id, parent, configs, set(), {parent}, set())
                        progress = True
                elif pst['primary_ch'] is not None and pst.get('hop_count', 1) < h_max and self._pmc_can_accept_parent(parent, configs):
                    self._pmc_attach(veh_id, parent, configs, set(), set(), set())
                    progress = True

        # unresolved nodes decrement counter and may become isolated CHs
        for veh_id in list(self.stand_alone.copy()):
            if veh_id not in self.veh_table.ids():
                continue
            st = self.veh_table.values(veh_id)
            if st['primary_ch'] is not None or st['cluster_head'] is True:
                continue
            if st['counter'] > 0:
                st['counter'] -= 1
            if st['counter'] <= 0:
                self.veh_table, self.all_chs, self.stand_alone, self.zone_stand_alone, self.zone_ch = util.set_ch(
                    veh_id, self.veh_table, self.all_chs, self.stand_alone, self.zone_stand_alone, self.zone_ch, configs, its_sa_clustering=True
                )
                uhelp.sync_node_cluster_fields(veh_id, self.veh_table, self.bus_table)
                self.veh_table.values(veh_id)['counter'] = configs.counter

        # final consistency
        for veh_id in self.veh_table.ids():
            uhelp.sync_node_cluster_fields(veh_id, self.veh_table, self.bus_table)

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
            T_i = depart_time - arrive_time + 1
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

    def _tvct_state(self, veh_id):
        """
        Return compact clustering state for TVCT counting.
        Leaving area / absent vehicle is handled outside and should not be counted.
        """
        if veh_id not in self.veh_table.ids():
            return None

        st = self.veh_table.values(veh_id)

        if st['cluster_head'] is True:
            return ('CH', veh_id)

        if st['primary_ch'] is not None:
            return ('MEM', st['primary_ch'])

        return ('SA', None)

    def _snapshot_tvct_states(self):
        """
        Snapshot only active in-area vehicles before clustering actions of the tick.
        """
        snap = {}
        for veh_id in self.veh_table.ids():
            st = self.veh_table.values(veh_id)
            if st.get('in_area', False) is True:
                snap[veh_id] = self._tvct_state(veh_id)
        return snap

    def update_tvct(self, pre_states):
        """
        Compute TVCT(t) by comparing states before and after clustering for active vehicles.

        Rules:
          - SA -> CH: count 1
          - CH -> SA: count 1
          - SA -> MEM: count 1
          - MEM -> SA: count 1
          - CH -> MEM: count 1
          - MEM -> CH: count 1
          - MEM(root_a) -> MEM(root_b), root_a != root_b: count 1
          - MEM(root_a) -> CH(root_a or self): count 1
          - CH -> MEM(root_b): count 1
          - No change or only parent/RN change within same root: count 0
          - Vehicle leaving area / disappearing: count 0
          - New arriving vehicle in same tick: count 0 unless you explicitly want otherwise
        """
        tvct_t = 0

        post_ids = set(self.veh_table.ids())
        pre_ids = set(pre_states.keys())

        # only vehicles that existed before and still exist after count for transition comparison
        common_ids = pre_ids.intersection(post_ids)

        for veh_id in common_ids:
            pre_state = pre_states[veh_id]
            post_state = self._tvct_state(veh_id)

            # if out of area after update, skip
            if post_state is None:
                continue

            # unpack
            pre_role, pre_root = pre_state
            post_role, post_root = post_state

            # same exact state
            if pre_role == post_role and pre_root == post_root:
                continue

            # member changing RN inside same root should count 0
            # since state only stores root for members, this is already handled:
            # ('MEM', same_root) -> ('MEM', same_root) becomes no change.

            # all other meaningful cluster-affiliation changes count 1
            tvct_t += 1

        self.tvct_history.append(tvct_t)
        self.tvct_cumulative += tvct_t
        self.avg_tvct_history.append(self.tvct_cumulative / len(self.tvct_history))

        return tvct_t, self.avg_tvct_history[-1]

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