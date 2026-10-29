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
import math
import random
from bisect import bisect_left
from functools import lru_cache

from graph import Graph
import utils.util as util
import utils.util_helper as uhelp
import utils.util_graph as util_graph
import hash
import builtins


class _IndexedHashTable(hash.HashTable):
    """Drop-in HashTable with O(1) value lookup and O(active) ids().

    Preserves legacy chained buckets (data_map) and every public method;
    maintains a parallel index only for entries inserted through set_item().
    Returning a fresh set from ids() retains the original API semantics.
    """
    def __init__(self, size):
        super().__init__(size)
        self._active_records = {}

    def hash_index(self, key):
        # The original sum-of-character-codes hash produces severe collisions
        # for SUMO IDs such as veh123 and veh321. A spread C-level hash avoids
        # long chained-list removals without changing lookup semantics.
        return builtins.hash(str(key)) % len(self.data_map)

    def set_item(self, key, value):
        result = super().set_item(key, value)
        # Preserve legacy duplicate-key read semantics: first inserted wins.
        self._active_records.setdefault(key, value)
        return result

    def values(self, key):
        return self._active_records.get(key)

    def ids(self):
        return set(self._active_records)

    def remove(self, key):
        result = super().remove(key)
        # The original chained table permits duplicate keys. After removing
        # its first occurrence, a subsequent duplicate becomes visible.
        successor = super().values(key)
        if successor is None:
            self._active_records.pop(key, None)
        else:
            self._active_records[key] = successor
        return result


class DataTable:
    # This class is determined for defining the hash_table, updating data, routing messages,
    # and defining ip addresses by using trace (which is sumo_trace)

    def __init__(self, config, zones):
        """

        :param config: look at options.py and config.py
        :param zones: all the zones of the area
        """

        self.map = None
        self.bus_table = _IndexedHashTable(config.n_cars * 100)
        self.veh_table = _IndexedHashTable(config.n_cars * 100)
        self.micro_zones = zones['micro']
        self.meso_zones = zones['meso']
        self.macro_zones = zones['macro']
        # Existing neighbor-zone enumeration is deterministic per macro-zone;
        # cache it once per active ZoneID object, without changing the API.
        original_neighbors = self.macro_zones.neighbor_zones
        cached_neighbors = lru_cache(maxsize=50000)(
            lambda zone_id: tuple(original_neighbors(zone_id)))
        # Return a new list on each call, as in the original ZoneID API.
        self.macro_zones.neighbor_zones = lambda zone_id: list(cached_neighbors(zone_id))

        # The original code repeatedly scanned all 2,000 zone hash buckets.
        zone_ids = self.macro_zones.zone_hash.ids()
        self.zone_vehicles = {zone_id: set() for zone_id in zone_ids}
        self.zone_buses = {zone_id: set() for zone_id in zone_ids}
        self.zone_ch = {zone_id: set() for zone_id in zone_ids}
        self.zone_stand_alone = {zone_id: set() for zone_id in zone_ids}
        self.stand_alone = set()
        self.all_chs = set()
        self.left_veh = dict()
        self.left_bus = dict()
        self.time = config.start_time
        self._sumo_timesteps = config.sumo_trace.documentElement.getElementsByTagName('timestep')
        # Pairwise geographic distances and headings are valid only within one tick.
        util.mmzca_begin_tick()
        self.understudied_area = self.macro_zones.understudied_area()
        self.init_count = 0  # this counter is just for defining the self.net_graph for the very first time
        self.edge_color = ''
        self.sumo_edges, self.sumo_nodes = util.sumo_net_info(config.sumo_edge, config.sumo_node)
        self.ch_net = None

        self.tvct_history = []  # stores TVCT(t) at each tick
        self.avg_tvct_history = []  # stores running average up to each tick
        self.tvct_cumulative = 0.0
        self.tvct_events = []       # event log: includes intra-tick reversals
        self._tvct_events_this_tick = []

        # MMZCA kernel protocol state. Per-vehicle/CH histograms, descriptors,
        # and versions live IN self.veh_table / self.bus_table dictionaries.
        # Only in-flight one-hop messages and persistent CH incarnation numbers
        # are kept here at the simulation-coordination level.
        self.kernel_messages = []
        self.kernel_root_registry = {}
        self.kernel_stats = {'kernel_messages_sent': 0,
                             'kernel_control_bytes_est': 0}

        for veh in self._sumo_timesteps[self.time].childNodes[
                   1::2]:
            self.init_count += 1
            micro_zone_id = self._mmzca_fast_zone(self.micro_zones, float(veh.getAttribute('y')),
                                                      # determine the micro_zone_id of the car (bus | veh)
                                                      float(veh.getAttribute('x'))
                                                      )
            meso_zone_id = self._mmzca_fast_zone(self.meso_zones, float(veh.getAttribute('y')),
                                                      # determine the meso_zone_id of the car (bus | veh)
                                                      float(veh.getAttribute('x'))
                                                      )
            macro_zone_id = self._mmzca_fast_zone(self.macro_zones, float(veh.getAttribute('y')),  # determine the macro_zone_id of the car (bus | veh)
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

        # Initial SUMO timestep is an observed tick, too. Ordinary vehicles
        # start as SAs, so this makes their presence denominator exact.
        self.record_mmzca_stability()

    @staticmethod
    def _mmzca_fast_zone(zones, lat, lon):
        """O(log rows + log cols) zone lookup for the existing regular grid.

        For boundary/outside cases use the original routine to retain its
        exact tie-breaking semantics. Does not alter ZoneID or zone IDs.
        """
        rows, cols = zones.lat_rows, zones.long_cols
        if lat <= rows[0] or lat >= rows[-1] or lon <= cols[0] or lon >= cols[-1]:
            return zones.det_zone(lat, lon)
        r = bisect_left(rows, lat) - 1
        c = bisect_left(cols, lon) - 1
        if (r < 0 or c < 0 or r >= zones.n_rows or c >= zones.n_cols
                or lat == rows[r] or lat == rows[r+1]
                or lon == cols[c] or lon == cols[c+1]):
            return zones.det_zone(lat, lon)
        return zones.zoning_mode + '_zone' + str(r * zones.n_cols + c)

    def update(self, config):
        """
        Update bus_table and veh_table for the next timestep.

        This version is compatible with the generalized MMZCA state:
          - micro / meso / macro zones
          - root_ch / parent_node / hop_count
          - legacy primary_ch / secondary_ch compatibility
        """
        self.time += 1
        begin_tick = getattr(util, 'mmzca_begin_tick', None)
        if begin_tick is not None:
            begin_tick()
        self._tvct_events_this_tick = []
        zone_lookup = getattr(self, '_mmzca_fast_zone', None)
        if zone_lookup is None:  # compatibility with isolated regression harnesses
            zone_lookup = lambda z, y, x: z.det_zone(y, x)
        timesteps = getattr(self, '_sumo_timesteps', None)
        if timesteps is None:
            timesteps = config.sumo_trace.documentElement.getElementsByTagName('timestep')
        bus_ids = set()
        veh_ids = set()
        self.net_graph.remove_edges_from(self.net_graph.edges())

        # -------- update all active nodes from SUMO --------
        for veh in timesteps[self.time].childNodes[1::2]:
            micro_zone_id = zone_lookup(self.micro_zones,
                float(veh.getAttribute('y')),
                float(veh.getAttribute('x'))
            )
            meso_zone_id = zone_lookup(self.meso_zones,
                float(veh.getAttribute('y')),
                float(veh.getAttribute('x'))
            )
            macro_zone_id = zone_lookup(self.macro_zones,
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
                self.bus_table.values(veh_id)['path_bottleneck'] = 0.0
                self.bus_table.values(veh_id)['committed_link_score'] = None
                self.bus_table.values(veh_id)['root_connection_valid'] = True
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
            # Start with ACTUAL immediate children; remove_member recursively
            # invalidates every descendant before its ancestor.
            children = self._mmzca_children()
            for m in list(children.get(k, ())):
                before = self._mmzca_subtree_tvct_states(m, children)
                self.veh_table, self.bus_table, self.stand_alone, self.zone_stand_alone = \
                    util.remove_member(
                        m, k, self.veh_table, self.bus_table, config,
                        self.stand_alone, self.zone_stand_alone,
                        ch_stays=False
                    )
                self._mmzca_log_tvct_transitions(before, 'bus_departure')

            self.zone_buses[self.bus_table.values(k)['macro_zone']].discard(k)
            self.zone_ch[self.bus_table.values(k)['macro_zone']].discard(k)
            self.all_chs.discard(k)

            self.bus_table.values(k)['depart_time'] = self.time - 1
            self.left_bus[k] = self.bus_table.values(k)

            self.bus_table.remove(k)
            self.net_graph.remove_node(k)

        # -------- remove vehicles that left the area --------
        temp_left_vehs = self.veh_table.ids() - veh_ids
        for k in temp_left_vehs:
            if self.veh_table.values(k)['cluster_head'] is True:
                children = self._mmzca_children()
                for m in list(children.get(k, ())):
                    before = self._mmzca_subtree_tvct_states(m, children)
                    self.veh_table, self.bus_table, self.stand_alone, self.zone_stand_alone = \
                        util.remove_member(
                            m, k, self.veh_table, self.bus_table, config,
                            self.stand_alone, self.zone_stand_alone,
                            ch_stays=False
                        )
                    self._mmzca_log_tvct_transitions(before, 'vehicle_ch_departure')

                self.zone_ch[self.veh_table.values(k)['macro_zone']].discard(k)
                self.all_chs.discard(k)

            elif self._mmzca_parent(k) is not None:
                # A departing CM may itself have arbitrary-depth descendants.
                # The corrected util.remove_* methods detach its whole branch.
                before = self._mmzca_subtree_tvct_states(k)
                k_root = self._mmzca_root(k)
                k_parent = self._mmzca_parent(k)

                if k_parent is not None and k_root is not None:
                    if k_parent == k_root:
                        self.veh_table, self.bus_table, self.stand_alone, self.zone_stand_alone = \
                            util.remove_member(
                                k, k_root, self.veh_table, self.bus_table, config,
                                self.stand_alone, self.zone_stand_alone,
                                mem_stays=False
                            )
                    else:
                        self.veh_table, self.bus_table, self.stand_alone, self.zone_stand_alone = \
                            util.remove_sub_member(
                                k, k_parent, k_root,
                                self.veh_table, self.bus_table, config,
                                self.stand_alone, self.zone_stand_alone,
                                sub_mem_stays=False
                            )
                self._mmzca_log_tvct_transitions(before, 'cm_departure')

            elif k in self.stand_alone:
                self.stand_alone.discard(k)
                self.zone_stand_alone[self.veh_table.values(k)['macro_zone']].discard(k)

            self.stand_alone.discard(k)
            self.zone_stand_alone[self.veh_table.values(k)['macro_zone']].discard(k)
            self.zone_vehicles[self.veh_table.values(k)['macro_zone']].discard(k)
            self.veh_table.values(k)['depart_time'] = self.time - 1
            self.left_veh[k] = self.veh_table.values(k)

            self.veh_table.remove(k)
            self.net_graph.remove_node(k)

        # -------- final consistency pass for active vehicles --------
        for veh_id in self.veh_table.ids():
            uhelp.sync_node_cluster_fields(veh_id, self.veh_table, self.bus_table)

    def _mmzca_is_sa(self, veh_id):
        st = self.veh_table.values(veh_id)
        return (st is not None and st.get('cluster_head') is not True
                and st.get('primary_ch') is None and st.get('parent_node') is None
                and st.get('in_area', True) is not False)

    def _mmzca_record(self, node_id):
        return self.veh_table.values(node_id) or self.bus_table.values(node_id)

    def _mmzca_parent(self, node_id):
        st = self._mmzca_record(node_id)
        if st is None:
            return None
        return (st.get('parent_node') or st.get('parent_id') or
                st.get('secondary_ch') or st.get('primary_ch'))

    def _mmzca_root(self, node_id):
        st = self._mmzca_record(node_id)
        if st is None:
            return None
        if self.bus_table.values(node_id) is not None or st.get('cluster_head') is True:
            return node_id
        return st.get('root_ch') or st.get('primary_ch')

    def _mmzca_children(self):
        """Use immediate parent pointers, never root cluster_members, for edges."""
        children = {node_id: set() for node_id in self.veh_table.ids() | self.bus_table.ids()}
        for node_id in self.veh_table.ids():
            parent_id = self._mmzca_parent(node_id)
            if parent_id in children and parent_id != node_id and not self._mmzca_is_sa(node_id):
                children[parent_id].add(node_id)
        return children

    def _mmzca_subtree_tvct_states(self, root_id, children=None):
        """Snapshot active vehicle states in a descendant subtree."""
        if children is None:
            children = self._mmzca_children()
        states, seen, stack = {}, set(), [root_id]
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            if node_id in self.veh_table.ids():
                states[node_id] = self._tvct_state(node_id)
            stack.extend(children.get(node_id, ()))
        return states

    def _mmzca_log_tvct_transitions(self, previous_states, reason):
        """Record meaningful changes immediately, including reversals in a tick."""
        for node_id, before in previous_states.items():
            after = self._tvct_state(node_id)
            if before is not None and after is not None and before != after:
                event = dict(time=self.time, vehicle=node_id, before=before,
                             after=after, reason=reason)
                self._tvct_events_this_tick.append(event)
                self.tvct_events.append(event)

    def _mmzca_sync_direct_children(self):
        children = self._mmzca_children()
        for node_id, child_ids in children.items():
            self._mmzca_record(node_id)['direct_children'] = child_ids
        return children

    def record_mmzca_stability(self):
        """Record one end-of-tick role/root state per ordinary vehicle.

        Run once after clustering in each 1-second SUMO tick. Records stay in
        veh_table and are run-length encoded. Unlike legacy cluster_record,
        these episodes also represent empty/effective CH states and in-area
        presence accurately. Parent switches WITHIN one root do not split CM
        residence episodes. Bus CHs are excluded from vehicle-based metrics.
        """
        children = self._mmzca_children()
        for veh_id in self.veh_table.ids():
            st = self.veh_table.values(veh_id)
            if not st.get('in_area', False):
                state = ('OUT', None, False)
            elif st.get('cluster_head') is True:
                state = ('CH', veh_id, bool(children.get(veh_id)))
            else:
                root = st.get('root_ch') or st.get('primary_ch')
                state = ('CM', root, True) if root is not None else ('SA', None, False)

            last_tick = st.get('stability_last_tick')
            if last_tick == self.time:
                raise RuntimeError('record_mmzca_stability must run once per tick')
            episodes = st.setdefault('stability_episodes', [])
            if (episodes and episodes[-1]['end_time'] == self.time - 1 and
                    (episodes[-1]['role'], episodes[-1]['root'],
                     episodes[-1]['effective']) == state):
                episodes[-1]['end_time'] = self.time
                episodes[-1]['timer'] += 1
            else:
                episodes.append(dict(start_time=self.time, end_time=self.time,
                                     timer=1, role=state[0], root=state[1],
                                     effective=state[2]))
            st['stability_last_tick'] = self.time

    def _mmzca_valid_committed_path(self, node_id, config):
        """Check exactly the committed parent chain and each physical edge."""
        root_id = self._mmzca_root(node_id)
        if root_id is None:
            return False
        root = self._mmzca_record(root_id)
        if root is None or (self.bus_table.values(root_id) is None
                            and root.get('cluster_head') is not True):
            return False
        epoch = root.get('kernel_root_incarnation')
        seen = set()
        current = node_id
        expected = None
        while current not in seen:
            seen.add(current)
            node = self._mmzca_record(current)
            if node is None or node.get('root_connection_valid') is False:
                return False
            if self._mmzca_root(current) != root_id:
                return False
            node_epoch = node.get('kernel_root_incarnation')
            if epoch is not None and node_epoch is not None and epoch != node_epoch:
                return False
            try:
                depth = int(node['hop_count'])
            except (KeyError, ValueError, TypeError):
                return False
            if depth < 0 or depth > int(config.h_max):
                return False
            if expected is not None and depth != expected:
                return False
            if current == root_id:
                return depth == 0
            if depth == 0:
                return False
            parent_id = self._mmzca_parent(current)
            parent = self._mmzca_record(parent_id)
            if parent is None:
                return False
            try:
                dist = util.det_dist(
                    current, self.bus_table if self.bus_table.values(current) is not None else self.veh_table,
                    parent_id, self.bus_table if self.bus_table.values(parent_id) is not None else self.veh_table
                )
                bound = min(float(node['trans_range']), float(parent['trans_range']))
                if not math.isfinite(dist) or dist > bound:
                    return False
            except (KeyError, TypeError, ValueError):
                return False
            expected = depth - 1
            current = parent_id
        return False

    def _mmzca_detach_branch(self, veh_id, config):
        """Invalidate an entire dependent branch, removing leaves first.

        This is the safe fallback when an atomic subtree-move API is absent.
        """
        children = self._mmzca_children()
        before = self._mmzca_subtree_tvct_states(veh_id, children)
        branch = []
        seen = set()

        def collect(node_id):
            if node_id in seen:
                return
            seen.add(node_id)
            for child_id in children.get(node_id, ()):
                collect(child_id)
            branch.append(node_id)

        collect(veh_id)
        for node_id in branch:
            st = self.veh_table.values(node_id)
            if st is None or self._mmzca_is_sa(node_id):
                continue
            parent_id = self._mmzca_parent(node_id)
            root_id = self._mmzca_root(node_id)
            if parent_id is None or root_id is None:
                raise RuntimeError('Cannot detach inconsistent MMZCA member ' + str(node_id))
            if parent_id == root_id:
                self.veh_table, self.bus_table, self.stand_alone, self.zone_stand_alone = \
                    util.remove_member(node_id, root_id, self.veh_table, self.bus_table,
                                       config, self.stand_alone, self.zone_stand_alone)
            else:
                self.veh_table, self.bus_table, self.stand_alone, self.zone_stand_alone = \
                    util.remove_sub_member(node_id, parent_id, root_id,
                                           self.veh_table, self.bus_table, config,
                                           self.stand_alone, self.zone_stand_alone)
            st = self.veh_table.values(node_id)
            if st is None:
                continue
            uhelp.reset_cluster_fields(node_id, self.veh_table, self.bus_table)
            st['root_ch'] = None
            st['parent_node'] = None
            st['hop_count'] = None
            st['path_bottleneck'] = None
            st['committed_link_score'] = None
            st['root_connection_valid'] = False
            st['direct_children'] = set()
            st['kernel_descriptor'] = None
            st['priority_ch'] = root_id
            st['priority_counter'] = int(config.priority_counter)
            st['counter'] = max(1, int(st.get('counter') or 0))
            self.stand_alone.add(node_id)
            self.zone_stand_alone[st['macro_zone']].add(node_id)
        self._mmzca_log_tvct_transitions(before, 'parent_link_failure')

    def _mmzca_join_sa(self, veh_id, config):
        """One selection/attachment path for both CH and CM parent candidates."""
        if not self._mmzca_is_sa(veh_id):
            return False
        before = self._tvct_state(veh_id)
        bus_candidates, ch_candidates, cm_candidates, other_vehs = util.det_near_ch(
            veh_id, self.veh_table, self.bus_table,
            self.zone_buses, self.zone_vehicles
        )
        bus_candidates = set(bus_candidates)
        ch_candidates = set(ch_candidates)
        cm_candidates = set(cm_candidates)
        st = self.veh_table.values(veh_id)

        # Same-root backup gets first opportunity during the bounded priority
        # window; infeasibility immediately opens the other-root fallback.
        options = []
        prior_root = st.get('priority_ch')
        if prior_root is not None and st.get('priority_counter', 0) > 0:
            preferred = tuple(
                {j for j in group if self._mmzca_root(j) == prior_root}
                for group in (bus_candidates, ch_candidates, cm_candidates)
            )
            if any(preferred):
                options.append(preferred)
        options.append((bus_candidates, ch_candidates, cm_candidates))

        for initial_groups in options:
            remaining = [set(group) for group in initial_groups]
            while any(remaining):
                parent_id, score, detail = util.choose_mmzca_parent(
                    veh_id, self.veh_table, self.bus_table,
                    remaining[0], remaining[1], remaining[2],
                    self.micro_zones, self.meso_zones, self.macro_zones,
                    config, self.time, return_details=True
                )
                if parent_id is None:
                    break
                root_id = detail['root_id']
                parent = self._mmzca_record(parent_id)
                root = self._mmzca_record(root_id)
                # Recheck immediate eligibility just before commitment.
                feasible = (parent is not None and root is not None
                            and self._mmzca_valid_committed_path(parent_id, config)
                            and parent.get('admit_children') is not False
                            and detail['depth'] <= int(config.h_max))
                if feasible:
                    self.bus_table, self.veh_table, self.stand_alone, self.zone_stand_alone = \
                        uhelp.attach_to_parent(
                            root_id, parent_id, veh_id,
                            self.veh_table, self.bus_table, config, self.time,
                            bus_candidates, ch_candidates,
                            self.stand_alone, self.zone_stand_alone, other_vehs,
                            root_cost_value=score if parent_id == root_id else None,
                            parent_cost_value=score
                        )
                    st = self.veh_table.values(veh_id)
                    if (st is None or st.get('primary_ch') != root_id
                            or (parent_id != root_id and st.get('secondary_ch') != parent_id)):
                        raise RuntimeError(
                            'attach_to_parent must commit the selected root and immediate parent; '
                            'check utils/util_helper.py for arbitrary MMZCA hop support'
                        )
                    # Authoritative generalized hierarchy, in the same records.
                    st['root_ch'] = root_id
                    st['parent_node'] = parent_id
                    st['hop_count'] = detail['depth']
                    st['path_bottleneck'] = detail['Q_candidate']
                    st['committed_link_score'] = detail['L']
                    st['root_connection_valid'] = True
                    st['kernel_root_incarnation'] = root.get('kernel_root_incarnation')
                    st['current_parent_cost'] = score
                    st['priority_ch'] = None
                    st['priority_counter'] = int(config.priority_counter)
                    self._mmzca_record(parent_id).setdefault('direct_children', set()).add(veh_id)
                    self.stand_alone.discard(veh_id)
                    self.zone_stand_alone[st['macro_zone']].discard(veh_id)
                    self._mmzca_log_tvct_transitions(
                        {veh_id: before}, 'join_parent'
                    )
                    return True

                for group in remaining:
                    group.discard(parent_id)
        return False

    def _mmzca_switch_valid_parents(self, config):
        """Electively reparent only within the SAME valid root, with hysteresis.

        No root switching: members retain their original root as long as its
        committed path stays valid. For an invalid path, existing detachment / SA
        reattachment remains the separate recovery behavior.
        """
        members = sorted(
            (vid for vid in self.veh_table.ids()
             if not self._mmzca_is_sa(vid)
             and self.veh_table.values(vid).get('cluster_head') is not True),
            key=lambda vid: int(self.veh_table.values(vid).get('hop_count') or 0)
        )
        # Parent pointers are the source of truth; reuse this tree across
        # all candidate evaluations in the current tick rather than rescanning
        # every vehicle for each CM.
        children = self._mmzca_children()
        switches = 0
        for veh_id in members:
            st = self.veh_table.values(veh_id)
            if (st is None or self._mmzca_is_sa(veh_id)
                    or not self._mmzca_valid_committed_path(veh_id, config)):
                continue
            root_id = self._mmzca_root(veh_id)
            current_parent = self._mmzca_parent(veh_id)
            if root_id is None or current_parent is None:
                continue
            descendants = set()
            stack = [(veh_id, 0)]
            tail_depth = 0
            while stack:
                node_id, distance = stack.pop()
                if node_id in descendants:
                    break
                descendants.add(node_id)
                tail_depth = max(tail_depth, distance)
                stack.extend((child, distance+1) for child in children.get(node_id, ()))
            else:
                bus_candidates, ch_candidates, cm_candidates, _ = util.det_near_ch(
                    veh_id, self.veh_table, self.bus_table,
                    self.zone_buses, self.zone_vehicles
                )
                bus_candidates, ch_candidates = set(bus_candidates), set(ch_candidates)
                cm_candidates = set(cm_candidates)
                # Ensure the incumbent is present for a fresh, comparable score.
                if self.bus_table.values(current_parent) is not None:
                    bus_candidates.add(current_parent)
                elif current_parent == root_id:
                    ch_candidates.add(current_parent)
                else:
                    cm_candidates.add(current_parent)
                for group in (bus_candidates, ch_candidates, cm_candidates):
                    group.difference_update(descendants)
                    group.intersection_update(
                        {j for j in group if self._mmzca_root(j) == root_id}
                    )
                if not (bus_candidates or ch_candidates or cm_candidates):
                    continue
                parent_id, score, detail = util.choose_mmzca_parent(
                    veh_id, self.veh_table, self.bus_table,
                    bus_candidates, ch_candidates, cm_candidates,
                    self.micro_zones, self.meso_zones, self.macro_zones,
                    config, self.time, return_details=True,
                    allowed_root_id=root_id,
                    extra_descendant_hops=tail_depth
                )
                if parent_id is None or parent_id == current_parent:
                    continue
                # The selector compares the incumbent and alternative under
                # one evaluation mode and enforces parent_hysteresis_delta.
                # Do not switch if the incumbent was not comparably scored.
                feasible = detail.get('all_candidates', {})
                incumbent = feasible.get(current_parent)
                selected = feasible.get(parent_id)
                mode = detail.get('selection_mode')
                key = 'G0' if mode == 'physical' else 'Gaff'
                if (incumbent is None or selected is None or
                        detail['root_id'] != root_id or
                        current_parent not in feasible):
                    continue
                # In mixed freshness, only compare a common score; never
                # treat missing affinity as an artificially favorable zero.
                if mode != 'physical' and incumbent['kernel_status'] == 'unavailable':
                    continue
                improvement = incumbent[key] - selected[key]
                if improvement <= float(config.parent_hysteresis_delta) + float(config.selection_tolerance):
                    continue
                try:
                    moved = uhelp.reparent_within_root(
                        veh_id, parent_id, self.veh_table, self.bus_table,
                        config, self.meso_zones, self.macro_zones, score,
                        child_map=children
                    )
                except ValueError:
                    moved = False
                if moved:
                    children[current_parent].discard(veh_id)
                    children[parent_id].add(veh_id)
                    switches += 1
        self._mmzca_sync_direct_children()
        self.mmzca_parent_switches = getattr(self, 'mmzca_parent_switches', 0) + switches
        return switches

    def update_cluster(self, veh_ids, config):
        """Maintain root-connected MMZCA trees; then join unresolved SAs.

        Every CH/CM edge is evaluated against its IMMEDIATE parent, not the
        root CH. A broken branch is invalidated bottom-up; prior-root repair
        gets preference without bypassing feasibility or the selection guard.
        CH election happens only in stand_alones_cluster().
        """
        for veh_id in self.veh_table.ids():
            st = self.veh_table.values(veh_id)
            st['other_chs'] = set()
            st['gates'] = {}
            st['gate_chs'] = set()
            st['other_vehs'] = set()
            uhelp.sync_node_cluster_fields(veh_id, self.veh_table, self.bus_table)
            if st.get('cluster_head') is True:
                st['root_ch'] = veh_id
                st['parent_node'] = None
                st['hop_count'] = 0
                st['path_bottleneck'] = 0.0
                st['committed_link_score'] = None
                st['root_connection_valid'] = True
        for bus_id in self.bus_table.ids():
            st = self.bus_table.values(bus_id)
            st['path_bottleneck'] = 0.0
            st['committed_link_score'] = None
            st['root_connection_valid'] = True

        # A parent fails before its children; detach the whole dependent branch
        # so no descendant retains a false root-connected affiliation.
        members = [k for k in self.veh_table.ids()
                   if not self._mmzca_is_sa(k)
                   and self.veh_table.values(k).get('cluster_head') is not True]
        members.sort(key=lambda k: int(self.veh_table.values(k).get('hop_count') or 0))
        for veh_id in members:
            if self._mmzca_is_sa(veh_id):
                continue
            if not self._mmzca_valid_committed_path(veh_id, config):
                self._mmzca_detach_branch(veh_id, config)
                continue
            st = self.veh_table.values(veh_id)
            parent = self._mmzca_record(self._mmzca_parent(veh_id))
            # Q is the bottleneck of ORIGINAL committed physical link scores.
            # Missing legacy L cannot be reconstructed as the old EF or G.
            upstream = parent.get('path_bottleneck') if parent is not None else None
            committed_link = st.get('committed_link_score')
            if upstream is not None and committed_link is not None:
                st['path_bottleneck'] = max(float(upstream), float(committed_link))
            else:
                # An unverified legacy score must not be advertised as Q_j.
                st['path_bottleneck'] = None
            rec = st.get('cluster_record')
            if rec is not None and rec.tail is not None:
                last = rec.tail.value
                if (last.get('start_time') is not None and last.get('timer') is not None
                        and last['start_time'] + last['timer'] - 1 != self.time):
                    last['timer'] += 1

        # Root-preserving voluntary parent switches for valid attached CMs.
        # No elective root reassociation takes place here.
        self._mmzca_sync_direct_children()
        self._mmzca_switch_valid_parents(config)

        # Empty ordinary CHs can demote after a later macro-zone transition.
        # Buses remain persistent CHs. This test uses the actual parent graph.
        children = self._mmzca_sync_direct_children()
        for ch_id in list(self.veh_table.ids()):
            st = self.veh_table.values(ch_id)
            if st.get('cluster_head') is not True or children.get(ch_id):
                continue
            if (st.get('start_ch_zone') != st.get('macro_zone')
                    and st.get('macro_zone_changed_this_tick', False)):
                before = self._tvct_state(ch_id)
                self.veh_table, self.zone_ch, self.all_chs, self.stand_alone, self.zone_stand_alone = \
                    util.set_ch_to_veh(
                        ch_id, self.veh_table, self.zone_ch,
                        self.all_chs, self.stand_alone, self.zone_stand_alone
                    )
                st = self.veh_table.values(ch_id)
                st.update(root_ch=None, parent_node=None, hop_count=None,
                          path_bottleneck=None, committed_link_score=None,
                          root_connection_valid=False, kernel_descriptor=None)
                st['direct_children'] = set()
                st['counter'] = max(1, int(st.get('counter') or 0))
                self._mmzca_log_tvct_transitions({ch_id: before}, 'ch_demoted')

        self._mmzca_sync_direct_children()
        for veh_id in list(self.stand_alone):
            self._mmzca_join_sa(veh_id, config)

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
                    if self.veh_table.values(veh_id)['cluster_head'] is True:
                        util.mmzca_note_ch_promotion(veh_id, self.kernel_root_registry)
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

    def _mmzca_pch_link_score(self, a_id, b_id, config):
        """Reuse util's physical score for SA-SA PCH election."""
        link = util.mmzca_physical_link(
            a_id, b_id, self.veh_table, self.veh_table,
            self.meso_zones, self.macro_zones, config
        )
        return None if link is None else link['L']

    def stand_alones_cluster(self, configs):
        """Wait lambda, locally elect PCHs, then join the newly formed CHs.

        SA neighbors are prospective CH-election peers, NOT eligible relay
        parents. Only a confirmed CH or root-connected CM can be a parent.
        """
        active_sa = {j for j in self.stand_alone if self._mmzca_is_sa(j)}
        if not active_sa:
            return

        # One countdown per simulation tick. `counter` is the existing lambda
        # waiting variable; no independent decrement occurs in update_cluster.
        eligible = set()
        for veh_id in active_sa:
            st = self.veh_table.values(veh_id)
            st['counter'] = max(0, int(st.get('counter') or 0) - 1)
            if st.get('priority_ch') is not None:
                st['priority_counter'] = max(0, int(st.get('priority_counter') or 0) - 1)
                if st['priority_counter'] == 0:
                    st['priority_ch'] = None
                    st['priority_counter'] = int(configs.priority_counter)
            if st['counter'] == 0:
                eligible.add(veh_id)

        # Form a symmetric, reciprocal-range one-hop SA neighborhood. Zone
        # discovery is only a spatial index; it is not a viability gate.
        near_sa = {}
        for veh_id in active_sa:
            discovered = set(util.det_near_sa(
                veh_id, self.veh_table, self.stand_alone, self.zone_stand_alone
            ))
            near_sa[veh_id] = set()
            for other in discovered & active_sa - {veh_id}:
                if self._mmzca_pch_link_score(veh_id, other, configs) is not None:
                    near_sa[veh_id].add(other)
        # Reciprocal discovery avoids asymmetric elections if indexes lag.
        for veh_id in active_sa:
            near_sa[veh_id] = {j for j in near_sa[veh_id]
                               if veh_id in near_sa.get(j, ())}

        # Paper's PCH order: (-SA degree, average physical link L).
        # Random numbers implement only the final local-contention tie-break.
        ranking = {}
        for veh_id in eligible:
            neighbors = near_sa[veh_id]
            avg_link = (sum(self._mmzca_pch_link_score(veh_id, j, configs)
                            for j in neighbors) / len(neighbors)
                        if neighbors else 1.0)
            ranking[veh_id] = (-len(neighbors), avg_link, random.random())

        # One-hop local minima are winners; two winners cannot be one-hop
        # neighbors. There is no global election over a multi-hop SA component.
        winners = {
            veh_id for veh_id in eligible
            if all(ranking[veh_id] < ranking[j]
                   for j in near_sa[veh_id] & eligible)
        }

        for veh_id in winners:
            if not self._mmzca_is_sa(veh_id):
                continue
            before = self._tvct_state(veh_id)
            self.veh_table, self.all_chs, self.stand_alone, self.zone_stand_alone, self.zone_ch = \
                util.set_ch(
                    veh_id, self.veh_table, self.all_chs,
                    self.stand_alone, self.zone_stand_alone,
                    self.zone_ch, configs, its_sa_clustering=True
                )
            st = self.veh_table.values(veh_id)
            if st['cluster_head'] is not True:
                continue
            epoch = util.mmzca_note_ch_promotion(veh_id, self.kernel_root_registry)
            st.update(root_ch=veh_id, parent_node=None, hop_count=0,
                      path_bottleneck=0.0, committed_link_score=None,
                      root_connection_valid=True, kernel_root_incarnation=epoch,
                      kernel_descriptor=None, current_root_cost=0.0,
                      current_parent_cost=0.0, priority_ch=None)
            st['direct_children'] = set()
            self._mmzca_log_tvct_transitions({veh_id: before}, 'pch_promotion')

        self._mmzca_sync_direct_children()
        # Remaining SAs can now attach to a promoted root, using the same
        # MZOTD/velocity/distance/affinity-aware parent-selection function.
        for veh_id in list(self.stand_alone):
            self._mmzca_join_sa(veh_id, configs)

        self._mmzca_sync_direct_children()
        for veh_id in self.veh_table.ids():
            uhelp.sync_node_cluster_fields(veh_id, self.veh_table, self.bus_table)

    def update_kernels(self, config):
        """Advance MMZCA sparse micro-zone reports and sketch dissemination.

        Call once per tick AFTER update_cluster() and stand_alones_cluster().
        The arriving sketch is not made available to a parent until the
        corresponding one-hop message reaches it at a later tick.
        """
        self.kernel_messages = util.mmzca_kernel_tick(
            self.veh_table, self.bus_table, self.micro_zones, config, self.time,
            self.kernel_messages, self.kernel_root_registry,
            stats=self.kernel_stats, distance_fn=util.det_dist
        )

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

    def vcsm_legacy(self, configs):
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


    def _mmzca_stability_summary(self):
        """Aggregate root-based MMZCA residence from compressed vehicle episodes.

        Observation unit: one simulation tick (= 1 second for these SUMO
        traces). Only ticks with in_area=True enter P_i. A CM and a relay RN
        are the SAME effective member state; their immediate parent may change
        without causing a new root-cluster episode.

        The separate historical cluster_record LinkedList is kept intact for
        the existing clustering logic and vcsm_legacy() comparison.
        """
        vehicles = [(vid, self.veh_table.values(vid)) for vid in self.veh_table.ids()]
        vehicles += list(self.left_veh.items())

        # Prevent a partially updated current tick from producing a plausible
        # but incorrect metric.
        for vid, st in vehicles:
            if vid in self.veh_table.ids() and st.get('stability_last_tick') != self.time:
                raise RuntimeError(
                    'Call record_mmzca_stability() after clustering every tick '
                    'before evaluating MMZCA stability metrics'
                )

        totals = dict(
            presence=0.0, A=0.0, A_cm=0.0, contribution_r=0.0,
            weighted_R=0.0, switches=0.0, switch_den=0.0,
            ch_role_time=0.0, ch_role_runs=0,
            ch_effective_time=0.0, ch_effective_runs=0,
            cm_episode_time=0.0, cm_episode_count=0,
            empty_ch_time=0.0, vcsm_sum=0.0,
            vcsm_member_count=0, vcsm_cm_sum=0.0,
            R_cm_sum=0.0, R_cm_weighted=0.0,
            A_cm_conditional=0.0, P_cm_conditional=0.0,
            cm_switches=0.0, cm_switch_den=0.0
        )
        cm_per_vehicle = {}

        def root_switches(sequence):
            return sum(a != b for a, b in zip(sequence, sequence[1:]))

        def concentration(residence, duration):
            if duration <= 0:
                return 0.0
            return sum((value / duration) ** 2 for value in residence.values())

        def quality(switches, duration):
            if duration <= 1:
                return 1.0
            return max(0.0, min(1.0, 1.0 - switches / (duration - 1.0)))

        for vid, st in vehicles:
            episodes = st.get('stability_episodes', ())
            presence = 0.0
            A_cm = 0.0
            A = 0.0
            cm_residence = {}
            effective_residence = {}
            cm_sequence, effective_sequence = [], []
            cm_episode_durations = []
            ch_role_lifetimes, ch_effective_lifetimes = [], []
            current_ch_role_run = 0.0
            current_ch_effective_run = 0.0
            empty_ch_time = 0.0

            for episode in episodes:
                duration = float(episode['timer'])
                if duration <= 0:
                    continue
                role, root = episode['role'], episode['root']
                effective = bool(episode['effective'])

                if role != 'OUT':
                    presence += duration
                if role == 'CM' and root is not None:
                    A_cm += duration
                    cm_episode_durations.append(duration)
                    cm_sequence.append(root)
                    cm_residence[root] = cm_residence.get(root, 0.0) + duration

                if role == 'CH':
                    current_ch_role_run += duration
                    if not effective:
                        empty_ch_time += duration
                elif current_ch_role_run:
                    ch_role_lifetimes.append(current_ch_role_run)
                    current_ch_role_run = 0.0

                if role == 'CH' and effective:
                    current_ch_effective_run += duration
                elif current_ch_effective_run:
                    ch_effective_lifetimes.append(current_ch_effective_run)
                    current_ch_effective_run = 0.0

                eff_root = root if ((role == 'CM') or
                                    (role == 'CH' and effective)) else None
                if eff_root is not None:
                    A += duration
                    effective_sequence.append(eff_root)
                    effective_residence[eff_root] = (
                        effective_residence.get(eff_root, 0.0) + duration
                    )

            if current_ch_role_run:
                ch_role_lifetimes.append(current_ch_role_run)
            if current_ch_effective_run:
                ch_effective_lifetimes.append(current_ch_effective_run)

            R = concentration(effective_residence, A)
            S = root_switches(effective_sequence)
            Q = quality(S, A)
            R_cm = concentration(cm_residence, A_cm)
            S_cm = root_switches(cm_sequence)
            Q_cm = quality(S_cm, A_cm)

            totals['presence'] += presence
            totals['A'] += A
            totals['A_cm'] += A_cm
            totals['contribution_r'] += A * R * Q
            totals['weighted_R'] += A * R
            totals['switches'] += S
            totals['switch_den'] += max(A - 1.0, 0.0)
            totals['ch_role_time'] += sum(ch_role_lifetimes)
            totals['ch_role_runs'] += len(ch_role_lifetimes)
            totals['ch_effective_time'] += sum(ch_effective_lifetimes)
            totals['ch_effective_runs'] += len(ch_effective_lifetimes)
            totals['cm_episode_time'] += sum(cm_episode_durations)
            totals['cm_episode_count'] += len(cm_episode_durations)
            totals['empty_ch_time'] += empty_ch_time

            if A_cm > 0 and presence > 0:
                gamma = len(cm_episode_durations)
                totals['vcsm_sum'] += A_cm / (gamma * presence)
                totals['vcsm_member_count'] += 1
                contribution_cm = (A_cm / presence) * R_cm * Q_cm
                totals['vcsm_cm_sum'] += contribution_cm
                totals['R_cm_sum'] += R_cm
                totals['R_cm_weighted'] += A_cm * R_cm
                totals['A_cm_conditional'] += A_cm
                totals['P_cm_conditional'] += presence
                totals['cm_switches'] += S_cm
                totals['cm_switch_den'] += max(A_cm - 1.0, 0.0)
                cm_per_vehicle[vid] = {
                    'vid': vid,
                    'P': presence,
                    'A_CM': A_cm,
                    'eta_CM': A_cm / presence,
                    'R_CM': R_cm,
                    'Q_CM': Q_cm,
                    'S_CM': float(S_cm),
                    'switch_den_CM': max(A_cm - 1.0, 0.0),
                    'contribution': contribution_cm,
                    'num_distinct_CH_CM': len(cm_residence),
                }

        P = totals['presence']
        A = totals['A']
        n_cm = totals['vcsm_member_count']
        ch_time = totals['ch_role_time']
        vcsm_r = {
            'vcsm_r': totals['contribution_r'] / P if P else 0.0,
            'AR': A / P if P else 0.0,
            'average_R': totals['weighted_R'] / A if A else 0.0,
            'CCR': (totals['switches'] / totals['switch_den']
                    if totals['switch_den'] else 0.0),
            'avg_L_CM': (totals['cm_episode_time'] / totals['cm_episode_count']
                         if totals['cm_episode_count'] else 0.0),
            'ECHR': totals['empty_ch_time'] / ch_time if ch_time else 0.0,
            'avg_ch_lifetime_role': (ch_time / totals['ch_role_runs']
                                     if totals['ch_role_runs'] else 0.0),
            'avg_ch_lifetime_effective': (
                totals['ch_effective_time'] / totals['ch_effective_runs']
                if totals['ch_effective_runs'] else 0.0
            ),
            'num_ch_role_episodes': totals['ch_role_runs'],
            'num_ch_effective_episodes': totals['ch_effective_runs'],
            'num_cm_episodes': totals['cm_episode_count'],
            'total_ch_role_time': ch_time,
            'total_ch_effective_time': totals['ch_effective_time'],
            'total_ch_role_time_raw': ch_time,
            'total_empty_ch_time': totals['empty_ch_time'],
            'total_presence': P,
            'total_clustered_time': A,
            'total_switches': totals['switches'],
        }

        vcsm_cm = {
            'vcsm_cm': totals['vcsm_cm_sum'] / n_cm if n_cm else 0.0,
            'average_R_CM': totals['R_cm_sum'] / n_cm if n_cm else 0.0,
            'average_R_CM_weighted': (
                totals['R_cm_weighted'] / totals['A_cm'] if totals['A_cm'] else 0.0
            ),
            'AR_CM_conditional': (
                totals['A_cm_conditional'] / totals['P_cm_conditional']
                if totals['P_cm_conditional'] else 0.0
            ),
            'CCR_CM': (totals['cm_switches'] / totals['cm_switch_den']
                        if totals['cm_switch_den'] else 0.0),
            'num_cm_vehicles': n_cm,
            'total_A_CM': totals['A_cm_conditional'],
            'total_P_CM_vehicles': totals['P_cm_conditional'],
            'total_switches_CM': totals['cm_switches'],
        }
        return {
            'vcsm': totals['vcsm_sum'] / n_cm if n_cm else 0.0,
            'vcsm_r': vcsm_r,
            'vcsm_cm': vcsm_cm,
            'per_vehicle': cm_per_vehicle,
        }

    def vcsm(self, configs):
        """MMZCA VCSM: average root-cluster CM residence per CM segment/presence."""
        return self._mmzca_stability_summary()['vcsm']

    def vcsm_r(self, configs):
        """MMZCA VCSM_R, including effective CH and CM occupancy diagnostics."""
        return self._mmzca_stability_summary()['vcsm_r']

    def vcsm_cm(self, configs, return_per_vehicle=False):
        """MMZCA CM-conditioned root-association residence stability."""
        result = self._mmzca_stability_summary()
        output = result['vcsm_cm']
        if return_per_vehicle:
            output = dict(output, per_vehicle=result['per_vehicle'])
        return output

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
        post_ids = set(self.veh_table.ids())
        pre_ids = set(pre_states.keys())

        # only vehicles that existed before and still exist after count for transition comparison
        common_ids = {
            veh_id for veh_id in pre_ids.intersection(post_ids)
            if self.veh_table.values(veh_id).get('in_area', False) is True
        }

        # A node may detach and reattach to the same root during one tick.
        # The endpoint state would be identical, but a real transition occurred.
        changed_ids = {
            event['vehicle'] for event in self._tvct_events_this_tick
        }.intersection(common_ids)

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
            changed_ids.add(veh_id)

        tvct_t = len(changed_ids)

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