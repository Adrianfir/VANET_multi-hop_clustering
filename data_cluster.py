"""
MetoidS-style vehicle-only clustering for SUMO FCD traces.

This implementation keeps the existing SUMO loading pipeline intact: the XML
objects are still supplied through constants.py/config.py.  It intentionally
removes bus/RSU logic and implements only the clustering part of MetoidS:

  - orientation-aware K-Medoids-style clustering
  - cosine distance between vehicle movement vectors
  - medoid/cluster-head selection from low-variance candidate vehicles
  - vehicle-only CH/CM/SA states
  - metrics: VCSM, TVCT(t), running average TVCT, number of CHs and SAs

Expected external framework objects:
  - hash.HashTable
  - linked_list.LinkedList through util.initiate_new_veh(...)
  - utils.util.initiate_new_veh(...)
  - ZoneID objects for micro/meso/macro zone detection
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import math

import hash
import utils.util as util


class DataTable:
    """Vehicle-only MetoidS clustering table."""

    def __init__(self, config, zones):
        self.bus_table = hash.HashTable(1)  # kept only for compatibility with old main.py prints
        self.veh_table = hash.HashTable(config.n_cars * 100)

        self.micro_zones = zones["micro"]
        self.meso_zones = zones["meso"]
        self.macro_zones = zones["macro"]
        self.understudied_area = self.macro_zones.understudied_area()

        zone_ids = list(self.macro_zones.zone_hash.ids())
        self.zone_vehicles: Dict[object, Set[str]] = {z: set() for z in zone_ids}
        self.zone_ch: Dict[object, Set[str]] = {z: set() for z in zone_ids}
        self.zone_stand_alone: Dict[object, Set[str]] = {z: set() for z in zone_ids}

        self.stand_alone: Set[str] = set()
        self.all_chs: Set[str] = set()
        self.left_veh: Dict[str, dict] = {}

        self.time = config.start_time
        self.net_graph = nx.Graph()

        self.tvct_history: List[int] = []
        self.avg_tvct_history: List[float] = []
        self.tvct_cumulative = 0.0

        self._load_initial_timestep(config)

    # ------------------------------------------------------------------
    # SUMO loading/update
    # ------------------------------------------------------------------
    def _vehicles_at_time(self, config, t: int):
        return config.sumo_trace.documentElement.getElementsByTagName("timestep")[t].childNodes[1::2]

    def _zone_ids_for_xml_vehicle(self, veh) -> Tuple[object, object, object]:
        y = float(veh.getAttribute("y"))
        x = float(veh.getAttribute("x"))
        return (
            self.micro_zones.det_zone(y, x),
            self.meso_zones.det_zone(y, x),
            self.macro_zones.det_zone(y, x),
        )

    def _load_initial_timestep(self, config) -> None:
        for veh in self._vehicles_at_time(config, self.time):
            veh_id = veh.getAttribute("id")
            if "bus" in veh_id.lower():
                continue
            micro_id, meso_id, macro_id = self._zone_ids_for_xml_vehicle(veh)
            self.veh_table.set_item(
                veh_id,
                util.initiate_new_veh(
                    veh,
                    self.macro_zones,
                    micro_id,
                    meso_id,
                    macro_id,
                    config,
                    self.understudied_area,
                ),
            )
            st = self.veh_table.values(veh_id)
            st["arrive_time"] = self.time
            self.zone_vehicles[macro_id].add(veh_id)
            self.stand_alone.add(veh_id)
            self.zone_stand_alone[macro_id].add(veh_id)
            self.net_graph.add_node(veh_id, pos=(float(veh.getAttribute("y")), float(veh.getAttribute("x"))))

    def update(self, config) -> None:
        """Advance one SUMO timestep and update vehicle states only."""
        self.time += 1
        current_ids: Set[str] = set()
        self.net_graph.remove_edges_from(list(self.net_graph.edges()))

        for veh in self._vehicles_at_time(config, self.time):
            veh_id = veh.getAttribute("id")
            if "bus" in veh_id.lower():
                continue
            current_ids.add(veh_id)
            micro_id, meso_id, macro_id = self._zone_ids_for_xml_vehicle(veh)

            if veh_id not in self.veh_table.ids():
                self.veh_table.set_item(
                    veh_id,
                    util.initiate_new_veh(
                        veh,
                        self.macro_zones,
                        micro_id,
                        meso_id,
                        macro_id,
                        config,
                        self.understudied_area,
                    ),
                )
                st = self.veh_table.values(veh_id)
                st["arrive_time"] = self.time
                self.stand_alone.add(veh_id)
                self.zone_stand_alone[macro_id].add(veh_id)
            else:
                st = self.veh_table.values(veh_id)
                old_macro = st["macro_zone"]
                if veh_id in self.zone_vehicles.get(old_macro, set()):
                    self.zone_vehicles[old_macro].remove(veh_id)
                if st["cluster_head"] and veh_id in self.zone_ch.get(old_macro, set()):
                    self.zone_ch[old_macro].remove(veh_id)
                if (not st["cluster_head"]) and st["primary_ch"] is None:
                    self.zone_stand_alone.get(old_macro, set()).discard(veh_id)

                st["prev_micro_zone"] = st["micro_zone"]
                st["prev_meso_zone"] = st["meso_zone"]
                st["prev_macro_zone"] = st["macro_zone"]
                st["long"] = float(veh.getAttribute("x"))
                st["lat"] = float(veh.getAttribute("y"))
                st["angle"] = float(veh.getAttribute("angle"))
                st["speed"] = float(veh.getAttribute("speed")) + 0.01
                st["pos"] = float(veh.getAttribute("pos"))
                st["micro_zone"] = micro_id
                st["meso_zone"] = meso_id
                st["macro_zone"] = macro_id
                st["neighbor_zones"] = self.macro_zones.neighbor_zones(macro_id)
                st["in_area"] = util.presence(self.understudied_area, veh)

            self.zone_vehicles[macro_id].add(veh_id)
            self.net_graph.add_node(veh_id, pos=(float(veh.getAttribute("y")), float(veh.getAttribute("x"))))

        for veh_id in list(self.veh_table.ids() - current_ids):
            self._remove_vehicle_that_left(veh_id)

    def _remove_vehicle_that_left(self, veh_id: str) -> None:
        st = self.veh_table.values(veh_id)
        if st is None:
            return
        macro = st["macro_zone"]
        self.zone_vehicles.get(macro, set()).discard(veh_id)
        self.zone_ch.get(macro, set()).discard(veh_id)
        self.zone_stand_alone.get(macro, set()).discard(veh_id)
        self.stand_alone.discard(veh_id)
        self.all_chs.discard(veh_id)

        if st.get("primary_ch") in self.veh_table.ids():
            self.veh_table.values(st["primary_ch"])["cluster_members"].discard(veh_id)

        if st.get("cluster_head") is True:
            for mem in list(st.get("cluster_members", set())):
                if mem in self.veh_table.ids():
                    self._reset_to_sa(mem)

        st["depart_time"] = self.time - 1
        self.left_veh[veh_id] = st
        self.veh_table.remove(veh_id)
        if veh_id in self.net_graph:
            self.net_graph.remove_node(veh_id)

    # ------------------------------------------------------------------
    # MetoidS clustering
    # ------------------------------------------------------------------
    def update_cluster(self, veh_ids: Iterable[str], config) -> None:
        """
        Vehicle-only MetoidS clustering for the current timestep.
        """

        self._clear_current_clusters()

        active_ids = [
            vid for vid in veh_ids
            if vid in self.veh_table.ids()
               and self.veh_table.values(vid).get("in_area", True)
        ]

        if not active_ids:
            return

        neighbor_graph = self._build_neighbor_graph(active_ids, config)
        self.net_graph = neighbor_graph.copy()

        for component in nx.connected_components(neighbor_graph):
            comp = list(component)

            if len(comp) < 2:
                self._set_sa(comp[0])
                continue

            target_size = getattr(config, "metoids_target_cluster_size", 8)
            k = max(1, math.ceil(len(comp) / target_size))

            medoids = self._choose_k_medoids(comp, k, config)

            if len(medoids) == 0:
                for vid in comp:
                    self._set_sa(vid)
                continue

            for medoid in medoids:
                self._set_ch(medoid, config)

            for vid in comp:
                if vid in medoids:
                    continue

                feasible_medoids = [
                    ch for ch in medoids
                    if self._is_feasible_member(vid, ch, config)
                ]

                if len(feasible_medoids) == 0:
                    self._set_sa(vid)
                    continue

                best_ch = min(
                    feasible_medoids,
                    key=lambda ch: self._member_cost(vid, ch, config)
                )

                self._attach_member(vid, best_ch, config)

        # important for VCSM
        for vid in active_ids:
            if vid in self.veh_table.ids():
                self._update_cluster_record(vid)

    def stand_alones_cluster(self, configs) -> None:
        """Compatibility hook. MetoidS clustering is completed in update_cluster()."""
        return

    def _clear_current_clusters(self) -> None:
        self.stand_alone.clear()
        self.all_chs.clear()
        for z in self.zone_ch:
            self.zone_ch[z].clear()
        for z in self.zone_stand_alone:
            self.zone_stand_alone[z].clear()

        for vid in self.veh_table.ids():
            st = self.veh_table.values(vid)
            st["cluster_head"] = False
            st["primary_ch"] = None
            st["secondary_ch"] = None
            st["root_ch"] = None
            st["parent_node"] = None
            st["hop_count"] = None
            st["cluster_members"] = set()
            st["sub_cluster_members"] = set()
            st["other_chs"] = set()
            st["other_vehs"] = set()
            st["gates"] = dict()
            st["gate_chs"] = set()

    def _build_neighbor_graph(self, active_ids: List[str], config) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(active_ids)
        for i, u in enumerate(active_ids):
            for v in active_ids[i + 1:]:
                if self._distance_m(u, v) <= min(self.veh_table.values(u)["trans_range"], self.veh_table.values(v)["trans_range"]):
                    g.add_edge(u, v)
        return g

    def _member_cost(self, veh_id, ch_id, config):
        cosine_part = self._cosine_distance(veh_id, ch_id)

        dist_part = min(
            self._distance_m(veh_id, ch_id) / max(config.veh_trans_range, 1),
            1.0
        )

        return 0.75 * cosine_part + 0.25 * dist_part

    def _choose_metiods_medoid(self, comp: List[str], config) -> Optional[str]:
        """Select CH/medoid using MetoidS cosine distance and variance filtering."""
        if not comp:
            return None
        if len(comp) == 1:
            return comp[0]

        # Mean pairwise cosine distance per vehicle.
        mean_codis = {}
        for u in comp:
            vals = [self._cosine_distance(u, v) for v in comp if v != u]
            mean_codis[u] = float(np.mean(vals)) if vals else 0.0

        # Eq. (3)-(4)-style low-variance medoid group: remove angular outliers.
        all_vals = np.array(list(mean_codis.values()), dtype=float)
        centroid = float(np.mean(all_vals))
        sigma = float(np.std(all_vals, ddof=1)) if len(all_vals) > 1 else 0.0
        delta = centroid + sigma
        medoid_group = [u for u in comp if mean_codis[u] <= delta]
        if not medoid_group:
            medoid_group = comp

        # Pick the node with minimum total hybrid distance to all vehicles.
        # Cosine orientation dominates; normalized geographic distance prevents
        # selecting a directionally-similar but geographically peripheral node.
        best_vid = None
        best_score = float("inf")
        tr = max(float(getattr(config, "veh_trans_range", 300)), 1.0)
        for cand in medoid_group:
            score = 0.0
            for other in comp:
                if other == cand:
                    continue
                codis = self._cosine_distance(cand, other)
                ndist = min(self._distance_m(cand, other) / tr, 1.0)
                score += 0.75 * codis + 0.25 * ndist
            if score < best_score:
                best_score = score
                best_vid = cand
        return best_vid

    def _choose_k_medoids(self, comp, k, config):
        if len(comp) == 0:
            return []

        if k >= len(comp):
            return list(comp)

        medoids = []

        first = self._choose_metiods_medoid(comp, config)
        if first is None:
            return []

        medoids.append(first)

        while len(medoids) < k:
            best_vid = None
            best_score = -1.0

            for vid in comp:
                if vid in medoids:
                    continue

                nearest_medoid_distance = min(
                    0.75 * self._cosine_distance(vid, m)
                    + 0.25 * min(self._distance_m(vid, m) / max(config.veh_trans_range, 1), 1)
                    for m in medoids
                )

                if nearest_medoid_distance > best_score:
                    best_score = nearest_medoid_distance
                    best_vid = vid

            if best_vid is None:
                break

            medoids.append(best_vid)

        return medoids

    def _is_feasible_member(self, vid: str, ch_id: str, config) -> bool:
        return self._distance_m(vid, ch_id) <= min(
            self.veh_table.values(vid)["trans_range"],
            self.veh_table.values(ch_id)["trans_range"]
        )

    def _set_ch(self, vid: str, config) -> None:
        st = self.veh_table.values(vid)
        st["cluster_head"] = True
        st["primary_ch"] = None
        st["secondary_ch"] = None
        st["root_ch"] = vid
        st["parent_node"] = None
        st["hop_count"] = 0
        st["start_ch_zone"] = st["macro_zone"]
        st["counter"] = config.counter
        self.all_chs.add(vid)
        self.zone_ch[st["macro_zone"]].add(vid)
        self.stand_alone.discard(vid)
        self.zone_stand_alone[st["macro_zone"]].discard(vid)

    def _attach_member(self, vid: str, ch_id: str, config) -> None:
        st = self.veh_table.values(vid)
        ch = self.veh_table.values(ch_id)
        st["cluster_head"] = False
        st["primary_ch"] = ch_id
        st["secondary_ch"] = None
        st["root_ch"] = ch_id
        st["parent_node"] = ch_id
        st["hop_count"] = 1
        st["counter"] = config.counter
        ch["cluster_members"].add(vid)
        self.stand_alone.discard(vid)
        self.zone_stand_alone[st["macro_zone"]].discard(vid)

    def _set_sa(self, vid: str) -> None:
        st = self.veh_table.values(vid)
        st["cluster_head"] = False
        st["primary_ch"] = None
        st["secondary_ch"] = None
        st["root_ch"] = None
        st["parent_node"] = None
        st["hop_count"] = None
        self.stand_alone.add(vid)
        self.zone_stand_alone[st["macro_zone"]].add(vid)

    def _reset_to_sa(self, vid: str) -> None:
        if vid in self.veh_table.ids():
            self._set_sa(vid)

    # ------------------------------------------------------------------
    # Distance/orientation helpers
    # ------------------------------------------------------------------
    def _heading_vector(self, vid: str) -> np.ndarray:
        st = self.veh_table.values(vid)
        theta = math.radians(float(st.get("angle", 0.0)))
        speed = max(float(st.get("speed", 0.01)), 0.01)
        return np.array([math.cos(theta) * speed, math.sin(theta) * speed], dtype=float)

    def _cosine_distance(self, u: str, v: str) -> float:
        vu = self._heading_vector(u)
        vv = self._heading_vector(v)
        denom = float(np.linalg.norm(vu) * np.linalg.norm(vv))
        if denom <= 1e-12:
            return 1.0
        cs = float(np.dot(vu, vv) / denom)
        cs = max(-1.0, min(1.0, cs))
        return 1.0 - cs

    def _distance_m(self, u: str, v: str) -> float:
        a = self.veh_table.values(u)
        b = self.veh_table.values(v)

        dx = float(a["long"]) - float(b["long"])
        dy = float(a["lat"]) - float(b["lat"])

        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371000.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # ------------------------------------------------------------------
    # Cluster-record and metrics
    # ------------------------------------------------------------------
    def _current_record_state(self, vid: str) -> Tuple[Optional[str], bool, Optional[int]]:
        st = self.veh_table.values(vid)
        if st["cluster_head"] is True:
            return vid, True, 1
        if st["primary_ch"] is not None:
            return st["primary_ch"], False, 1
        return None, False, None

    def _update_cluster_record(self, vid: str) -> None:
        st = self.veh_table.values(vid)
        rec = st["cluster_record"]
        key, is_ch, timer = self._current_record_state(vid)

        tail = rec.tail
        tail_key = getattr(tail, "key", None)
        tail_val = getattr(tail, "value", {})
        tail_is_ch = tail_val.get("is_ch", False) if isinstance(tail_val, dict) else False

        same_segment = (tail_key == key) and (tail_is_ch == is_ch)
        if same_segment:
            if timer is not None:
                if tail_val.get("timer") is None:
                    tail_val["timer"] = 1
                    tail_val["start_time"] = self.time
                else:
                    tail_val["timer"] += 1
            return

        rec.append(
            key,
            {
                "is_ch": is_ch,
                "secondary_ch": [None],
                "start_time": self.time if timer is not None else None,
                "ef": None,
                "timer": timer,
                "interrupt": [],
            },
        )

    def _snapshot_tvct_states(self) -> Dict[str, Tuple[str, Optional[str]]]:
        return {vid: self._tvct_state(vid) for vid in self.veh_table.ids() if self.veh_table.values(vid).get("in_area", True)}

    def _tvct_state(self, vid: str) -> Tuple[str, Optional[str]]:
        st = self.veh_table.values(vid)
        if st["cluster_head"] is True:
            return ("CH", vid)
        if st["primary_ch"] is not None:
            return ("MEM", st["primary_ch"])
        return ("SA", None)

    def update_tvct(self, pre_states: Dict[str, Tuple[str, Optional[str]]]) -> Tuple[int, float]:
        tvct_t = 0
        for vid in set(pre_states.keys()).intersection(self.veh_table.ids()):
            if pre_states[vid] != self._tvct_state(vid):
                tvct_t += 1
        self.tvct_history.append(tvct_t)
        self.tvct_cumulative += tvct_t
        self.avg_tvct_history.append(self.tvct_cumulative / len(self.tvct_history))
        return tvct_t, self.avg_tvct_history[-1]

    def vcsm(self, configs) -> float:
        def one_vehicle_vcsm(v: dict) -> Optional[float]:
            arrive = v.get("arrive_time", configs.start_time)
            depart = v.get("depart_time") if v.get("depart_time") is not None else self.time
            total_time = max(depart - arrive + 1, 1)

            clustered_time = 0.0
            gamma = 0
            temp = v["cluster_record"].head
            while temp:
                key = getattr(temp, "key", None)
                val = getattr(temp, "value", {})
                if key is not None and isinstance(val, dict) and val.get("is_ch", False) is False:
                    t = val.get("timer")
                    if t is not None:
                        clustered_time += t
                        gamma += 1
                temp = temp.next
            if gamma == 0:
                return None
            return clustered_time / (gamma * total_time)

        vals = []
        for vid in self.veh_table.ids():
            out = one_vehicle_vcsm(self.veh_table.values(vid))
            if out is not None:
                vals.append(out)
        for v in self.left_veh.values():
            out = one_vehicle_vcsm(v)
            if out is not None:
                vals.append(out)
        return float(np.mean(vals)) if vals else 0.0

    # ------------------------------------------------------------------
    # Optional compatibility/diagnostic methods
    # ------------------------------------------------------------------
    def connected_components(self):
        if len(self.net_graph.nodes()) == 0:
            return 0
        return nx.number_connected_components(self.net_graph)

    def update_other_connections(self):
        return

    def form_net_graph(self):
        return

    def show_graph(self, configs):
        raise NotImplementedError("Map visualization was intentionally omitted from this clustering-only version.")

    def save_map_img(self, zoom, name):
        raise NotImplementedError("Map-image saving was intentionally omitted from this clustering-only version.")

    def print_table(self):
        self.veh_table.print_hash_table()
