"""
Stable vehicle-only MetoidS-style clustering for SUMO FCD traces.

This file is designed to be a drop-in replacement for data_cluster.py in the
current framework:
    from data_cluster import DataTable

It keeps the existing input pipeline intact:
    constants.py/config.py load SUMO XML files
    main.py creates Configs, ZoneID objects, and passes them to DataTable

Implemented clustering part only:
    - vehicle-only clustering; buses/RSUs are ignored
    - transmission-range connected components
    - MetoidS-inspired medoid/CH selection using cosine orientation distance
      and geographic compactness
    - dynamic CH preservation to avoid full re-clustering every tick
    - metrics: avg CHs/stand-alones via main.py, VCSM, TVCT(t), avg TVCT
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

import hash
import utils.util as util


class DataTable:
    """Vehicle-only clustering table compatible with the existing main.py."""

    def __init__(self, config, zones):
        # Compatibility only.  This algorithm ignores buses/RSUs.
        self.bus_table = hash.HashTable(1)
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
            st["depart_time"] = None
            st["in_area"] = util.presence(self.understudied_area, veh)

            self.zone_vehicles.setdefault(macro_id, set()).add(veh_id)
            self._set_sa(veh_id, update_record=False)
            self.net_graph.add_node(veh_id, pos=(float(veh.getAttribute("y")), float(veh.getAttribute("x"))))

    def update(self, config) -> None:
        """Advance one SUMO timestep and update vehicle states only."""
        self.time += 1
        current_ids: Set[str] = set()

        if self.time >= len(config.sumo_trace.documentElement.getElementsByTagName("timestep")):
            return

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
                st["depart_time"] = None
            else:
                st = self.veh_table.values(veh_id)
                old_macro = st.get("macro_zone")
                if old_macro in self.zone_vehicles:
                    self.zone_vehicles[old_macro].discard(veh_id)

                st["prev_micro_zone"] = st.get("micro_zone")
                st["prev_meso_zone"] = st.get("meso_zone")
                st["prev_macro_zone"] = st.get("macro_zone")

            # Keep the same field names as util.initiate_new_veh.
            st["long"] = float(veh.getAttribute("x"))
            st["lat"] = float(veh.getAttribute("y"))
            st["angle"] = float(veh.getAttribute("angle"))
            st["speed"] = float(veh.getAttribute("speed")) + 0.01
            st["pos"] = float(veh.getAttribute("pos"))
            st["micro_zone"] = micro_id
            st["meso_zone"] = meso_id
            st["macro_zone"] = macro_id
            st["neighbor_zones"] = self.macro_zones.neighbor_zones(macro_id) if macro_id is not None else set()
            st["in_area"] = util.presence(self.understudied_area, veh)

            self.zone_vehicles.setdefault(macro_id, set()).add(veh_id)
            self.net_graph.add_node(veh_id, pos=(st["lat"], st["long"]))

        for veh_id in list(self.veh_table.ids() - current_ids):
            self._remove_vehicle_that_left(veh_id)

    def _remove_vehicle_that_left(self, veh_id: str) -> None:
        st = self.veh_table.values(veh_id)
        if st is None:
            return

        macro = st.get("macro_zone")
        if macro in self.zone_vehicles:
            self.zone_vehicles[macro].discard(veh_id)
        if macro in self.zone_ch:
            self.zone_ch[macro].discard(veh_id)
        if macro in self.zone_stand_alone:
            self.zone_stand_alone[macro].discard(veh_id)

        self.stand_alone.discard(veh_id)
        self.all_chs.discard(veh_id)

        # Remove it from any active CH membership list.
        for ch_id in list(self.all_chs):
            if ch_id in self.veh_table.ids():
                self.veh_table.values(ch_id).get("cluster_members", set()).discard(veh_id)

        # If a CH leaves, its members become SA until the next clustering call.
        if st.get("cluster_head") is True:
            for mem in list(st.get("cluster_members", set())):
                if mem in self.veh_table.ids():
                    self._set_sa(mem, update_record=False)

        st["depart_time"] = self.time - 1
        self.left_veh[veh_id] = st
        self.veh_table.remove(veh_id)
        if veh_id in self.net_graph:
            self.net_graph.remove_node(veh_id)

    # ------------------------------------------------------------------
    # Stable MetoidS-style clustering
    # ------------------------------------------------------------------
    def update_cluster(self, veh_ids: Iterable[str], config) -> None:
        """
        Cluster active vehicles for the current timestep.

        Key design choice: the paper describes K-Medoids-style clustering but
        does not define a dynamic cluster-maintenance mechanism.  Running a
        fresh K-Medoids solution every tick causes excessive CH changes and
        destroys VCSM.  This implementation therefore uses MetoidS-inspired
        medoid selection inside each transmission-range component while strongly
        preferring previous roots when they remain feasible.
        """
        active_ids = [
            vid for vid in veh_ids
            if vid in self.veh_table.ids()
            and self.veh_table.values(vid).get("in_area", True)
        ]

        # Previous root memory before resetting per-tick sets.
        prev_root = {vid: self.veh_table.values(vid).get("root_ch") for vid in active_ids}
        prev_is_ch = {vid: bool(self.veh_table.values(vid).get("cluster_head")) for vid in active_ids}

        self._reset_current_cluster_sets(active_ids)

        if not active_ids:
            return

        neighbor_graph = self._build_neighbor_graph(active_ids, config)
        self.net_graph = neighbor_graph.copy()

        target_size = int(getattr(config, "metoids_target_cluster_size", 8))
        target_size = max(target_size, 2)

        for component in nx.connected_components(neighbor_graph):
            comp = list(component)
            if len(comp) == 1:
                self._set_sa(comp[0], update_record=False)
                continue

            k = max(1, math.ceil(len(comp) / target_size))

            # Prefer old CHs that are still inside this component.  If there are
            # too many old CHs, keep the best medoid-like old CHs only.
            old_chs = [vid for vid in comp if prev_is_ch.get(vid, False)]
            if len(old_chs) >= k:
                medoids = self._select_best_medoids_from_candidates(comp, old_chs, k, config)
            else:
                medoids = list(old_chs)
                missing = k - len(medoids)
                medoids.extend(self._choose_additional_medoids(comp, medoids, missing, config))

            # Fallback: no old CH and additional selection failed.
            if not medoids:
                medoid = self._choose_metoids_medoid(comp, config)
                medoids = [medoid] if medoid is not None else []

            if not medoids:
                for vid in comp:
                    self._set_sa(vid, update_record=False)
                continue

            for ch_id in medoids:
                self._set_ch(ch_id, config, update_record=False)

            for vid in comp:
                if vid in medoids:
                    continue

                feasible_medoids = [ch for ch in medoids if self._is_feasible_member(vid, ch, config)]
                if not feasible_medoids:
                    self._set_sa(vid, update_record=False)
                    continue

                # Strong root persistence: keep previous root if it remains one
                # of the selected CHs and is feasible.  This directly improves
                # VCSM and reduces TVCT.
                old_root = prev_root.get(vid)
                if old_root in feasible_medoids:
                    chosen = old_root
                else:
                    chosen = min(feasible_medoids, key=lambda ch: self._member_cost(vid, ch, config))

                self._attach_member(vid, chosen, config, update_record=False)

        # Record one state segment per active vehicle per tick.  This is required
        # for VCSM; without this, VCSM remains zero even when clustering exists.
        for vid in active_ids:
            if vid in self.veh_table.ids():
                self._update_cluster_record(vid)

    def stand_alones_cluster(self, configs) -> None:
        """Compatibility hook. Clustering is completed in update_cluster()."""
        return

    def _reset_current_cluster_sets(self, active_ids: List[str]) -> None:
        self.stand_alone.clear()
        self.all_chs.clear()
        for z in self.zone_ch:
            self.zone_ch[z].clear()
        for z in self.zone_stand_alone:
            self.zone_stand_alone[z].clear()

        for vid in active_ids:
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
            ur = float(self.veh_table.values(u).get("trans_range", getattr(config, "veh_trans_range", 300)))
            for v in active_ids[i + 1:]:
                vr = float(self.veh_table.values(v).get("trans_range", getattr(config, "veh_trans_range", 300)))
                if self._distance_m(u, v) <= min(ur, vr):
                    g.add_edge(u, v)
        return g

    def _choose_metoids_medoid(self, comp: List[str], config) -> Optional[str]:
        if not comp:
            return None
        if len(comp) == 1:
            return comp[0]

        # Low-variance candidate group: remove angular outliers.
        mean_codis = {}
        for u in comp:
            vals = [self._cosine_distance(u, v) for v in comp if v != u]
            mean_codis[u] = float(np.mean(vals)) if vals else 0.0

        vals = np.array(list(mean_codis.values()), dtype=float)
        threshold = float(np.mean(vals) + np.std(vals)) if len(vals) > 1 else float("inf")
        candidates = [u for u in comp if mean_codis[u] <= threshold]
        if not candidates:
            candidates = comp

        return min(candidates, key=lambda c: self._component_medoid_score(c, comp, config))

    # Backward-compatible misspelled alias used in earlier versions.
    def _choose_metiods_medoid(self, comp: List[str], config) -> Optional[str]:
        return self._choose_metoids_medoid(comp, config)

    def _select_best_medoids_from_candidates(self, comp: List[str], candidates: List[str], k: int, config) -> List[str]:
        ranked = sorted(candidates, key=lambda c: self._component_medoid_score(c, comp, config))
        return ranked[:k]

    def _choose_additional_medoids(self, comp: List[str], existing: List[str], count: int, config) -> List[str]:
        chosen = list(existing)
        added: List[str] = []
        if count <= 0:
            return added

        if not chosen:
            first = self._choose_metoids_medoid(comp, config)
            if first is None:
                return added
            chosen.append(first)
            added.append(first)
            count -= 1

        while count > 0:
            best_vid = None
            best_score = -1.0
            for vid in comp:
                if vid in chosen:
                    continue
                # Farthest-first from existing medoids, using the same hybrid
                # cost. This prevents all medoids from being selected in one
                # local pocket of the component.
                score = min(self._member_cost(vid, m, config) for m in chosen)
                if score > best_score:
                    best_score = score
                    best_vid = vid
            if best_vid is None:
                break
            chosen.append(best_vid)
            added.append(best_vid)
            count -= 1
        return added

    def _component_medoid_score(self, cand: str, comp: List[str], config) -> float:
        if len(comp) <= 1:
            return 0.0
        return sum(self._member_cost(cand, other, config) for other in comp if other != cand)

    def _member_cost(self, veh_id: str, ch_id: str, config) -> float:
        # In the paper, cosine distance is the primary clustering criterion.
        # A geographic term is kept to avoid selecting directionally similar
        # but spatially peripheral CHs.
        w_cos = float(getattr(config, "metoids_w_cosine", 0.65))
        w_dist = 1.0 - w_cos
        tr = max(float(getattr(config, "veh_trans_range", 300)), 1.0)
        cosine_part = self._cosine_distance(veh_id, ch_id) / 2.0  # normalize [0,2] to [0,1]
        dist_part = min(self._distance_m(veh_id, ch_id) / tr, 1.0)
        return w_cos * cosine_part + w_dist * dist_part

    def _is_feasible_member(self, vid: str, ch_id: str, config) -> bool:
        # Feasibility should be communication feasibility only.  Do not reject
        # members by cosine distance; that was the cause of too many SAs.
        vr = float(self.veh_table.values(vid).get("trans_range", getattr(config, "veh_trans_range", 300)))
        cr = float(self.veh_table.values(ch_id).get("trans_range", getattr(config, "veh_trans_range", 300)))
        return self._distance_m(vid, ch_id) <= min(vr, cr)

    def _set_ch(self, vid: str, config, update_record: bool = False) -> None:
        st = self.veh_table.values(vid)
        st["cluster_head"] = True
        st["primary_ch"] = None
        st["secondary_ch"] = None
        st["root_ch"] = vid
        st["parent_node"] = None
        st["hop_count"] = 0
        st["start_ch_zone"] = st.get("macro_zone")
        st["counter"] = getattr(config, "counter", st.get("counter", 0))
        st["cluster_members"] = st.get("cluster_members", set())

        self.all_chs.add(vid)
        macro = st.get("macro_zone")
        if macro in self.zone_ch:
            self.zone_ch[macro].add(vid)
        self.stand_alone.discard(vid)
        if macro in self.zone_stand_alone:
            self.zone_stand_alone[macro].discard(vid)

        if update_record:
            self._update_cluster_record(vid)

    def _attach_member(self, vid: str, ch_id: str, config, update_record: bool = False) -> None:
        st = self.veh_table.values(vid)
        ch = self.veh_table.values(ch_id)

        st["cluster_head"] = False
        st["primary_ch"] = ch_id
        st["secondary_ch"] = None
        st["root_ch"] = ch_id
        st["parent_node"] = ch_id
        st["hop_count"] = 1
        st["counter"] = getattr(config, "counter", st.get("counter", 0))

        ch.setdefault("cluster_members", set()).add(vid)
        self.stand_alone.discard(vid)
        macro = st.get("macro_zone")
        if macro in self.zone_stand_alone:
            self.zone_stand_alone[macro].discard(vid)

        if update_record:
            self._update_cluster_record(vid)

    def _set_sa(self, vid: str, update_record: bool = False) -> None:
        st = self.veh_table.values(vid)
        st["cluster_head"] = False
        st["primary_ch"] = None
        st["secondary_ch"] = None
        st["root_ch"] = None
        st["parent_node"] = None
        st["hop_count"] = None

        self.stand_alone.add(vid)
        macro = st.get("macro_zone")
        if macro in self.zone_stand_alone:
            self.zone_stand_alone[macro].add(vid)

        if update_record:
            self._update_cluster_record(vid)

    def _reset_to_sa(self, vid: str) -> None:
        if vid in self.veh_table.ids():
            self._set_sa(vid, update_record=False)

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
        """Distance in meters, auto-detecting SUMO coordinate mode.

        If x/y look like longitude/latitude, use haversine.  If they look like
        projected SUMO meter coordinates, use Euclidean distance.
        """
        a = self.veh_table.values(u)
        b = self.veh_table.values(v)
        lat1, lon1 = float(a["lat"]), float(a["long"])
        lat2, lon2 = float(b["lat"]), float(b["long"])

        looks_geo = (
            -90.0 <= lat1 <= 90.0 and -90.0 <= lat2 <= 90.0 and
            -180.0 <= lon1 <= 180.0 and -180.0 <= lon2 <= 180.0
        )
        if looks_geo:
            return self._haversine_m(lat1, lon1, lat2, lon2)

        dx = lon1 - lon2
        dy = lat1 - lat2
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
    # Cluster record and metrics
    # ------------------------------------------------------------------
    def _current_record_state(self, vid: str) -> Tuple[Optional[str], bool, Optional[int]]:
        st = self.veh_table.values(vid)
        if st.get("cluster_head") is True:
            return vid, True, 1
        if st.get("primary_ch") is not None:
            return st.get("primary_ch"), False, 1
        return None, False, None

    def _update_cluster_record(self, vid: str) -> None:
        st = self.veh_table.values(vid)
        rec = st["cluster_record"]
        key, is_ch, timer = self._current_record_state(vid)

        tail = rec.tail
        tail_key = getattr(tail, "key", None)
        tail_val = getattr(tail, "value", {})
        if not isinstance(tail_val, dict):
            tail_val = {}
        tail_is_ch = bool(tail_val.get("is_ch", False))

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
        return {
            vid: self._tvct_state(vid)
            for vid in self.veh_table.ids()
            if self.veh_table.values(vid).get("in_area", True)
        }

    def _tvct_state(self, vid: str) -> Tuple[str, Optional[str]]:
        st = self.veh_table.values(vid)
        if st.get("cluster_head") is True:
            return "CH", vid
        if st.get("primary_ch") is not None:
            return "MEM", st.get("primary_ch")
        return "SA", None

    def update_tvct(self, pre_states: Dict[str, Tuple[str, Optional[str]]]) -> Tuple[int, float]:
        tvct_t = 0
        for vid in set(pre_states.keys()).intersection(self.veh_table.ids()):
            if not self.veh_table.values(vid).get("in_area", True):
                continue
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
                    if t is not None and t > 0:
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
        raise NotImplementedError("Map visualization is omitted from this clustering-only implementation.")

    def save_map_img(self, zoom, name):
        raise NotImplementedError("Map-image saving is omitted from this clustering-only implementation.")

    def print_table(self):
        self.veh_table.print_hash_table()
