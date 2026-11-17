"""
This is the utils file including the small functions to add more functions for new_mmzca
"""
__author__: str = "Pouya 'Adrian' Firouzmakan"
__all__ = []



import numpy as np
import haversine as hs
import utils.util as util
from functools import lru_cache

def is_bus(node_id):
    # Match the SUMO ID convention used by DataTable.update().
    return isinstance(node_id, str) and 'bus' in node_id


def is_vehicle(node_id):
    return isinstance(node_id, str) and not is_bus(node_id)


def get_node_table(node_id, veh_table, bus_table):
    # The actual HashTable membership, not an ID naming convention, is
    # authoritative for data already present in the simulation.
    if bus_table.values(node_id) is not None:
        return bus_table
    return veh_table


def get_node_state(node_id, veh_table, bus_table):
    table = get_node_table(node_id, veh_table, bus_table)
    return table.values(node_id)

def sync_node_cluster_fields(node_id, veh_table, bus_table):
    """
    Keep legacy and generalized fields consistent without losing a committed
    MMZCA parent, depth, bottleneck, or kernel state.

    In particular, the legacy secondary_ch is the IMMEDIATE parent, not an
    assumption that the member is exactly two hops away from the root.
    """
    table = get_node_table(node_id, veh_table, bus_table)
    st = table.values(node_id)
    if st is None:
        return

    if table is bus_table or st.get('cluster_head') is True:
        st['root_ch'] = node_id
        st['parent_id'] = None
        st['parent_node'] = None
        st['hop_count'] = 0
        st['path_bottleneck'] = 0.0
        st['committed_link_score'] = None
        st['root_connection_valid'] = True
        st.setdefault('direct_children', set())
        return

    root_id = st.get('root_ch') or st.get('primary_ch')
    if root_id is None:
        reset_cluster_fields(node_id, veh_table, bus_table)
        return

    parent_id = (st.get('parent_node') or st.get('parent_id') or
                 st.get('secondary_ch') or root_id)
    st['root_ch'] = root_id
    st['primary_ch'] = root_id
    st['parent_node'] = parent_id
    st['parent_id'] = parent_id
    st['secondary_ch'] = None if parent_id == root_id else parent_id
    if st.get('hop_count') is None:
        parent = get_node_state(parent_id, veh_table, bus_table)
        parent_depth = parent.get('hop_count') if parent else None
        st['hop_count'] = (int(parent_depth) + 1
                           if parent_depth is not None else None)
    st.setdefault('direct_children', set())
    st.setdefault('root_connection_valid', False)
    st.setdefault('path_bottleneck', None)
    st.setdefault('committed_link_score', None)
    st['role'] = 'RN' if st['direct_children'] else 'CM'


def reset_cluster_fields(node_id, veh_table, bus_table):
    table = get_node_table(node_id, veh_table, bus_table)
    st = table.values(node_id)

    if st is not None and table is veh_table and st.get('cluster_head') is not True:
        st['primary_ch'] = None
        st['secondary_ch'] = None
        st['root_ch'] = None
        st['parent_id'] = None
        st['parent_node'] = None
        st['hop_count'] = None
        st['current_root_cost'] = None
        st['current_parent_cost'] = None
        st['path_bottleneck'] = None
        st['committed_link_score'] = None
        st['root_connection_valid'] = False
        st['kernel_descriptor'] = None
        st['direct_children'] = set()
        st['role'] = 'SA'

def is_rn(node_id, veh_table, bus_table):
    """
    RN = non-CH attached vehicle with at least one downstream sub-member.
    """
    if get_node_table(node_id, veh_table, bus_table) is bus_table:
        return False
    st = veh_table.values(node_id)
    return (st['cluster_head'] is False and
            st['primary_ch'] is not None and
            len(st.get('direct_children', set())) > 0)

def zone_index(zone_id):
    """
    Example:
      'mic_zone15' -> 15
      'mes_zone8'  -> 8
      'mac_zone103' -> 103
    """
    if zone_id is None:
        return None
    return int(zone_id.split('zone')[-1])

def zone_row_col(zone_id, zones_obj):
    """
    Return zero-based row, col in the corresponding zoning grid.
    """
    idx = zone_index(zone_id)
    if idx is None:
        return None, None
    row = idx // zones_obj.n_cols
    col = idx % zones_obj.n_cols
    return row, col

@lru_cache(maxsize=30000)
def zone_center(zone_id, zones_obj):
    """
    Return center (lat, long) of a zone.
    """
    if zone_id is None:
        return None, None
    z = zones_obj.zone_hash.values(zone_id)
    c_lat = 0.5 * (z['min_lat'] + z['max_lat'])
    c_long = 0.5 * (z['min_long'] + z['max_long'])
    return c_lat, c_long

def zone_grid_distance(zone_a, zone_b, zones_obj, mode='manhattan'):
    """
    Grid distance between two zones.
    mode:
      - 'manhattan'
      - 'chebyshev'
    """
    if zone_a is None or zone_b is None:
        return float('inf')

    ra, ca = zone_row_col(zone_a, zones_obj)
    rb, cb = zone_row_col(zone_b, zones_obj)

    if mode == 'chebyshev':
        return max(abs(ra - rb), abs(ca - cb))
    return abs(ra - rb) + abs(ca - cb)

def normalized_zone_distance(zone_a, zone_b, zones_obj, mode='manhattan'):
    d = zone_grid_distance(zone_a, zone_b, zones_obj, mode=mode)
    denom = max(zones_obj.n_rows + zones_obj.n_cols - 2, 1)
    return min(d / denom, 1.0)

def node_zone_fields(level):
    """
    level in {'micro', 'meso', 'macro'}
    """
    if level == 'micro':
        return 'micro_zone', 'prev_micro_zone'
    if level == 'meso':
        return 'meso_zone', 'prev_meso_zone'
    return 'macro_zone', 'prev_macro_zone'

def node_motion_vector_from_zones(node_id, veh_table, bus_table, zones_obj, level='macro'):
    """
    Construct a zone-induced motion vector using:
      previous zone center -> current position
    Returns a 2D vector in meters-like relative space.
    """
    st = get_node_state(node_id, veh_table, bus_table)
    zone_key, prev_zone_key = node_zone_fields(level)

    prev_zone = st.get(prev_zone_key)
    if prev_zone is None:
        return np.array([0.0, 0.0])

    prev_lat, prev_long = zone_center(prev_zone, zones_obj)
    cur_lat = st['lat']
    cur_long = st['long']

    # approximate directional components via haversine along lat/long axes
    dx = hs.haversine((prev_lat, prev_long), (prev_lat, cur_long), unit=hs.Unit.METERS)
    dy = hs.haversine((prev_lat, prev_long), (cur_lat, prev_long), unit=hs.Unit.METERS)

    if cur_long < prev_long:
        dx *= -1.0
    if cur_lat < prev_lat:
        dy *= -1.0

    return np.array([dx, dy], dtype=float)

def zotsim_angle(node_i, node_j, veh_table, bus_table, zones_obj, level='macro', eps=1e-9):
    """
    Returns ZOTSim as an angle in radians in [0, pi].
    """
    vi = node_motion_vector_from_zones(node_i, veh_table, bus_table, zones_obj, level=level)
    vj = node_motion_vector_from_zones(node_j, veh_table, bus_table, zones_obj, level=level)

    ni = np.linalg.norm(vi)
    nj = np.linalg.norm(vj)
    if ni < eps or nj < eps:
        return 0.5 * np.pi  # neutral fallback

    cos_sim = float(np.dot(vi, vj) / (ni * nj + eps))
    cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
    return float(np.arccos(cos_sim))


def zotsim_similarity(node_i, node_j, veh_table, bus_table, zones_obj, level='macro'):
    """
    Returns normalized similarity in [0,1] derived from ZOTSim angle:
        similarity = 1 - angle/pi
    """
    theta = zotsim_angle(node_i, node_j, veh_table, bus_table, zones_obj, level=level)
    return 1.0 - min(theta / np.pi, 1.0)


def zotsim_cost(node_i, node_j, veh_table, bus_table, zones_obj, level='macro'):
    """
    Returns normalized angular mismatch in [0,1]:
        cost = angle/pi
    """
    theta = zotsim_angle(node_i, node_j, veh_table, bus_table, zones_obj, level=level)
    return min(theta / np.pi, 1.0)

def get_root(node_id, veh_table, bus_table):
    st = get_node_state(node_id, veh_table, bus_table)

    if is_bus(node_id):
        return node_id
    if st['cluster_head'] is True:
        return node_id
    if st.get('root_ch') is not None:
        return st['root_ch']
    return st.get('primary_ch')

def get_hop_count(node_id, veh_table, bus_table):
    st = get_node_state(node_id, veh_table, bus_table)

    if is_bus(node_id):
        return 0
    if st['cluster_head'] is True:
        return 0
    return st.get('hop_count')

def candidate_root(candidate_id, veh_table, bus_table):
    return get_root(candidate_id, veh_table, bus_table)

def candidate_parent_hop(candidate_id, veh_table, bus_table):
    hc = get_hop_count(candidate_id, veh_table, bus_table)
    return float('inf') if hc is None else hc

def node_load(node_id, veh_table, bus_table):
    st = get_node_state(node_id, veh_table, bus_table)
    # Direct-child admission is bounded by direct capacity, not by a
    # transitive root membership list that may contain grandchildren.
    direct = st.get('direct_children')
    if direct is not None:
        return len(direct)
    return len(st.get('cluster_members', set()))

def normalized_node_load(node_id, veh_table, bus_table, config, is_root=False):
    load = node_load(node_id, veh_table, bus_table)
    cap = config.max_ch_members if is_root else config.max_rn_members
    cap = max(cap, 1)
    return min(load / cap, 1.0)

def direct_link_feasible(node_i, node_j, veh_table, bus_table):
    table_j = get_node_table(node_j, veh_table, bus_table)
    d = util.det_dist(node_i, veh_table, node_j, table_j)
    return d <= min(veh_table.values(node_i)['trans_range'],
                    table_j.values(node_j)['trans_range'])

def hierarchical_integrity_ok(veh_id, parent_id, root_id,
                              veh_table, bus_table,
                              micro_zones, meso_zones, macro_zones,
                              config):
    vi = veh_table.values(veh_id)
    vp = get_node_state(parent_id, veh_table, bus_table)
    vr = get_node_state(root_id, veh_table, bus_table)

    d_micro = zone_grid_distance(vi['micro_zone'], vp['micro_zone'], micro_zones, mode='chebyshev')
    d_meso = zone_grid_distance(vi['meso_zone'], vp['meso_zone'], meso_zones, mode='chebyshev')
    d_macro = zone_grid_distance(vi['macro_zone'], vr['macro_zone'], macro_zones, mode='chebyshev')

    return (d_micro <= config.micro_zone_max and
            d_meso <= config.meso_zone_max and
            d_macro <= config.macro_zone_max)

def hop_limit_ok(parent_id, veh_table, bus_table, config):
    """
    If veh attaches to parent_id, its hop_count becomes parent_hop + 1.
    """
    ph = get_hop_count(parent_id, veh_table, bus_table)
    if ph is None:
        return False
    return (ph + 1) <= config.h_max

def feasible_parent_for_root(veh_id, parent_id, root_id,
                             veh_table, bus_table,
                             micro_zones, meso_zones, macro_zones,
                             config):
    if not direct_link_feasible(veh_id, parent_id, veh_table, bus_table):
        return False

    if candidate_root(parent_id, veh_table, bus_table) != root_id:
        return False

    if not hierarchical_integrity_ok(veh_id, parent_id, root_id,
                                     veh_table, bus_table,
                                     micro_zones, meso_zones, macro_zones,
                                     config):
        return False

    if not hop_limit_ok(parent_id, veh_table, bus_table, config):
        return False

    return True

def normalized_speed_diff(node_i, node_j, veh_table, bus_table, eps=1e-6):
    si = veh_table.values(node_i)['speed']
    sj = get_node_state(node_j, veh_table, bus_table)['speed']
    denom = max(abs(si), abs(sj), eps)
    return min(abs(si - sj) / denom, 1.0)

def normalized_distance(node_i, node_j, veh_table, bus_table):
    table_j = get_node_table(node_j, veh_table, bus_table)
    d = util.det_dist(node_i, veh_table, node_j, table_j)
    denom = max(min(veh_table.values(node_i)['trans_range'],
                    table_j.values(node_j)['trans_range']), 1.0)
    return min(d / denom, 1.0)

def estimated_attach_hop_to_root(parent_id, veh_table, bus_table):
    ph = get_hop_count(parent_id, veh_table, bus_table)
    return float('inf') if ph is None else (ph + 1)

def normalized_hop_cost(hop_value, config):
    if hop_value in [None, float('inf')]:
        return 1.0
    return min(hop_value / max(config.h_max, 1), 1.0)

def root_cost(veh_id, root_id,
              veh_table, bus_table,
              meso_zones, macro_zones,
              config):
    """
    Root-level suitability using only dimensionless normalized terms:
      - meso directional mismatch
      - macro directional mismatch
      - macro-zone distance
      - normalized speed mismatch to root
      - root load
      - previous-root preference
      - DRU preference
    """
    vi = veh_table.values(veh_id)
    vr = get_node_state(root_id, veh_table, bus_table)

    # normalized angular mismatch terms in [0,1]
    mes_dir = zotsim_cost(veh_id, root_id, veh_table, bus_table, meso_zones, level='meso')
    mac_dir = zotsim_cost(veh_id, root_id, veh_table, bus_table, macro_zones, level='macro')

    # normalized macro-zone distance in [0,1]
    mac_dist = normalized_zone_distance(
        vi['macro_zone'], vr['macro_zone'],
        macro_zones, mode='manhattan'
    )

    # normalized speed mismatch in [0,1]
    speed_term = normalized_speed_diff(veh_id, root_id, veh_table, bus_table)

    # normalized load in [0,1]
    load_term = normalized_node_load(root_id, veh_table, bus_table, config, is_root=True)

    # bonuses
    prev_bonus = 1.0 if vi.get('priority_ch') == root_id and vi.get('priority_counter', 0) > 0 else 0.0
    dru_bonus = 1.0 if is_bus(root_id) else 0.0

    cost = (
        config.w_root_meso_dir * mes_dir +
        config.w_root_macro_dir * mac_dir +
        config.w_root_macro_dist * mac_dist +
        config.w_root_speed * speed_term +
        config.w_root_load * load_term -
        config.w_root_prev * prev_bonus -
        config.w_root_dru * dru_bonus
    )
    return float(cost)

def parent_cost(veh_id, parent_id,
                veh_table, bus_table,
                micro_zones,
                config):
    """
    Parent-level local suitability using only dimensionless normalized terms:
      - normalized micro angular mismatch
      - normalized speed mismatch
      - normalized distance
      - normalized hop depth
      - normalized parent load
    """
    micro_dir = zotsim_cost(veh_id, parent_id, veh_table, bus_table, micro_zones, level='micro')
    speed_term = normalized_speed_diff(veh_id, parent_id, veh_table, bus_table)
    dist_term = normalized_distance(veh_id, parent_id, veh_table, bus_table)
    hop_term = normalized_hop_cost(estimated_attach_hop_to_root(parent_id, veh_table, bus_table), config)
    load_term = normalized_node_load(parent_id, veh_table, bus_table, config, is_root=False)

    cost = (
        config.w_parent_micro_dir * micro_dir +
        config.w_parent_speed * speed_term +
        config.w_parent_dist * dist_term +
        config.w_parent_hop * hop_term +
        config.w_parent_load * load_term
    )
    return float(cost)

def distinct_roots_from_candidates(bus_candidates, ch_candidates, rn_candidates,
                                   veh_table, bus_table):
    roots = set()

    for b in bus_candidates:
        roots.add(b)

    for ch in ch_candidates:
        roots.add(ch)

    for rn in rn_candidates:
        r = candidate_root(rn, veh_table, bus_table)
        if r is not None:
            roots.add(r)

    return roots

def visible_parents_for_root(root_id, bus_candidates, ch_candidates, rn_candidates,
                             veh_table, bus_table):
    """
    Return visible candidates that lead to root_id.
    """
    parents = set()

    for b in bus_candidates:
        if b == root_id:
            parents.add(b)

    for ch in ch_candidates:
        if ch == root_id:
            parents.add(ch)

    for rn in rn_candidates:
        if candidate_root(rn, veh_table, bus_table) == root_id:
            parents.add(rn)

    return parents

def feasible_roots_and_parents(veh_id,
                               bus_candidates, ch_candidates, rn_candidates,
                               veh_table, bus_table,
                               micro_zones, meso_zones, macro_zones,
                               config):
    """
    Returns:
      dict[root_id] = set(feasible parent_ids leading to root_id)
    """
    roots = distinct_roots_from_candidates(bus_candidates, ch_candidates, rn_candidates,
                                           veh_table, bus_table)
    root_to_parents = {}

    for root_id in roots:
        parents = visible_parents_for_root(root_id, bus_candidates, ch_candidates, rn_candidates,
                                           veh_table, bus_table)
        feasible = set()
        for p in parents:
            if feasible_parent_for_root(veh_id, p, root_id,
                                        veh_table, bus_table,
                                        micro_zones, meso_zones, macro_zones,
                                        config):
                feasible.add(p)
        if len(feasible) > 0:
            root_to_parents[root_id] = feasible

    return root_to_parents


def switch_allowed(current_cost, new_cost, delta):
    if current_cost is None:
        return True
    return (new_cost + delta) < current_cost

def rn_utility(node_id, veh_table, bus_table,
               stand_alone, zone_stand_alone,
               micro_zones, meso_zones, macro_zones,
               config):
    """
    Clustering-oriented relay usefulness.
    Higher is better.
    """
    if is_bus(node_id):
        return -float('inf')

    st = veh_table.values(node_id)
    if st['cluster_head'] is True or st['primary_ch'] is None:
        return -float('inf')

    # SA-absorption gain: nearby unresolved SAs in neighbor macro zones
    nearby_sa = util.det_near_sa(node_id, veh_table, stand_alone, zone_stand_alone)
    sa_gain = len(nearby_sa)

    # Root gain: prefer nodes already attached to a real root with smaller hop
    root_gain = 1.0
    hop_term = get_hop_count(node_id, veh_table, bus_table)
    hop_penalty = normalized_hop_cost(hop_term, config)

    # Coherence gain: compare node with its root
    root_id = get_root(node_id, veh_table, bus_table)
    if root_id is None:
        coherence = 0.0
    else:
        mes_sim = zotsim_similarity(node_id, root_id, veh_table, bus_table, meso_zones, level='meso')
        mac_sim = zotsim_similarity(node_id, root_id, veh_table, bus_table, macro_zones, level='macro')
        coherence = 0.5 * (mes_sim + mac_sim)

    load_penalty = normalized_node_load(node_id, veh_table, bus_table, config, is_root=False)

    util_val = (
        config.w_rn_sa_gain * min(sa_gain / 10.0, 1.0) +
        config.w_rn_root_gain * root_gain +
        config.w_rn_coherence * coherence -
        config.w_rn_hop_penalty * hop_penalty -
        config.w_rn_load_penalty * load_penalty
    )
    return float(util_val)

def pch_score(veh_id, cand_id, near_sa,
              veh_table,
              micro_zones, meso_zones,
              config):
    """
    Higher is better.
    """
    if cand_id != veh_id and cand_id not in near_sa.get(veh_id, set()):
        return -float('inf')

    degree_term = min(len(near_sa.get(cand_id, set())) / 10.0, 1.0)
    dir_term = zotsim_similarity(veh_id, cand_id, veh_table, veh_table, micro_zones, level='micro')
    dist_term = normalized_distance(veh_id, cand_id, veh_table, veh_table)

    vi = veh_table.values(veh_id)
    vc = veh_table.values(cand_id)

    micro_zone_term = normalized_zone_distance(vi['micro_zone'], vc['micro_zone'], micro_zones, mode='chebyshev')
    meso_zone_term = normalized_zone_distance(vi['meso_zone'], vc['meso_zone'], meso_zones, mode='chebyshev')

    score = (
        config.w_pch_degree * degree_term +
        config.w_pch_micro_dir * dir_term -
        config.w_pch_dist * dist_term -
        config.w_pch_micro_zone * micro_zone_term -
        config.w_pch_meso_zone * meso_zone_term
    )
    return float(score)

def best_pch_for_sa(veh_id, near_sa,
                    veh_table,
                    micro_zones, meso_zones,
                    config):
    candidates = set(near_sa.get(veh_id, set())) | {veh_id}
    best_id = None
    best_score = -float('inf')

    for cand in candidates:
        if cand not in veh_table.ids():
            continue
        sc = pch_score(veh_id, cand, near_sa, veh_table, micro_zones, meso_zones, config)
        if sc > best_score:
            best_score = sc
            best_id = cand

    return best_id, best_score

def attach_to_parent(root_id, parent_id, veh_id,
                     veh_table, bus_table, config, sec,
                     bus_candidates, ch_candidates,
                     stand_alone, zone_stand_alone, other_vehs,
                     root_cost_value=None, parent_cost_value=None):
    """
    Commit a fresh SA attachment to a validated arbitrary-depth parent chain.

    Preserve the legacy membership tables. A move of an ALREADY ATTACHED CM
    requires an atomic subtree-reparent operation and is rejected here rather
    than silently corrupting its descendants or the cluster episode record.
    """
    member = veh_table.values(veh_id)
    root = get_node_state(root_id, veh_table, bus_table)
    parent = get_node_state(parent_id, veh_table, bus_table)
    if member is None or root is None or parent is None:
        raise ValueError('MMZCA attachment references a missing node')
    if veh_id in (root_id, parent_id):
        raise ValueError('A member cannot be its own root or parent')
    if get_node_table(root_id, veh_table, bus_table) is not bus_table and not root.get('cluster_head'):
        raise ValueError('Selected root is not a current CH')
    if (member.get('primary_ch') is not None or member.get('parent_node') is not None
            or member.get('parent_id') is not None):
        raise ValueError('A currently attached member needs atomic subtree reparenting')

    # Validate the entire parent chain, not just a presumed two-hop RN.
    seen = {veh_id}
    cursor = parent_id
    while cursor != root_id:
        if cursor in seen:
            raise ValueError('Cycle detected in the chosen parent chain')
        seen.add(cursor)
        node = get_node_state(cursor, veh_table, bus_table)
        if (node is None or node.get('root_connection_valid') is False or
                get_root(cursor, veh_table, bus_table) != root_id):
            raise ValueError('Parent chain is not root-connected')
        cursor = get_parent(cursor, veh_table, bus_table)
        if cursor is None:
            raise ValueError('Parent chain does not reach its root')

    parent_depth = get_hop_count(parent_id, veh_table, bus_table)
    if parent_depth is None or int(parent_depth) + 1 > int(config.h_max):
        raise ValueError('Attachment violates the MMZCA maximum hop count')
    capacity = (config.max_ch_members if parent_id == root_id
                else config.max_rn_members)
    children = parent.get('direct_children', set())
    if len(children) >= int(capacity) and veh_id not in children:
        raise ValueError('The selected parent has reached its child capacity')

    # direct attachment to root
    if parent_id == root_id:
        bus_table, veh_table, stand_alone, zone_stand_alone = util.add_member(
            root_id, bus_table, veh_id, veh_table, config, parent_cost_value, sec,
            bus_candidates, ch_candidates, stand_alone, zone_stand_alone, other_vehs
        )
        veh_table.values(veh_id)['root_ch'] = root_id
        veh_table.values(veh_id)['parent_node'] = parent_id
        veh_table.values(veh_id)['hop_count'] = 1
    else:
        bus_table, veh_table, stand_alone, zone_stand_alone = util.add_sub_member(
            root_id, bus_table, veh_id, parent_id, veh_table, config, parent_cost_value, sec,
            bus_candidates, ch_candidates, stand_alone, zone_stand_alone, other_vehs
        )
        veh_table.values(veh_id)['root_ch'] = root_id
        veh_table.values(veh_id)['parent_node'] = parent_id
        veh_table.values(veh_id)['hop_count'] = int(parent_depth) + 1

    veh_table.values(veh_id)['primary_ch'] = root_id
    veh_table.values(veh_id)['secondary_ch'] = None if parent_id == root_id else parent_id
    veh_table.values(veh_id)['parent_id'] = parent_id
    veh_table.values(veh_id)['current_root_cost'] = root_cost_value
    veh_table.values(veh_id)['current_parent_cost'] = parent_cost_value
    veh_table.values(veh_id)['path_bottleneck'] = None
    veh_table.values(veh_id)['committed_link_score'] = None
    veh_table.values(veh_id)['kernel_descriptor'] = None
    veh_table.values(veh_id)['kernel_root_incarnation'] = root.get('kernel_root_incarnation')
    veh_table.values(veh_id)['root_connection_valid'] = True
    parent.setdefault('direct_children', set()).add(veh_id)
    veh_table.values(veh_id)['unresolved_start_macro_zone'] = None

    return bus_table, veh_table, stand_alone, zone_stand_alone


def reparent_within_root(veh_id, new_parent_id, veh_table, bus_table,
                         config, meso_zones, macro_zones, new_parent_cost,
                         child_map=None):
    """Atomically change an attached CM's parent while preserving its root.

    Validation finishes BEFORE any mutation.  The entire descendant subtree
    moves with the member, without detachment, cluster-record interruption,
    or an SA/CH transition.  Throws ValueError for an invalid proposed move.
    The caller must have already checked the *improvement* against hysteresis.
    """
    member = veh_table.values(veh_id)
    if member is None or member.get('cluster_head') is True:
        raise ValueError('Only attached ordinary CMs may switch parents')
    root_id = member.get('root_ch') or member.get('primary_ch')
    old_parent_id = get_parent(veh_id, veh_table, bus_table)
    if root_id is None or old_parent_id is None:
        raise ValueError('The member must have an existing valid root affiliation')
    if new_parent_id == old_parent_id:
        return False
    root = get_node_state(root_id, veh_table, bus_table)
    old_parent = get_node_state(old_parent_id, veh_table, bus_table)
    new_parent = get_node_state(new_parent_id, veh_table, bus_table)
    if not all(x is not None for x in (root, old_parent, new_parent)):
        raise ValueError('Root or parent has departed')
    if (get_root(new_parent_id, veh_table, bus_table) != root_id or
            (new_parent_id != root_id and new_parent.get('root_connection_valid') is not True)):
        raise ValueError('Elective switching across roots is prohibited')
    if new_parent.get('admit_children') is False:
        raise ValueError('New parent does not admit children')

    # Derive current actual immediate-child edges; do not trust transitive
    # legacy sub_cluster_members as a hierarchy or capacity measure.
    if child_map is None:
        child_map = {x: set() for x in veh_table.ids() | bus_table.ids()}
        for node_id in veh_table.ids():
            node = veh_table.values(node_id)
            if node.get('cluster_head') is True:
                continue
            parent = (node.get('parent_node') or node.get('parent_id') or
                      node.get('secondary_ch') or node.get('primary_ch'))
            if parent in child_map:
                child_map[parent].add(node_id)

    # Validate the whole committed subtree, including its relative depths.
    subtree, relative_depth, stack = [], {}, [(veh_id, 0)]
    while stack:
        node_id, rel = stack.pop()
        if node_id in relative_depth:
            raise ValueError('Cycle in existing subtree')
        node = veh_table.values(node_id)
        if node is None or get_root(node_id, veh_table, bus_table) != root_id:
            raise ValueError('Existing subtree is not root-consistent')
        # The move must not silently repair a previously broken hierarchy.
        try:
            expected = int(member['hop_count']) + rel
            if node.get('hop_count') != expected:
                raise ValueError('Existing subtree has inconsistent hop depths')
        except (TypeError, KeyError):
            raise ValueError('Existing subtree has missing hop depth')
        relative_depth[node_id] = rel
        subtree.append(node_id)
        stack.extend((child, rel + 1) for child in child_map.get(node_id, ()))
    if new_parent_id in relative_depth:
        raise ValueError('Reparenting to oneself or a descendant creates a cycle')

    # Validate the candidate's upstream path, including correct depth and
    # physically feasible links. No remote/root membership list is assumed.
    seen, cur = set(relative_depth), new_parent_id
    while cur != root_id:
        if cur in seen:
            raise ValueError("Candidate's ancestry intersects the moved subtree")
        seen.add(cur)
        state = get_node_state(cur, veh_table, bus_table)
        if (state is None or state.get('root_connection_valid') is not True or
                get_root(cur, veh_table, bus_table) != root_id):
            raise ValueError('New parent lacks a valid root connection')
        parent_id = get_parent(cur, veh_table, bus_table)
        parent = get_node_state(parent_id, veh_table, bus_table)
        if parent is None or state.get('hop_count') != parent.get('hop_count', 0) + 1:
            raise ValueError('Invalid candidate upstream depth')
        parent_table = get_node_table(parent_id, veh_table, bus_table)
        state_table = get_node_table(cur, veh_table, bus_table)
        if util.det_dist(cur, state_table, parent_id, parent_table) > min(
                float(state['trans_range']), float(parent['trans_range'])):
            raise ValueError('Candidate upstream link is no longer feasible')
        cur = parent_id
    if root.get('cluster_head') is not True and get_node_table(root_id, veh_table, bus_table) is not bus_table:
        raise ValueError('The committed root has lost CH status')

    depth = int(new_parent['hop_count']) + 1
    if depth + max(relative_depth.values()) > int(config.h_max):
        raise ValueError('Moving this subtree exceeds the maximum hop count')
    limit = int(config.max_ch_members if new_parent_id == root_id
                else config.max_rn_members)
    if len(child_map[new_parent_id]) >= limit:
        raise ValueError('The new parent is full')
    link = util.mmzca_physical_link(veh_id, new_parent_id,
                                  veh_table, get_node_table(new_parent_id, veh_table, bus_table),
                                  meso_zones, macro_zones, config)
    if link is None:
        raise ValueError('Proposed new parent link is not physically feasible')

    # Compute every future bottleneck before touching ANY committed state.
    upstream_Q = (0.0 if new_parent_id == root_id
                  else new_parent.get('path_bottleneck'))
    if upstream_Q is None:
        raise ValueError('New parent has no verified upstream bottleneck')
    prospective_Q = {}
    for node_id in sorted(subtree, key=lambda n: relative_depth[n]):
        node = veh_table.values(node_id)
        if node_id == veh_id:
            parent_Q, L = float(upstream_Q), float(link['L'])
        else:
            parent_id = get_parent(node_id, veh_table, bus_table)
            if parent_id not in prospective_Q:
                raise ValueError('Subtree has an inconsistent parent order')
            parent_Q, L = prospective_Q[parent_id], node.get('committed_link_score')
            if L is None:
                raise ValueError('Descendant lacks a committed physical link score')
            parent = get_node_state(parent_id, veh_table, bus_table)
            if (parent is None or util.det_dist(node_id, veh_table, parent_id, veh_table)
                    > min(float(node['trans_range']), float(parent['trans_range']))):
                raise ValueError('Descendant has an invalid physical link')
        prospective_Q[node_id] = max(float(parent_Q), float(L))

    # Synchronous simulator commit: there is no externally visible intermediate
    # tree state. All expected validation errors have already been raised.
    old_parent.setdefault('direct_children', set()).discard(veh_id)
    new_parent.setdefault('direct_children', set()).add(veh_id)
    if old_parent_id == root_id:
        root.setdefault('cluster_members', set()).discard(veh_id)
        root.setdefault('gates', {}).pop(veh_id, None)
    else:
        old_parent.setdefault('sub_cluster_members', set()).discard(veh_id)
    if new_parent_id == root_id:
        root.setdefault('cluster_members', set()).add(veh_id)
        root.setdefault('sub_cluster_members', set()).discard(veh_id)
        root.setdefault('gates', {})[veh_id] = set(member.get('other_chs', ()))
    else:
        new_parent.setdefault('sub_cluster_members', set()).add(veh_id)
        root.setdefault('sub_cluster_members', set()).add(veh_id)
    if old_parent_id != root_id and new_parent_id == root_id:
        root.setdefault('sub_cluster_members', set()).discard(veh_id)
    # The previously accepted histogram of the moved child belongs to its old
    # parent and must not contribute to its next kernel aggregation.
    old_parent.setdefault('kernel_child_reports', {}).pop(veh_id, None)
    old_parent.setdefault('kernel_known_children', set()).discard(veh_id)
    old_parent['kernel_last_signature'] = None
    new_parent.setdefault('kernel_child_reports', {}).pop(veh_id, None)
    new_parent.setdefault('kernel_known_children', set()).discard(veh_id)
    new_parent['kernel_last_signature'] = None
    for p_id in (old_parent_id, new_parent_id):
        p = get_node_state(p_id, veh_table, bus_table)
        if p_id != root_id and p.get('role') != 'CH':
            p['role'] = 'RN' if p.get('direct_children') else 'CM'
    root['gate_chs'] = set().union(*(
        set(xs) for xs in root.get('gates', {}).values()
    )) if root.get('gates') else set()

    member['parent_node'] = new_parent_id
    member['parent_id'] = new_parent_id
    member['secondary_ch'] = None if new_parent_id == root_id else new_parent_id
    member['committed_link_score'] = float(link['L'])
    member['current_parent_cost'] = float(new_parent_cost)
    member['role'] = 'RN' if child_map.get(veh_id) else 'CM'
    epoch = root.get('kernel_root_incarnation')
    for node_id in sorted(subtree, key=lambda n: relative_depth[n]):
        st = veh_table.values(node_id)
        st['hop_count'] = depth + relative_depth[node_id]
        st['root_ch'] = root_id
        st['primary_ch'] = root_id
        st['root_connection_valid'] = True
        st['kernel_root_incarnation'] = epoch
        st['path_bottleneck'] = prospective_Q[node_id]
        st['kernel_last_signature'] = None  # force updated subtree hist versions
    # Retain the same root-based CM residence episode, but record the changed
    # immediate relay in the existing optional legacy metadata.
    record = member.get('cluster_record')
    if record is not None and getattr(record, 'tail', None) is not None:
        fields = record.tail.value
        if isinstance(fields, dict) and isinstance(fields.get('secondary_ch'), list):
            fields['secondary_ch'].append(None if new_parent_id == root_id else new_parent_id)
    return True


def detach_from_current_cluster(veh_id, veh_table, bus_table, config,
                                stand_alone, zone_stand_alone):
    """
    Generic detach wrapper.
    """
    st = veh_table.values(veh_id)
    root_id = get_root(veh_id, veh_table, bus_table)
    parent_id = get_parent(veh_id, veh_table, bus_table)

    if root_id is None or parent_id is None:
        return veh_table, bus_table, stand_alone, zone_stand_alone

    if parent_id == root_id:
        veh_table, bus_table, stand_alone, zone_stand_alone = util.remove_member(
            veh_id, root_id, veh_table, bus_table, config, stand_alone, zone_stand_alone
        )
    else:
        veh_table, bus_table, stand_alone, zone_stand_alone = util.remove_sub_member(
            veh_id, parent_id, root_id, veh_table, bus_table, config, stand_alone, zone_stand_alone
        )

    reset_cluster_fields(veh_id, veh_table, bus_table)
    veh_table.values(veh_id)['unresolved_start_macro_zone'] = veh_table.values(veh_id)['macro_zone']
    return veh_table, bus_table, stand_alone, zone_stand_alone

def choose_best_root(veh_id, root_to_parents,
                     veh_table, bus_table,
                     meso_zones, macro_zones,
                     config):
    """
    Returns best root and its cost.
    """
    best_root = None
    best_cost = float('inf')

    for root_id in root_to_parents.keys():
        c = root_cost(veh_id, root_id, veh_table, bus_table, meso_zones, macro_zones, config)
        if c < best_cost:
            best_cost = c
            best_root = root_id

    return best_root, best_cost

def choose_best_parent_for_root(veh_id, root_id, candidate_parents,
                                veh_table, bus_table,
                                micro_zones,
                                config):
    best_parent = None
    best_cost = float('inf')

    for p in candidate_parents:
        c = parent_cost(veh_id, p, veh_table, bus_table, micro_zones, config)
        if c < best_cost:
            best_cost = c
            best_parent = p

    return best_parent, best_cost

def get_parent(node_id, veh_table, bus_table):
    st = get_node_state(node_id, veh_table, bus_table)

    if get_node_table(node_id, veh_table, bus_table) is bus_table:
        return None
    if st['cluster_head'] is True:
        return None
    if st.get('parent_node') is not None:
        return st['parent_node']
    if st.get('parent_id') is not None:
        return st['parent_id']
    if st.get('secondary_ch') is not None:
        return st['secondary_ch']
    return st.get('primary_ch')

def candidate_type(node_id, veh_table, bus_table):
    if get_node_table(node_id, veh_table, bus_table) is bus_table:
        return 'root'
    st = veh_table.values(node_id)
    if st['cluster_head'] is True:
        return 'root'
    return 'rn'

def rn_reported_score(rn_id, veh_table, default_value=1.0):
    """
    Score RN advertises from the time it joined its current cluster.
    Uses cluster_record tail ef as compatibility proxy.
    """
    if rn_id not in veh_table.ids():
        return default_value

    tail = veh_table.values(rn_id).get('cluster_record', None)
    if tail is None or tail.tail is None:
        return default_value

    ef_val = tail.tail.value.get('ef', None)
    if ef_val is None:
        return default_value

    return float(ef_val)

def join_cost_for_candidate(veh_id, cand_id,
                            veh_table, bus_table,
                            micro_zones, meso_zones, macro_zones,
                            config):
    """
    One-step join score:
      - root CH / bus: use C_root
      - RN: weighted sum of local C_parent and RN's reported score
    """
    ctype = candidate_type(cand_id, veh_table, bus_table)

    if ctype == 'root':
        return root_cost(
            veh_id, cand_id,
            veh_table, bus_table,
            meso_zones, macro_zones,
            config
        )

    # RN case
    local_parent = parent_cost(
        veh_id, cand_id,
        veh_table, bus_table,
        micro_zones,
        config
    )
    reported = rn_reported_score(cand_id, veh_table, default_value=1.0)

    return float(
        config.lambda_rn * local_parent +
        (1.0 - config.lambda_rn) * reported
    )

def feasible_visible_candidate(veh_id, cand_id,
                               veh_table, bus_table,
                               micro_zones, meso_zones, macro_zones,
                               config):
    """
    Root candidates:
      - direct link feasible
      - optional light integrity checks

    RN candidates:
      - direct link feasible
      - same checks
      - hop limit feasible through RN
    """
    if not direct_link_feasible(veh_id, cand_id, veh_table, bus_table):
        return False

    ctype = candidate_type(cand_id, veh_table, bus_table)

    if ctype == 'root':
        root_id = cand_id
        parent_id = cand_id
    else:
        root_id = get_root(cand_id, veh_table, bus_table)
        parent_id = cand_id
        if root_id is None:
            return False

    # keep your integrity logic here for now
    if not hierarchical_integrity_ok(
        veh_id, parent_id, root_id,
        veh_table, bus_table,
        micro_zones, meso_zones, macro_zones,
        config
    ):
        return False

    if ctype == 'rn':
        if not hop_limit_ok(cand_id, veh_table, bus_table, config):
            return False

    return True

def priority_visible_candidates(veh_id, bus_candidates, ch_candidates, rn_candidates, veh_table):
    """
    Restrict visible candidates to the previous root cluster during priority window.
    """
    pr = veh_table.values(veh_id).get('priority_ch')
    pc = veh_table.values(veh_id).get('priority_counter', 0)

    if pr is None or pc <= 0:
        return bus_candidates, ch_candidates, rn_candidates

    new_bus = set()
    new_ch = set()
    new_rn = set()

    for b in bus_candidates:
        if b == pr:
            new_bus.add(b)

    for ch in ch_candidates:
        if ch == pr:
            new_ch.add(ch)

    for rn in rn_candidates:
        if veh_table.values(rn)['primary_ch'] == pr:
            new_rn.add(rn)

    if len(new_bus) + len(new_ch) + len(new_rn) > 0:
        return new_bus, new_ch, new_rn

    return bus_candidates, ch_candidates, rn_candidates

def choose_best_visible_candidate(veh_id,
                                  bus_candidates, ch_candidates, rn_candidates,
                                  veh_table, bus_table,
                                  micro_zones, meso_zones, macro_zones,
                                  config):
    best_cand = None
    best_cost = float('inf')

    all_candidates = list(bus_candidates | ch_candidates | rn_candidates)

    for cand in all_candidates:
        if not feasible_visible_candidate(
            veh_id, cand,
            veh_table, bus_table,
            micro_zones, meso_zones, macro_zones,
            config
        ):
            continue

        c = join_cost_for_candidate(
            veh_id, cand,
            veh_table, bus_table,
            micro_zones, meso_zones, macro_zones,
            config
        )

        if c < best_cost:
            best_cost = c
            best_cand = cand

    return best_cand, best_cost

def intra_cluster_switch_allowed(veh_id, new_cand_id, veh_table, bus_table):
    """
    Allow switch only inside the same root cluster.
    """
    current_root = get_root(veh_id, veh_table, bus_table)

    if current_root is None:
        return True

    ctype = candidate_type(new_cand_id, veh_table, bus_table)
    if ctype == 'root':
        new_root = new_cand_id
    else:
        new_root = get_root(new_cand_id, veh_table, bus_table)

    return new_root == current_root


# ---------------------------------------------------------------------------
# DHCV forest construction/commit. Existing MMZCA functions remain untouched.
# ---------------------------------------------------------------------------

def dhcv_bfs(root, graph, allowed, depth_limit, cost=None):
    """Rooted, cycle-free, <=D-hop tree over currently unassigned nodes."""
    from collections import deque
    allowed = set(allowed)
    if root not in allowed:
        return {}
    discovered = {root: (root, None, 0)}
    queue = deque([root])
    while queue:
        parent = queue.popleft()
        depth = discovered[parent][2]
        if depth >= depth_limit:
            continue
        neighbors = sorted((j for j in graph.get(parent, ()) if j in allowed
                            and j not in discovered),
                           key=lambda j: ((cost(parent, j) if cost else 0.0), str(j)))
        for j in neighbors:
            if j in discovered:
                continue
            discovered[j] = (root, parent, depth+1)
            queue.append(j)
    return discovered


def dhcv_form_forest(active_ids, graph, pch_ranks, veh_table, depth_limit,
                     cnn_chains=None):
    """Resolve the paper's CNN/PCH nominations into non-overlapping D-hop trees.

    Crucial: do NOT take a BFS over every available edge. DHCV selects a
    *single* least-relative-mobility CNN neighbor at each hop. A BFS over all
    edges can incorrectly absorb whole connected components into one CH and
    produce unrealistically low CH counts and long CM residence.

    A member follows its CNN_1 choice toward a nominated PCH, or, under
    Exception 1, the CH of CNN_1 when the preferred path cannot be completed.
    In the existing CH/CM/SA framework, an uncovered vehicle remains SA.
    The article does not give a disconnected-singleton state convention.
    """
    active = set(active_ids)
    if not active:
        return {}
    cnn_chains = (cnn_chains or {})
    first = {i: (cnn_chains[i][1] if len(cnn_chains.get(i, ())) > 1 else None)
             for i in active}
    # A node with no admissible CNN_1 is not a valid forwarding parent in
    # its own local RelativeMax search. All assignments still check 'graph'.
    # During selective VI repair, `active` contains ONLY unresolved vehicles,
    # while rankings/chains were computed over the whole physical network.
    # A PCH nominated outside `active` may already be a CM with descendants
    # in the retained forest. Promoting it to a new CH here would overwrite
    # its existing root/parent and corrupt every dependent ancestor chain.
    # Existing clusters are handled separately by dhcv_attach_to_existing().
    nominators = {i: pch_ranks[i][0] for i in active
                  if (first[i] is not None and pch_ranks.get(i)
                      and pch_ranks[i][0] in active)}
    nominated = set(nominators.values())
    support = {root: sum(1 for i, r in nominators.items()
                         if r == root and i != root) for root in nominated}
    order = sorted(nominated, key=lambda i: (
        -support[i], -len(graph.get(i, ())),
        float(veh_table.values(i)['speed']), str(i)))
    plan = {}
    for root in order:
        if root in plan:
            continue
        proposal = {root: (root, None, 0)}
        # Reverse the directed, chosen CNN_1 links. All incoming nodes must
        # independently nominate this root, rather than merely be D-hop
        # reachable via an arbitrary graph path.
        for _ in range(depth_limit):
            added = []
            for i in sorted(active - plan.keys() - proposal.keys()):
                if nominators.get(i) != root:
                    continue
                parent = first[i]
                if parent not in proposal or parent not in graph.get(i, ()):
                    continue
                depth = proposal[parent][2] + 1
                if depth <= depth_limit:
                    added.append((i, root, parent, depth))
            if not added:
                break
            for i, r, parent, depth in added:
                proposal[i] = (r, parent, depth)
        # Exception 3: no member => no effective cluster in the existing
        # framework. Avoid automatic singleton CH promotion.
        if len(proposal) >= 2:
            plan.update(proposal)

    # Exception 1: failed PCH route may instead join CNN_1's root, provided
    # the same selected one-hop link is physically valid and D is respected.
    pending = active - plan.keys()
    plan, pending = dhcv_attach_to_existing(
        plan, graph, pending, pch_ranks, depth_limit,
        cnn_chains=cnn_chains)

    # A group of nodes can nominate a CH that is not yet committed but would
    # gain members whose selected CNN chain points to it. Try again after
    # earlier conflicts have resolved, without inventing BFS paths.
    for root in order:
        if root not in pending:
            continue
        proposal = {root: (root, None, 0)}
        for _ in range(depth_limit):
            added = []
            for i in sorted(pending - proposal.keys()):
                if nominators.get(i) != root:
                    continue
                parent = first[i]
                if parent in proposal and parent in graph.get(i, ()):
                    depth = proposal[parent][2] + 1
                    if depth <= depth_limit:
                        added.append((i, root, parent, depth))
            if not added:
                break
            for i, r, parent, depth in added:
                proposal[i] = (r, parent, depth)
        if len(proposal) > 1:
            plan.update(proposal)
            pending.difference_update(proposal)
            plan, pending = dhcv_attach_to_existing(
                plan, graph, pending, pch_ranks, depth_limit,
                cnn_chains=cnn_chains)

    # Exception 3 (CNN_1 fallback): a nominated, childless CH can join its
    # chosen CNN_1 instead of remaining a singleton. When that CNN_1 is
    # unresolved, the endpoint becomes the new CH and recruits the initiator;
    # no arbitrary graph-neighbor or network-wide root is introduced.
    # This also handles asymmetric local thresholds (i chooses j, but j does
    # not choose i) without fabricating all-node CH assignments.
    for i in sorted(tuple(pending), key=lambda k: (
            -len(graph.get(k, ())), float(veh_table.values(k)['speed']), str(k))):
        if i not in pending:
            continue
        j = first.get(i)
        if j not in pending or j not in graph.get(i, ()) or j == i:
            continue
        plan[j] = (j, None, 0)
        plan[i] = (j, j, 1)
        pending.discard(i)
        pending.discard(j)
        plan, pending = dhcv_attach_to_existing(
            plan, graph, pending, pch_ranks, depth_limit,
            cnn_chains=cnn_chains)

    # Exception 2: CH which is also CM elsewhere may join another CH only if
    # its *whole* branch can move within D, and only through its CNN_1 link.
    for head in sorted(tuple(plan)):
        if plan[head][1] is not None:
            continue
        nominees = pch_ranks.get(head, ())
        neighbor = first.get(head)
        if not nominees or nominees[0] == head or neighbor not in plan:
            continue
        other_root, _, other_depth = plan[neighbor]
        if other_root == head or neighbor not in graph.get(head, ()):
            continue
        subtree = {i for i, (root, _, _) in plan.items() if root == head}
        tail_depth = max(plan[i][2] for i in subtree)
        if other_depth + 1 + tail_depth > depth_limit:
            continue
        for i in subtree:
            _, old_parent, old_depth = plan[i]
            plan[i] = (other_root,
                       neighbor if i == head else old_parent,
                       other_depth + 1 + old_depth)
    # Defensive contract: a local forest formation must never take ownership
    # of a retained vehicle outside the explicitly supplied recovery set.
    unexpected = set(plan).difference(active)
    if unexpected:
        raise AssertionError('DHCV local formation captured external nodes: '
                             + repr(sorted(unexpected)))
    return plan


def dhcv_attach_to_existing(plan, graph, available, pch_ranks, depth_limit,
                            eligible_parents=None, cnn_chains=None):
    """Repair or initial fallback using the node's selected CNN_1 hop.

    This is intentionally NOT a greedy search over all graph neighbors.
    Otherwise almost every vehicle can be attached under a distant CH that
    it never selected, which destroys the published DHCV decision rule.
    """
    pending = set(available)
    if not pending:
        return plan, pending
    safe = set(plan) if eligible_parents is None else set(eligible_parents)
    cnn_chains = cnn_chains or {}
    for _ in range(depth_limit):
        best = []
        for i in pending:
            path = cnn_chains.get(i, ())
            if len(path) < 2:
                continue
            j = path[1]
            if j not in safe or j not in graph.get(i, ()):
                continue
            root, _, depth = plan[j]
            if depth + 1 > depth_limit:
                continue
            preferred = pch_ranks.get(i, ())
            key = (preferred.index(root) if root in preferred else len(preferred),
                   depth + 1, str(root), str(i))
            best.append((key, i, j, root, depth + 1))
        if not best:
            break
        # Single synchronous round of local assertions; newly connected
        # parents only become eligible on the next iteration of this loop.
        attachments = sorted(best)
        for _, i, j, root, depth in attachments:
            if i not in pending or j not in safe:
                continue
            plan[i] = (root, j, depth)
            pending.discard(i)
            safe.add(i)
    return plan, pending


def dhcv_validate_forest(plan, physical, D, allow_grace=()):
    """Strong connected-tree invariants for every non-grace assignment."""
    grace = set(allow_grace)
    for node, (root, parent, depth) in plan.items():
        if depth > D or depth < 0:
            raise AssertionError('DHCV hop limit violated: ' + str(node))
        if parent is None:
            if node != root or depth != 0:
                raise AssertionError('Non-root DHCV singleton: ' + str(node))
            continue
        if node not in grace and parent not in physical.get(node, ()):
            raise AssertionError('DHCV nonphysical parent link: ' + str(node))
        if parent not in plan:
            raise AssertionError('DHCV missing parent: ' + str(node))
        proot, _, pd = plan[parent]
        if proot != root or pd + 1 != depth:
            raise AssertionError(
                'DHCV inconsistent root/ancestor: node=' + str(node) +
                ' assignment=' + repr((root, parent, depth)) +
                ' parent_assignment=' + repr(plan[parent]))
    return True


def dhcv_commit_forest(cluster, plan, config, *, event_reason='dhcv_renew', grace=()):
    """Atomic role, root, parent and zone-index commit on EXISTING vehicle records.

    Critically this does not replace DataTable, HashTable, SUMO update, or the
    shared root-based stability metric. All membership sets are reconstructed
    from the authoritative parent pointers. A renewed CM of the same root
    remains in the same residence episode, even if its relay parent changes.
    """
    grace = set(grace)
    ids = cluster.veh_table.ids()
    children = {i: set() for i in ids}
    for i, (root, parent, depth) in plan.items():
        if parent is not None:
            children[parent].add(i)
    root_direct = {i: set() for i in ids}
    root_indirect = {i: set() for i in ids}
    for i, (root, parent, depth) in plan.items():
        if parent is None:
            continue
        if depth == 1:
            root_direct[root].add(i)
        else:
            root_indirect[root].add(i)
    previous = {i: cluster._tvct_state(i) for i in ids}
    count_ch_to_cm = 0
    for i in sorted(ids):
        st = cluster.veh_table.values(i)
        old_role, old_root = previous[i]
        new = plan.get(i)
        role = ('SA' if new is None else 'CH' if new[1] is None else 'MEM')
        root = None if new is None else new[0]
        parent = None if new is None else new[1]
        depth = None if new is None else new[2]
        if old_role == 'CH' and role == 'MEM':
            count_ch_to_cm += 1

        # Preserve the existing linked-list episode *format* used elsewhere in
        # the project, not just the newer stability_episodes metric.
        record = st.get('cluster_record')
        if record is not None and getattr(record, 'tail', None) is not None:
            current_tail = record.tail
            value = current_tail.value
            same_cm = (role == 'MEM' and old_role == 'MEM' and
                       old_root == root and current_tail.key == root and
                       value.get('is_ch') is False and value.get('timer') is not None)
            if same_cm:
                if value.get('start_time') is not None and (
                        value['start_time'] + value['timer'] - 1 < cluster.time):
                    value['timer'] += 1
                parent_list = value.setdefault('secondary_ch', [])
                relay = None if parent == root else parent
                if not parent_list or parent_list[-1] != relay:
                    parent_list.append(relay)
            elif role == 'MEM' and (old_role != role or old_root != root):
                new_value = {'is_ch': False, 'secondary_ch': [None if parent == root else parent],
                             'start_time': cluster.time, 'ef': None, 'timer': 1,
                             'interrupt': []}
                if current_tail.key is None and value.get('timer') is None and (
                        value.get('start_time') is None and not value.get('is_ch')):
                    current_tail.key = root
                    current_tail.value = new_value
                else:
                    record.append(root, new_value)
            elif role == 'CH' and old_role != 'CH':
                if (current_tail.key is None and value.get('timer') is None and
                        value.get('start_time') is None and not value.get('is_ch')):
                    value['is_ch'] = True
                else:
                    record.append(None, {'is_ch': True, 'secondary_ch': [],
                                         'start_time': None, 'ef': None,
                                         'timer': None, 'interrupt': []})
            elif role == 'SA' and old_role != 'SA':
                record.append(None, {'is_ch': False, 'secondary_ch': [],
                                     'start_time': None, 'ef': None,
                                     'timer': None, 'interrupt': []})

        st['cluster_head'] = role == 'CH'
        st['primary_ch'] = root if role == 'MEM' else None
        st['secondary_ch'] = parent if role == 'MEM' and parent != root else None
        st['root_ch'] = root
        st['parent_node'] = parent
        st['parent_id'] = parent
        st['hop_count'] = depth
        st['role'] = ('CH' if role == 'CH' else
                      ('RN' if children[i] else 'CM') if role == 'MEM' else 'SA')
        st['root_connection_valid'] = role == 'CH' or (role == 'MEM' and i not in grace)
        st['direct_children'] = children[i]
        st['cluster_members'] = root_direct[i] if role == 'CH' else set()
        st['sub_cluster_members'] = (root_indirect[i] if role == 'CH'
                                     else set(children[i]))
        st['gates'] = {}
        st['gate_chs'] = set()
        st['path_bottleneck'] = None if role != 'CH' else 0.0
        st['committed_link_score'] = None
        st['current_root_cost'] = None
        st['current_parent_cost'] = None
        if role == 'CH' and old_role != 'CH':
            st['start_ch_zone'] = st['macro_zone']
        elif role != 'CH':
            st['start_ch_zone'] = None
        if role == 'MEM' and root != old_root:
            st['priority_ch'] = root
            st['priority_counter'] = getattr(config, 'priority_counter', 0)
        elif role == 'CH':
            st['priority_ch'] = None
        if role != old_role or root != old_root:
            st['counter'] = getattr(config, 'counter', 0)

    cluster.dhcv_ch_to_cm_changes = getattr(cluster, 'dhcv_ch_to_cm_changes', 0) + count_ch_to_cm
    cluster.all_chs = {i for i,(r,p,d) in plan.items() if p is None} | cluster.bus_table.ids()
    cluster.stand_alone = ids - set(plan)
    for zone in cluster.zone_ch:
        cluster.zone_ch[zone].clear()
        cluster.zone_stand_alone[zone].clear()
    for i in ids:
        st = cluster.veh_table.values(i)
        z = st['macro_zone']
        if st['cluster_head']:
            cluster.zone_ch[z].add(i)
        elif i in cluster.stand_alone:
            cluster.zone_stand_alone[z].add(i)
    for i in cluster.bus_table.ids():
        cluster.zone_ch[cluster.bus_table.values(i)['macro_zone']].add(i)
    cluster._mmzca_log_tvct_transitions(previous, event_reason)
