"""
This is the utils file including the small functions to add more functions for new_mmzca
"""
__author__: str = "Pouya 'Adrian' Firouzmakan"
__all__ = []



import numpy as np
import haversine as hs
import utils.util as util

def is_bus(node_id):
    return isinstance(node_id, str) and node_id.startswith('bus')


def is_vehicle(node_id):
    return isinstance(node_id, str) and node_id.startswith('veh')


def get_node_table(node_id, veh_table, bus_table):
    return bus_table if is_bus(node_id) else veh_table


def get_node_state(node_id, veh_table, bus_table):
    table = get_node_table(node_id, veh_table, bus_table)
    return table.values(node_id)

def sync_node_cluster_fields(node_id, veh_table, bus_table):
    """
    Synchronize compatibility fields:
      - root_ch
      - parent_node
      - hop_count
    with the legacy fields:
      - cluster_head
      - primary_ch
      - secondary_ch
    """
    table = get_node_table(node_id, veh_table, bus_table)
    st = table.values(node_id)

    if is_bus(node_id):
        st['root_ch'] = node_id
        st['parent_node'] = None
        st['hop_count'] = 0
        return

    if st['cluster_head'] is True:
        st['root_ch'] = node_id
        st['parent_node'] = None
        st['hop_count'] = 0
        return

    if st['primary_ch'] is None:
        st['root_ch'] = None
        st['parent_node'] = None
        st['hop_count'] = None
        return

    st['root_ch'] = st['primary_ch']
    if st['secondary_ch'] is None:
        st['parent_node'] = st['primary_ch']
        if st.get('hop_count') is None or st.get('hop_count') < 1:
            st['hop_count'] = 1
    else:
        st['parent_node'] = st['secondary_ch']
        if st.get('hop_count') is None or st.get('hop_count') < 2:
            st['hop_count'] = 2


def reset_cluster_fields(node_id, veh_table, bus_table):
    table = get_node_table(node_id, veh_table, bus_table)
    st = table.values(node_id)

    if is_vehicle(node_id):
        st['primary_ch'] = None
        st['secondary_ch'] = None
        st['root_ch'] = None
        st['parent_node'] = None
        st['hop_count'] = None
        st['current_root_cost'] = None
        st['current_parent_cost'] = None

def is_rn(node_id, veh_table, bus_table):
    """
    RN = non-CH attached vehicle with at least one downstream sub-member.
    """
    if is_bus(node_id):
        return False
    st = veh_table.values(node_id)
    return (st['cluster_head'] is False and
            st['primary_ch'] is not None and
            len(st.get('sub_cluster_members', set())) > 0)

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

    direct_members = len(st.get('cluster_members', set()))
    downstream = len(st.get('sub_cluster_members', set()))
    return direct_members + downstream

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
    Generalized attachment wrapper.
    Uses old add_member/add_sub_member underneath, but updates generalized fields too.
    """
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
        parent_hop = get_hop_count(parent_id, veh_table, bus_table)
        veh_table.values(veh_id)['root_ch'] = root_id
        veh_table.values(veh_id)['parent_node'] = parent_id
        veh_table.values(veh_id)['hop_count'] = None if parent_hop is None else parent_hop + 1

    veh_table.values(veh_id)['primary_ch'] = root_id
    veh_table.values(veh_id)['secondary_ch'] = None if parent_id == root_id else parent_id
    veh_table.values(veh_id)['current_root_cost'] = root_cost_value
    veh_table.values(veh_id)['current_parent_cost'] = parent_cost_value
    veh_table.values(veh_id)['unresolved_start_macro_zone'] = None

    return bus_table, veh_table, stand_alone, zone_stand_alone

def detach_from_current_cluster(veh_id, veh_table, bus_table, config,
                                stand_alone, zone_stand_alone):
    """
    Generic detach wrapper.
    """
    st = veh_table.values(veh_id)
    root_id = st.get('primary_ch')
    parent_id = st.get('secondary_ch')

    if root_id is None:
        return veh_table, bus_table, stand_alone, zone_stand_alone

    if parent_id is None:
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

    if is_bus(node_id):
        return None
    if st['cluster_head'] is True:
        return None
    if st.get('parent_node') is not None:
        return st['parent_node']
    if st.get('secondary_ch') is not None:
        return st['secondary_ch']
    return st.get('primary_ch')

def candidate_type(node_id, veh_table, bus_table):
    if is_bus(node_id):
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



