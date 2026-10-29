"""MMZCA experiment configuration (active simulation settings only).

The values below reproduce the USER-SUPPLIED large-density, TR=300 m,
200-tick experiment.  Configuration parameters used only by retired
SMZCA-style helper paths have been removed from the active argument parser.

Units: zone side lengths in kilometres; transmission ranges in metres;
SUMO trace timestamps, waiting times and kernel ages in one-second ticks.

Related manuscript sections:
  - Communication and Hierarchical Association
  - Multi-Scale Spatial and Mobility Abstraction
  - Root-Centred Sparse Micro-Zone Kernel
  - Root-Connected Parent Evaluation and Selection
  - Decentralized Formation, Repair, and Properties
"""

__author__: str = "Pouya 'Adrian' Firouzmakan"

import argparse
import pathlib
import xml.dom.minidom

import numpy as np


class Inputs:
    def __init__(self):
        # ==================================================================
        # 1. SUMO dataset and observation window
        # ==================================================================
        # Select the traffic-density dataset; this is a file-selection option,
        # not an optimization weight or vehicle communication parameter.
        data_density = 'largesize'
        traffic_folder = (
            pathlib.Path(__file__).parent.parent.parent.absolute()
            / 'traffic_data'
            / ('final_data_Richmondhill_' + data_density)
        )

        # The no-bus/RSU trace contains ordinary vehicles only.  Bus support
        # remains configured because MMZCA also supports DRU/bus-rooted runs.
        sumo_trace = xml.dom.minidom.parse(str(traffic_folder / 'sumoTrace_no_bus_and_rsu.xml'))
        sumo_edge = xml.dom.minidom.parse(str(traffic_folder / 'osm.net.xml'))
        sumo_node = xml.dom.minidom.parse(str(traffic_folder / 'osm_bbox.osm.xml'))

        # Study rectangle, in geographical latitude/longitude (degrees).
        # Used to generate the common, aligned micro/meso/macro zoning grids.
        area = dict(
            min_lat=43.586568,
            min_long=-79.540771,
            max_lat=44.012923,
            max_long=-79.238069,
        )

        # SUMO timestep index at which the observation begins.  'iter' is the
        # number of successive one-second updates after initialization.
        start_time = 1600
        iter = 200

        # Hash-table sizing parameter used by DataTable initialization; it is
        # a storage setting, NOT the number of active vehicles in the trace.
        n_cars = 8000

        # ==================================================================
        # 2. Zoning, radio connectivity, and root-connected hierarchy
        # ==================================================================
        # Side lengths alpha_mu, alpha_m, alpha_M of the three zoning grids,
        # in kilometres.  Micro cells encode root-cluster occupancy; meso/
        # macro cells provide trajectory features.  Macro-zone transitions
        # also trigger reversion of memberless ordinary CHs to SA.
        alpha_micro = 0.1
        alpha_meso = 0.5
        alpha_macro = 1

        # Transmission ranges in metres.  A physical parent-child link (i,j)
        # is feasible only if its distance <= min(TR_i, TR_j), per Eq. (comm).
        veh_trans_range = 300
        bus_trans_range = 300

        # Maximum number of parent links from a root CH to any CM:
        # d_i = d_parent + 1 <= H_max.  A CM at H_max cannot admit children.
        h_max = 4

        # Maximum number of direct children accepted by a root CH / CM relay.
        # Admission also requires a valid root path, depth, and physical link.
        max_ch_members = 35
        max_rn_members = 20

        # ==================================================================
        # 3. Physical-link score L_ij and affinity-aware parent evaluation
        # ==================================================================
        # MZOTD_ij = omega_m * theta^meso_ij + omega_M * theta^macro_ij.
        # These weights sum to one; lower angular dissimilarity is preferred.
        omega_meso = 0.5
        omega_macro = 0.5

        # Raw relative-feature weights for L_ij, in the order:
        # [MZOTD/pi, normalized speed difference, distance / reciprocal TR].
        # The current selector NORMALIZES these raw weights to sum to one.
        # Thus [0.4, 0.2, 0.2] is effectively [0.5, 0.25, 0.25]; retain the
        # supplied values to preserve the existing experiment's behaviour.
        weights_m = np.array([0.4, 0.2, 0.2])

        # Positive speed-normalization floor v_0 in metres per second:
        # |speed_i - speed_j| / max(speed_i, speed_j, v_0).
        v0 = 1.0

        # Physical-link fraction in L_aff(i,j) = iota*L_ij +
        # (1-iota)*A_tilde_{i|root(j)}^micro.  Lower scores are better.
        iota = 0.5

        # Parent-selection weights w_G on:
        # [candidate-path bottleneck Q_i^(j), affinity-aware link L_aff,
        #  candidate depth (d_j+1)/H_max], respectively.
        # The same w_G is used for the physical-only missing-kernel fallback.
        weights_G = np.array([0.4, 0.4, 0.2])

        # Treat scores within this numerical tolerance as ties, using physical
        # L_ij, depth, then random contention as the manuscript specifies.
        selection_tolerance = 1e-8

        # Delta_B: admissible score premium for selecting a bus-rooted parent.
        # It applies only after normal link, capacity, and depth validation.
        delta_bus = 0.05

        # ==================================================================
        # 4. Root-preserving parent switching and cluster formation
        # ==================================================================
        # Delta_P: valid CMs may switch to a DIFFERENT PARENT OF THE SAME ROOT
        # only if G(current) - G(candidate) > Delta_P (plus numerical tolerance).
        # No voluntary root switching is performed; consequently there is
        # intentionally NO root_hysteresis_delta setting.
        parent_hysteresis_delta = 0.05

        # Lambda in the PCH formation section: an unresolved SA attempts to
        # join a feasible existing root for this many 1-s beacon periods before
        # participating in the local PCH election (SA degree, then mean L_ij).
        # An isolated eligible SA may become CH after the waiting period.
        counter = 3

        # Number of 1-s ticks for short-lived previous-root preference after
        # disconnection.  This is a REPAIR priority, not root-switching
        # hysteresis, and cannot override physical feasibility or hop limits.
        priority_counter = 3

        # ==================================================================
        # 5. Versioned root-centred sparse micro-zone kernel
        # ==================================================================
        # M midpoint-quantile samples for EACH of the two diagonal projection
        # distributions (rho=u+v and sigma=u-v) in S_h^micro.
        kernel_quantile_samples = 8

        # T_ker: a kernel descriptor is usable only while its root incarnation
        # matches and its age is between zero and T_ker beacon ticks.
        kernel_ttl = 10

        # Maximum scheduled interval between successive root snapshots;
        # root occupancy/anchor changes can trigger earlier regeneration.
        kernel_refresh_period = 5

        # Discrete one-hop message latency for occupancy reports and sketch
        # propagation, in 1-s beacon ticks.  Prevents instantaneous global
        # dissemination of information across several relays.
        kernel_message_delay_ticks = 1

        # Upper bound on any committed parent link, in METRES, for the
        # root-centred kernel coverage radius:
        # r_K = ceil((H_max-1)*R_edge_max/(1000*alpha_micro)), k=2*r_K+1.
        # Here H_max=3, range=300 m, alpha_micro=0.1 km => r_K=6, k=13.
        kernel_max_edge_range_m = max(veh_trans_range, bus_trans_range)

        # ==================================================================
        # 6. Optional visualisation parameters (used by DataTable.show_graph)
        # ==================================================================
        map_zoom = 15.3         # Interactive map's initial zoom level.
        center_loc = [43.869846, -79.443523]  # Map centre [latitude, longitude].

        # ==================================================================
        # Argument parser: retain the original Configs().config API
        # ==================================================================
        parser = argparse.ArgumentParser()

        parser.add_argument('--area', type=dict, default=area,
                            help='Geographical bounds for all zone grids (degrees).')
        parser.add_argument('--n_cars', type=int, default=n_cars,
                            help='Hash-table sizing; not the simulated vehicle count.')
        parser.add_argument('--sumo_trace', type=xml.dom.minidom.Document, default=sumo_trace,
                            help='Parsed SUMO FCD vehicle trace.')
        parser.add_argument('--sumo_edge', type=xml.dom.minidom.Document, default=sumo_edge,
                            help='Parsed SUMO road-edge network.')
        parser.add_argument('--sumo_node', type=xml.dom.minidom.Document, default=sumo_node,
                            help='Parsed SUMO road-node network.')
        parser.add_argument('--start_time', type=int, default=start_time,
                            help='Initial SUMO timestep index.')
        parser.add_argument('--iter', type=int, default=iter,
                            help='Number of one-second updates after initialisation.')

        parser.add_argument('--alpha_micro', type=float, default=alpha_micro,
                            help='Micro-zone side length alpha_mu (km): kernel occupancy grid.')
        parser.add_argument('--alpha_meso', type=float, default=alpha_meso,
                            help='Meso-zone side length alpha_m (km): MZOTD motion.')
        parser.add_argument('--alpha_macro', type=float, default=alpha_macro,
                            help='Macro-zone side length alpha_M (km): MZOTD and CH demotion.')
        parser.add_argument('--veh_trans_range', type=int, default=veh_trans_range,
                            help='Ordinary vehicle transmission range (m).')
        parser.add_argument('--bus_trans_range', type=int, default=bus_trans_range,
                            help='Bus transmission range (m).')
        parser.add_argument('--h_max', type=int, default=h_max,
                            help='Maximum permitted root-to-member hop depth H_max.')
        parser.add_argument('--max_ch_members', type=int, default=max_ch_members,
                            help='Maximum direct-child capacity of a root CH.')
        parser.add_argument('--max_rn_members', type=int, default=max_rn_members,
                            help='Maximum direct-child capacity of a CM relay.')

        parser.add_argument('--omega_meso', type=float, default=omega_meso,
                            help='Meso angular weight in MZOTD (omega_m).')
        parser.add_argument('--omega_macro', type=float, default=omega_macro,
                            help='Macro angular weight in MZOTD (omega_M).')
        parser.add_argument('--weights_m', nargs=3, type=float, default=weights_m,
                            help='Raw physical-link weights [direction, speed, distance]; normalized internally.')
        parser.add_argument('--v0', type=float, default=v0,
                            help='Positive speed mismatch normalization floor (m/s).')
        parser.add_argument('--iota', type=float, default=iota,
                            help='Physical-link fraction in L_aff; remainder weights cluster affinity.')
        parser.add_argument('--weights_G', nargs=3, type=float, default=weights_G,
                            help='Parent score weights [Q, L_aff, depth/H_max].')
        parser.add_argument('--selection_tolerance', type=float, default=selection_tolerance,
                            help='Numeric tolerance for parent-score ties and hysteresis.')
        parser.add_argument('--delta_bus', type=float, default=delta_bus,
                            help='Score premium Delta_B allowed for feasible bus-rooted parents.')

        parser.add_argument('--parent_hysteresis_delta', type=float, default=parent_hysteresis_delta,
                            help='Delta_P: minimum improvement for elective same-root parent switching.')
        parser.add_argument('--counter', type=int, default=counter,
                            help='Lambda: SA waiting period in one-second beacon ticks.')
        parser.add_argument('--priority_counter', type=int, default=priority_counter,
                            help='Previous-root recovery preference duration (ticks).')

        parser.add_argument('--kernel_quantile_samples', type=int, default=kernel_quantile_samples,
                            help='M diagonal midpoint-quantile samples per kernel projection.')
        parser.add_argument('--kernel_ttl', type=int, default=kernel_ttl,
                            help='T_ker: maximum usable kernel snapshot age (ticks).')
        parser.add_argument('--kernel_refresh_period', type=int, default=kernel_refresh_period,
                            help='Maximum scheduled snapshot refresh interval (ticks).')
        parser.add_argument('--kernel_message_delay_ticks', type=int, default=kernel_message_delay_ticks,
                            help='Per-hop delay for kernel and histogram messages (ticks).')
        parser.add_argument('--kernel_max_edge_range_m', type=float, default=kernel_max_edge_range_m,
                            help='Safe parent-link length bound R_edge_max for kernel radius (m).')

        parser.add_argument('--map_zoom', type=float, default=map_zoom,
                            help='Optional visualization map zoom.')
        parser.add_argument('--center_loc', type=float, default=center_loc,
                            help='Optional visualization map centre [latitude, longitude].')

        self.parser = parser

    def get_parser(self):
        """Return the existing argparse Namespace consumed by Configs()."""
        return self.parser.parse_args()
