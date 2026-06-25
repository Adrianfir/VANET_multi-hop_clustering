"""
This .py file is for adding arguments to argparse
"""
__author__: str = "Pouya 'Adrian' Firouzmakan"

import numpy
import numpy as np
import argparse
import pathlib
import xml.dom.minidom


class Inputs:
    def __init__(self):
        # Constants that we need to pass as arguments
        trace_path = str(pathlib.Path(__file__).parent.parent.parent.absolute().
                         joinpath('traffic_data', 'final_data_Richmondhill_smallsize', 'sumoTrace.xml'))
        sumo_edge_path = str(pathlib.Path(__file__).parent.parent.parent.absolute().
                             joinpath('traffic_data', 'final_data_Richmondhill_smallsize', 'osm.net.xml'))
        sumo_node_path = str(pathlib.Path(__file__).parent.parent.parent.absolute().
                             joinpath('traffic_data', 'final_data_Richmondhill_smallsize', 'osm_bbox.osm.xml'))
        sumo_trace = xml.dom.minidom.parse(trace_path)
        sumo_edge = xml.dom.minidom.parse(sumo_edge_path)
        sumo_node = xml.dom.minidom.parse(sumo_node_path)
        fcd = sumo_trace.documentElement
        times = fcd.getElementsByTagName('timestep')
        area = dict(min_lat=43.586568,
                    min_long=-79.540771,
                    max_lat=44.012923,
                    max_long=-79.238069)
        alpha = float()
        alpha_micro = 0.5
        alpha_meso = 0.75
        alpha_macro= 1
        veh_trans_range = 300
        bus_trans_range = 800
        start_time = 1600
        iter = 60
        counter = 3
        priority_counter = 3
        map_zoom = 15.3
        center_loc = [43.869846, -79.443523]
        fps = 5
        weights_s = np.array([1.0, 0.0, 0.0])       # direction's angle, speed, distance for single-hop
        weights_m = np.array([0.6, 0.2, 0.2])       # direction's angle, speed, distance for multi-hop


        # ---------- New Values I need for new_mmzca ----------

        h_max = 2
        root_hysteresis_delta = 0.05
        parent_hysteresis_delta = 0.05

        micro_zone_max = 1
        meso_zone_max = 1
        macro_zone_max = 1

        w_root_meso_dir = 0.3
        w_root_macro_dir = 0.6
        w_root_macro_dist = 1
        w_root_hop = 0.10
        w_root_load = 0.05
        w_root_prev = 0.10
        w_root_dru = 0.05

        w_parent_micro_dir = 0.7
        w_parent_speed = 0.10
        w_root_speed = 0.0      # I think the root level speed makes things worse
        w_parent_dist = 0.10
        w_parent_hop = 0.10
        w_parent_load = 0.00

        w_pch_degree = 0.35
        w_pch_micro_dir = 0.30
        w_pch_dist = 0.15
        w_pch_micro_zone = 0.10
        w_pch_meso_zone = 0.10

        w_rn_sa_gain = 0.40
        w_rn_root_gain = 0.25
        w_rn_coherence = 0.20
        w_rn_hop_penalty = 0.10
        w_rn_load_penalty = 0.05

        max_rn_members = 20
        max_ch_members = 35

        lambda_rn = 1

        # ---------- K-Medoids----------
        metoids_target_cluster_size = 4
        metoids_max_cosine_distance = 2
        metoids_switch_margin = 10




        parser = argparse.ArgumentParser()
        parser.add_argument('--area', type=dict, default=area,
                            help='this argument is the latitudes and longitudes of the understudied area')
        parser.add_argument('--n_cars', type=int, default=8000,
                            help='this is an assumption regarding the number of cars in order to create a HashTabel')
        parser.add_argument('--sumo_trace', type=xml.dom.minidom.Document, default=sumo_trace,
                            help='This is the sumo_trace file that includes all the data we need from the traffic')
        parser.add_argument('--sumo_edge', type=xml.dom.minidom.Document, default=sumo_edge,
                            help='This is the sumo_trace file that includes the edges (lanes) information')
        parser.add_argument('--sumo_node', type=xml.dom.minidom.Document, default=sumo_node,
                            help='This is the sumo_trace file that includes the nodes information of sumo net')
        parser.add_argument('--fcd', type=xml.dom.minidom.Element, default=fcd,
                            help='Floating Car Data (FCD) from sumoTrace.xml file')
        parser.add_argument('--times', type=xml.dom.minidom.NodeList, default=times,
                            help='includes data for all seconds')
        parser.add_argument('--alpha', type=float, default=alpha,
                            help='this is the regularization coefficient to change the size of the zones')
        parser.add_argument('--alpha_micro', type=float, default=alpha_micro,
                            help='this is the regularization coefficient to create micro zones')
        parser.add_argument('--alpha_meso', type=float, default=alpha_meso,
                            help='this is the regularization coefficient to create meso zones')
        parser.add_argument('--alpha_macro', type=float, default=alpha_macro,
                            help='this is the regularization coefficient to create macro zones')

        parser.add_argument('--veh_trans_range', type=int, default=veh_trans_range,
                            help='this is the transmission range of vehicles considered in this project and it can '
                                 'be up to 2000')
        parser.add_argument('--bus_trans_range', type=int, default=bus_trans_range,
                            help='this is the transmission range of buses considered in this project and it can '
                                 'be up to 2000')
        parser.add_argument('--start_time', type=int, default=start_time,
                            help='This is the time that the initial values would be extract from sumo_trace.xml file')
        parser.add_argument('--counter', type=int, default=counter,
                            help='This is the a counter for vehicle to make themselves as CH if they can not'
                                 ' find any Ch or nearby stand-alone vehicles to create a cluster')
        parser.add_argument('--priority_counter', type=int, default=priority_counter,
                            help='This is the a counter for vehicle to join same cluster through sub_chs after leaving '
                                 'that cluster')
        parser.add_argument('--map_zoom', type=float, default=map_zoom,
                            help='This is the amount to have a specific zoom on the map')
        parser.add_argument('--center_loc', type=float, default=center_loc,
                            help='The specific center location of the map for saving images and make slide-show')
        parser.add_argument('--fps', type=float, default=fps, help='frame per second')
        parser.add_argument('--iter', type=int, default=iter, help='number of intervals to run')
        parser.add_argument('--weights_s', type=numpy.ndarray, default=weights_s, help='weights used for '
                                                                                       'single-hop clustering')
        parser.add_argument('--weights_m', type=numpy.ndarray, default=weights_m, help='weights used for '
                                                                                       'multi-hop clustering')



        # ---------- New Values I need for new_mmzca ----------

        parser.add_argument('--h_max', type=int, default=h_max,
                            help='maximum number of hops from a vehicle to the root CH in multi-hop clustering')

        parser.add_argument('--root_hysteresis_delta', type=float, default=root_hysteresis_delta,
                            help='minimum improvement required to switch from the current root CH to a new root CH')

        parser.add_argument('--parent_hysteresis_delta', type=float, default=parent_hysteresis_delta,
                            help='minimum improvement required to switch from the current parent/anchor to a new parent/anchor')

        parser.add_argument('--micro_zone_max', type=int, default=micro_zone_max,
                            help='maximum allowed micro-zone grid distance for local parent feasibility')

        parser.add_argument('--meso_zone_max', type=int, default=meso_zone_max,
                            help='maximum allowed meso-zone grid distance for branch-level feasibility')

        parser.add_argument('--macro_zone_max', type=int, default=macro_zone_max,
                            help='maximum allowed macro-zone grid distance for root-cluster feasibility')

        parser.add_argument('--w_root_meso_dir', type=float, default=w_root_meso_dir,
                            help='weight of meso-level directional similarity term in root cost')

        parser.add_argument('--w_root_macro_dir', type=float, default=w_root_macro_dir,
                            help='weight of macro-level directional similarity term in root cost')

        parser.add_argument('--w_root_macro_dist', type=float, default=w_root_macro_dist,
                            help='weight of macro-zone distance term in root cost')

        parser.add_argument('--w_root_hop', type=float, default=w_root_hop,
                            help='weight of hop-depth term in root cost')

        parser.add_argument('--w_root_load', type=float, default=w_root_load,
                            help='weight of root-load term in root cost')

        parser.add_argument('--w_root_prev', type=float, default=w_root_prev,
                            help='weight of previous-root preference term in root cost')

        parser.add_argument('--w_root_dru', type=float, default=w_root_dru,
                            help='weight of DRU-root preference term in root cost')

        parser.add_argument('--w_parent_micro_dir', type=float, default=w_parent_micro_dir,
                            help='weight of micro-level directional similarity term in parent cost')

        parser.add_argument('--w_parent_speed', type=float, default=w_parent_speed,
                            help='weight of speed mismatch term in parent cost')

        parser.add_argument('--w_root_speed', type=float, default=w_root_speed,
                            help='normalized speed mismatch weight for root cost')

        parser.add_argument('--w_parent_dist', type=float, default=w_parent_dist,
                            help='weight of Euclidean distance term in parent cost')

        parser.add_argument('--w_parent_hop', type=float, default=w_parent_hop,
                            help='weight of parent hop-depth term in parent cost')

        parser.add_argument('--w_parent_load', type=float, default=w_parent_load,
                            help='weight of parent-load term in parent cost')

        parser.add_argument('--w_pch_degree', type=float, default=w_pch_degree,
                            help='weight of unresolved-SA neighborhood degree term in PCH score')

        parser.add_argument('--w_pch_micro_dir', type=float, default=w_pch_micro_dir,
                            help='weight of micro-level directional similarity term in PCH score')

        parser.add_argument('--w_pch_dist', type=float, default=w_pch_dist,
                            help='weight of Euclidean distance term in PCH score')

        parser.add_argument('--w_pch_micro_zone', type=float, default=w_pch_micro_zone,
                            help='weight of micro-zone compatibility term in PCH score')

        parser.add_argument('--w_pch_meso_zone', type=float, default=w_pch_meso_zone,
                            help='weight of meso-zone compatibility term in PCH score')

        parser.add_argument('--w_rn_sa_gain', type=float, default=w_rn_sa_gain,
                            help='weight of unresolved-SA absorption gain term in RN utility')

        parser.add_argument('--w_rn_root_gain', type=float, default=w_rn_root_gain,
                            help='weight of root-reachability gain term in RN utility')

        parser.add_argument('--w_rn_coherence', type=float, default=w_rn_coherence,
                            help='weight of zoning and directional coherence term in RN utility')

        parser.add_argument('--w_rn_hop_penalty', type=float, default=w_rn_hop_penalty,
                            help='weight of hop-depth penalty term in RN utility')

        parser.add_argument('--w_rn_load_penalty', type=float, default=w_rn_load_penalty,
                            help='weight of load penalty term in RN utility')

        parser.add_argument('--max_rn_members', type=int, default=max_rn_members,
                            help='maximum allowed number of downstream members for a relay node')

        parser.add_argument('--max_ch_members', type=int, default=max_ch_members,
                            help='maximum allowed number of direct cluster members for a cluster head')

        parser.add_argument('--lambda_rn', type=float, default=lambda_rn,
                            help='weight for combining local RN parent cost with RN reported join score')


        # ---------------------- K-Medoids ----------------------

        parser.add_argument('--metoids_target_cluster_size', type=dict, default=metoids_target_cluster_size,
                            help='for improving k-medoids')
        parser.add_argument('--metoids_max_cosine_distance', type=dict, default=metoids_max_cosine_distance,
                            help='for improving k-medoids')
        parser.add_argument('--metoids_switch_margin', type=dict, default=metoids_switch_margin,
                            help='for improving k-medoids')



        self.parser = parser

    def get_parser(self):
        """
        this methods can be used on order to return the parser and be used in config file
        :return: it returns the parser
        """
        return self.parser.parse_args()