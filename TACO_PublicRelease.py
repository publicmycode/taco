import os, sys, pickle
import argparse, sys

import numpy as np
import pandas as pd
import networkx as nx

import random
import math
import time
import copy


parser = argparse.ArgumentParser()
#parser.add_argument('-input_topology', help=' : Please set the topology name') 
parser.add_argument('input_topology', help=' : Please set the topology name') 
parser.add_argument('input_n_flows', type=int, help=' : Please set the number of flows')
parser.add_argument('input_seed_G', type=int, help=' : Please set the random seed for graphs')
parser.add_argument('input_seed_F', type=int, help=' : Please set the random seed for flows')
parser.add_argument('input_n_pop', type=int, help=' : Please set the pop size') 
parser.add_argument('input_n_elite', type=int, help=' : Please set the number of elite') 
parser.add_argument('input_n_invader', type=int, help=' : Please set the number of invader') 
parser.add_argument('input_mutation', type=float, help=' : Please set the mutation size (should be 0 < x < n_flows | 0~2 or util. based are recommended)') 
args = parser.parse_args()


##############################################################################################################################################################
def makeG(g_seed):
    Graphs = {}

    # CEV
    Graphs['CEV'] = {
        'G': nx.DiGraph(),
        'end_station': [],
        'nodes': [x for x in range(46)]
    }
    Graphs['CEV']['G'].add_edges_from([
        (0, 31), (1, 31), (2, 31), (3, 32), (4, 32), (5, 33), (6, 33), (7, 33), (8, 33), (9, 33),
        (10, 34), (11, 34), (12, 35), (13, 35), (14, 36), (15, 36), (16, 36), (17, 37), (18, 37), (19, 38), (20, 38),
        (21, 39), (22, 39), (23, 40), (24, 40), (25, 41), (26, 41), (27, 42), (28, 42), (29, 42), (30, 43),
        (31, 0), (31, 1), (31, 2), (31, 35), (31, 43),
        (32, 3), (32, 4), (32, 35), (32, 43),
        (33, 5), (33, 6), (33, 7), (33, 8), (33, 9), (33, 35), (33, 43),
        (34, 10), (34, 11), (34, 35), (34, 43),
        (35, 12), (35, 13), (35, 31), (35, 32), (35, 33), (35, 34), (35, 36), (35, 45),
        (36, 14), (36, 15), (36, 16), (36, 35), (36, 37), (36, 39), (36, 44), (36, 45),
        (37, 17), (37, 18), (37, 36), (37, 38),
        (38, 19), (38, 20), (38, 37), (38, 44),
        (39, 21), (39, 22), (39, 36), (39, 42), 
        (40, 23), (40, 24), (40, 41), (40, 44), 
        (41, 25), (41, 26), (41, 40), (41, 42), 
        (42, 27), (42, 28), (42, 29), (42, 39), (42, 41), (42, 43), (42, 44), (42, 45), 
        (43, 30), (43, 31), (43, 32), (43, 33), (43, 34), (43, 42), (43, 45), 
        (44, 36), (44, 38), (44, 40), (44, 42), 
        (45, 35), (45, 36), (45, 42), (45, 43), 
    ], weight=1) 
    for i in range(31):
        Graphs['CEV']['end_station'].append(i)

    # DFC
    Graphs['DFC'] = {
    'G': nx.DiGraph(),
    'nodes': [x for x in range(18)]
    }
    Graphs['DFC']['G'].add_edges_from([
        (0, 1), (0, 2), (0, 4), (0, 6), (0, 12),
        (1, 0), (1, 3), (1, 4), (1, 7), (1, 13),
        (2, 0), (2, 3), (2, 5), (2, 8), (2, 14),
        (3, 1), (3, 2), (3, 5), (3, 9), (3, 15),
        (4, 0), (4, 1), (4, 5), (4, 10), (4, 16),
        (5, 2), (5, 3), (5, 4), (5, 11), (5, 17),
        (6, 0), (6, 7), (6, 8), (6, 10), (6, 12),
        (7, 1), (7, 6), (7, 9), (7, 10), (7, 13),
        (8, 2), (8, 6), (8, 9), (8, 11), (8, 14),
        (9, 3), (9, 7), (9, 8), (9, 11), (9, 15),
        (10, 4), (10, 6), (10, 7), (10, 11), (10, 16),
        (11, 5), (11, 8), (11, 9), (11, 10), (11, 17),
        (12, 0), (12, 6), (12, 13), (12, 14), (12, 16),
        (13, 1), (13, 7), (13, 12), (13, 15), (13, 16),
        (14, 2), (14, 8), (14, 12), (14, 15), (14, 17),
        (15, 3), (15, 9), (15, 13), (15, 14), (15, 17),
        (16, 4), (16, 10), (16, 12), (16, 13), (16, 17),
        (17, 5), (17, 11), (17, 14), (17, 15), (17, 16)
    ], weight=1) 

    # ERG
    node_number = 15
    input_edges = []
    graph = nx.erdos_renyi_graph(node_number, 0.3, seed = g_seed)
    for (u,v,w) in graph.edges(data=True):
        edge = (u, v)
        reverse_edge = (v, u)
        input_edges.append(edge)
        input_edges.append(reverse_edge)
    Graphs['ERG'] = {
    'G': nx.DiGraph(),
    'nodes': [x for x in range(node_number)]
    }
    Graphs['ERG']['G'].add_edges_from(input_edges, weight=1) 

    # BAG
    node_number = 30
    input_edges = []
    graph = nx.barabasi_albert_graph(node_number, 3, seed = g_seed)
    for (u,v,w) in graph.edges(data=True):
        edge = (u, v)
        reverse_edge = (v, u)
        input_edges.append(edge)
        input_edges.append(reverse_edge)
    Graphs['BAG'] = {
    'G': nx.DiGraph(),
    'nodes': [x for x in range(node_number)]
    }
    Graphs['BAG']['G'].add_edges_from(input_edges, weight=1) 

    # RRG
    node_number = 15
    input_edges = []
    graph = nx.random_regular_graph(d=4, n=node_number, seed = g_seed)
    for (u,v,w) in graph.edges(data=True):
        edge = (u, v)
        reverse_edge = (v, u)
        input_edges.append(edge)
        input_edges.append(reverse_edge)
    Graphs['RRG'] = {
    'G': nx.DiGraph(),
    'nodes': [x for x in range(node_number)]
    }
    Graphs['RRG']['G'].add_edges_from(input_edges, weight=1) 

    return Graphs


def makeFlowSet(st1, st2, st3, graph, graph_type, f_seed):
    flow_RTSF = {}
    flow_RTSF['ST1'] = []
    flow_RTSF['ST2'] = []
    flow_RTSF['ST3'] = []

    rng=random.Random(f_seed)
    origin_link_bandwidth = 1.25e+7 

    node_list = []
    if graph_type == 0:
        node_list = copy.deepcopy(graph['end_station'])
    elif graph_type == 1:
        node_list = copy.deepcopy(graph['nodes'])
    elif graph_type == 2:
        for i in graph['nodes']:
            e_n = rng.randint(0,3)
            for _ in range(e_n):
                node_list.append(i)
    else:
        print("Graph type error!")

    flows = []
    idx = 0
    # st-1
    for i in range(st1):
        flow = {}
        flow['id'] = idx
        src = 0
        dst = 0
        while src == dst:
            src, dst = rng.sample(node_list, 2)
        flow['src'] = src 
        flow['dst'] = dst
        flow['size'] = 64 # byte
        flow['interval'] = 0.2 # ms
        flow['deadline'] = 0.1 # ms
        flow['path'] = []
        flow['sch'] = {}
        flow['latency'] = {} 
        flow['trans_delay'] = round(flow['size']/origin_link_bandwidth * 1000, 10)
        flows.append(flow)
        flow_RTSF['ST1'].append(flow)
        idx = idx + 1

    # st-2
    for i in range(st2):
        flow = {}
        flow['id'] = idx
        src = 0
        dst = 0
        while src == dst:
            src, dst = rng.sample(node_list, 2)
        flow['src'] = src 
        flow['dst'] = dst 
        flow['size'] = 128 # byte
        flow['interval'] = 0.5 # ms
        flow['deadline'] = 0.1 # ms
        flow['path'] = []
        flow['sch'] = {}
        flow['latency'] = {}
        flow['trans_delay'] = round(flow['size']/origin_link_bandwidth * 1000, 10)
        flows.append(flow)
        flow_RTSF['ST2'].append(flow)
        idx = idx + 1

    # st-3
    for i in range(st3):
        flow = {}
        flow['id'] = idx
        src = 0
        dst = 0
        while src == dst:
            src, dst = rng.sample(node_list, 2)
        flow['src'] = src 
        flow['dst'] = dst 
        flow['size'] = 192 # byte
        flow['interval'] = 1 # ms
        flow['deadline'] = 0.1 # ms
        flow['path'] = []
        flow['sch'] = {}
        flow['latency'] = {}
        flow['trans_delay'] = round(flow['size']/origin_link_bandwidth * 1000, 10)
        flows.append(flow)
        flow_RTSF['ST3'].append(flow)
        idx = idx + 1

    return flows


##############################################################################################################################################################
def make_random_walk_forwarding_map(network_topology):
    adjacency_matrix = nx.to_numpy_array(network_topology['G'], nodelist=sorted(network_topology['nodes']))
    nodes_number = len(network_topology['nodes'])
    rw_forwarding_map = {}

    for i in range(nodes_number):
        prob_item = []
        for j in range(len(adjacency_matrix[i])):
            for k in range(int(adjacency_matrix [i][j])):
                prob_item.append(j)

        rw_forwarding_map[i]= prob_item

    return rw_forwarding_map

def make_sp_forwarding_map(network_topology):
    nodes_number = len(network_topology['nodes'])
    sp_forwarding_map = {}

    for i in range(nodes_number):
        for j in range(nodes_number):
            if i != j:
                key = str(i) + "to" + str(j) 
                sp_forwarding_map[key] = {}

                paths = [p for p in nx.all_shortest_paths(network_topology['G'], source=i, target= j)]

                for path in paths:
                    for idx in range(len(path)-1):
                        try:
                            sp_forwarding_map[key][path[idx]].append(path[idx+1])
                        except:
                            sp_forwarding_map[key][path[idx]] = []
                            sp_forwarding_map[key][path[idx]].append(path[idx+1])

    return sp_forwarding_map

def biased_random_walk_routing(sp_degree, sp_routing_map, rw_routing_map, src, dst):
    current_location = src
    generated_path = []
    generated_path.append(current_location)

    while current_location != dst:
        decision_prob = random.random()

        if decision_prob < sp_degree:
            sp_key1 = str(current_location) + "to" + str(dst) 
            forwarding_list = sp_routing_map[sp_key1][current_location]
            current_location = random.choice(forwarding_list)
        else:
            forwarding_list = rw_routing_map[current_location]
            current_location = random.choice(forwarding_list)

        try:
            slicing_index = generated_path.index(current_location)
            generated_path = generated_path[0:slicing_index]
        except:
            pass
        
        generated_path.append(current_location)

    return generated_path

def make_routing_space(nodes, sp_degree, sp_routing_map, rw_routing_map, current_path_num):
    routes_distribution = {}
    for i in range(nodes):
        routes_distribution[i] = {}
        for j in range(nodes):
            if i != j:
                routes_distribution[i][j] = {}
                for itr in range(current_path_num):
                    path = biased_random_walk_routing(sp_degree, sp_routing_map, rw_routing_map, i, j)
                    path_str = ""

                    for hop in path:
                        int_str = ""
                        if hop < 10:
                            int_str = "0" + str(hop)
                        else:
                            int_str = str(hop)

                        if hop != path[-1]:
                            path_str = path_str  + int_str + "-"
                        else:
                            path_str = path_str  + int_str

                    if path_str in routes_distribution[i][j]:
                        routes_distribution[i][j][path_str] += 1
                    else:
                        routes_distribution[i][j][path_str] = 1

    routing_space = {}
    init_space = {}
    for i in range(nodes):
        routing_space[i] = {}
        init_space[i] = {}
        for j in range(nodes):
            if i != j:
                routing_space[i][j] = {}
                init_space[i][j] = 0
                start = 0
                end = 0
                sorted_distribution = sorted(routes_distribution[i][j].items(), key=lambda x:x[1], reverse=True)  
                before_len = 100000
                for pair in sorted_distribution: 
                    end = start + pair[1]
                    routing_space[i][j][end/current_path_num] = pair[0]
                    start = end
                    if len(pair[0]) <= before_len:
                        before_len = len(pair[0])
                        init_space[i][j] = end/current_path_num

    return routing_space, init_space


class Scheduler: 
    def __init__(self, flows, G, bias, slot_search_step, p_num):
        self.flows = copy.deepcopy(flows)
        self.LCM = 1 # ms

        self.G = copy.deepcopy(G)
        self.graph = copy.deepcopy(self.G['G'])
        self.nodes = self.G['nodes']

        self.origin_link_bandwidth = 1.25e+7 # 100mbps => 1.25e+7byte/s
        self.link_bandwidth = 1.25e+7 / 1000 * self.LCM 
        self.utilization_matrix = np.zeros((len(self.nodes), len(self.nodes)))

        self.schedules = {}
        self.scheduleInit()

        self.route_space = None
        self.route_init_space = None
        start = time.time()
        self.routeGenerating(bias, p_num)
        print("Time: ", time.time()-start)

        self.routing_time = time.time()-start
        self.start = start

        self.cost_divisor = 0
        for f in self.flows:
            self.cost_divisor += f['deadline']*self.LCM/f['interval']
        self.cost_divisor = round(self.cost_divisor, 10)

        self.slot_search_step = slot_search_step

    def scheduleInit(self):
        for u,v,d in self.graph.edges(data=True):
            self.schedules[u, v] = []
            self.schedules[u, v].append([0, self.LCM]) 
            for f in self.flows: 
                f['sch'][u, v] = {}

    def routeGenerating(self, w, p_num):
        sp_map = make_sp_forwarding_map(self.G)
        rw_map = make_random_walk_forwarding_map(self.G)
        self.route_space,  self.route_init_space = make_routing_space(len(self.nodes), w, sp_map, rw_map, p_num)

    def routing(self, route_solution):
        i = 0
        for f in self.flows:
            self.distributionPath(f, route_solution[i])
            i += 1

    def distributionPath(self, f, y):
        for key in self.route_space[f['src']][f['dst']]:
            if y <= key:
                path_result = self.route_space[f['src']][f['dst']][key].split('-')
                f['path'] = list(map(int, path_result))
                break

    def calScore(self, start_position, new_order):
        flows_id = np.array(range(len(self.flows)))
        flows_id = flows_id[new_order]
        
        for u,v,d in self.graph.edges(data=True):
            self.schedules[u, v] = []
            self.schedules[u, v].append([0, self.LCM]) 
        for i in flows_id:
            self.oneFlowSch(self.flows[i], start_position[i])

    def getLatency(self):
        flow_fail_count = 0
        fail_count = 0
        cost = 0
        for f in self.flows:
            f_flag = 0
            is_scheduled = 0
            starting_time = f['sch'][(f['path'][0], f['path'][1])][0][0]
            for it in range(int(self.LCM/f['interval'])):
                temp_latency = 0
                adding = 0
                for p_idx in range(len(f['path'])-2):
                    if f['sch'][(f['path'][p_idx], f['path'][p_idx+1])][it][1] == -1 or f['sch'][(f['path'][-2], f['path'][-1])][it][1] == -1:
                        is_scheduled = 1
                        break
                    if p_idx == 0:
                        adding = f['sch'][(f['path'][p_idx+1], f['path'][p_idx+2])][it][1] - (starting_time + it*f['interval'])
                        adding = round(adding, 10)
                    else:
                        adding = f['sch'][(f['path'][p_idx+1], f['path'][p_idx+2])][it][1] - f['sch'][(f['path'][p_idx], f['path'][p_idx+1])][it][1] 
                        adding = round(adding, 10)
                    if adding < 0:
                        adding = adding + self.LCM
                        adding = round(adding, 10)
                    temp_latency = temp_latency + adding

                if len(f['path']) == 2:
                    if f['sch'][(f['path'][0], f['path'][1])][it][1] == -1:
                        is_scheduled = 1
                    else:
                        temp_latency = f['sch'][(f['path'][0], f['path'][1])][it][1] - (starting_time + it*f['interval'])

                if is_scheduled == 1:
                    temp_latency = f['deadline']*10
                    
                f['latency'][it] = round(temp_latency, 10)
                if f['latency'][it] > f['deadline']:
                    cost = cost + f['latency'][it]
                    fail_count = fail_count + 1
                    f_flag = 1
                else:
                    cost += f['latency'][it]/(self.cost_divisor*10)
            flow_fail_count = flow_fail_count + f_flag
                    
        return flow_fail_count, fail_count, round(cost, 10) 

    def oneFlowSch(self, f, s_pos):    
        tx_start = round((s_pos//self.slot_search_step)*self.slot_search_step, 10)
        
        for it in range(int(self.LCM/f['interval'])):
            for p_idx in range(len(f['path'])-1):
                lcm_flag = 0
                src = f['path'][p_idx]
                dst = f['path'][p_idx+1]
                f['sch'][src, dst][it] = [-1, -1] 

                if p_idx != 0:
                    tx_start = f['sch'][f['path'][p_idx-1], f['path'][p_idx]][it][1]
                elif it !=0:
                    tx_start = round(f['sch'][f['path'][0], f['path'][1]][0][0] + it*f['interval'], 10)

                if tx_start < 0:
                    continue

                if tx_start > self.LCM:
                    lcm_flag = tx_start//self.LCM # 1
                    tx_start = round(tx_start%self.LCM, 10)
                    if lcm_flag > 1:
                        continue

                expected_end = round(tx_start+f['trans_delay'], 10)
                init_start = tx_start
                for slot in self.schedules[src, dst]:
                    if tx_start < slot[0]:
                        tx_start = slot[0]
                        expected_end = round(tx_start+f['trans_delay'], 10)
                    if expected_end > self.LCM:
                        break 

                    if expected_end <= slot[1]:
                        if lcm_flag != 0:
                            if round(expected_end+self.LCM - f['sch'][f['path'][0], f['path'][1]][0][0], 10) > 1:
                                break
                            f['sch'][src, dst][it][0] = round(tx_start+self.LCM, 10)
                            f['sch'][src, dst][it][1] = round(expected_end+self.LCM, 10)                    
                        else:
                            f['sch'][src, dst][it][0] = tx_start
                            f['sch'][src, dst][it][1] = expected_end 

                        front_slot = [slot[0], tx_start]
                        back_slot = [expected_end, slot[1]]
                        if front_slot[0] != front_slot[1]:
                            self.schedules[src, dst].append(front_slot)
                        if back_slot[0] != back_slot[1]:
                            self.schedules[src, dst].append(back_slot)

                        self.schedules[src, dst].remove(slot)
                        self.schedules[src, dst].sort(key=lambda x:x[0])
                        break

                if expected_end > self.LCM and lcm_flag < 1:
                    new_for_flag = 0
                    if self.schedules[src, dst][-1][1] == self.LCM and self.schedules[src, dst][0][0] == 0:
                        if self.schedules[src, dst][-1][0] > tx_start:
                            tx_start = self.schedules[src, dst][-1][0]
                            expected_end = round(tx_start+f['trans_delay'], 10)

                        if self.schedules[src, dst][-1][0] <= tx_start and round(expected_end % self.LCM, 10) <= self.schedules[src, dst][0][1]:
                            new_for_flag = 1
                            if round(expected_end - f['sch'][f['path'][0], f['path'][1]][0][0], 10) > 1:
                                continue

                            f['sch'][src, dst][it][0] = tx_start
                            f['sch'][src, dst][it][1] = expected_end 

                            if self.schedules[src, dst][-1][0] == tx_start:
                                self.schedules[src, dst].pop()
                            else:
                                self.schedules[src, dst][-1][1] = tx_start

                            self.schedules[src, dst][0][0] = round(expected_end % self.LCM, 10)
                            if self.schedules[src, dst][0][0] == self.schedules[src, dst][0][1]:
                                self.schedules[src, dst].remove(self.schedules[src, dst][0])

                    if new_for_flag == 0:
                        tx_start = 0
                        expected_end = f['trans_delay']
                        for slot in self.schedules[src, dst]:
                            if tx_start < slot[0]:
                                tx_start = slot[0]
                                expected_end = round(tx_start+f['trans_delay'], 10)     
                            if round(expected_end+self.LCM - f['sch'][f['path'][0], f['path'][1]][0][0], 10) > 1 or init_start <= tx_start:
                                break

                            if expected_end <= slot[1]:
                                f['sch'][src, dst][it][0] = round(tx_start+self.LCM, 10)
                                f['sch'][src, dst][it][1] = round(expected_end+self.LCM, 10)                    

                                front_slot = [slot[0], tx_start]
                                back_slot = [expected_end, slot[1]]
                                if front_slot[0] != front_slot[1]:
                                    self.schedules[src, dst].append(front_slot)
                                if back_slot[0] != back_slot[1]:
                                    self.schedules[src, dst].append(back_slot)

                                self.schedules[src, dst].remove(slot)
                                self.schedules[src, dst].sort(key=lambda x:x[0])
                                break


##############################################################################################################################################################
class BBTO:
    def __init__(self, schedule, n_pop, n_elite, n_invader, p_mu, mutation_distance, offset_resolution, termination_time, graph_name):
        self.start = time.time()
        self.routing_time = schedule.routing_time

        self.sch = schedule
        self.max_it = 100000
        self.n_var = len(self.sch.flows)                    
        self.var_size = np.array([1, self.n_var])            
        self.var_min = np.array([0])                         
        self.var_max = np.array([self.n_var-1])              

        self.termination_time = termination_time             
        self.graph_name = graph_name                         

        self.n_pop = n_pop
        self.n_elite = n_elite
        self.migrant = n_pop - self.n_elite   
        self.n_invader = n_invader
        self.p_mutation = p_mu   
        self.sigma = round(mutation_distance * (self.var_min[0] - self.var_max[0]), 10)
        self.offset_resolution = round(self.n_var/offset_resolution, 10)

        self.mu = (self.n_pop + 1 - np.array(range(1, self.n_pop + 1))) / (self.n_pop + 1)  
        self.mr = 1 - self.mu

        self.x_pop_position = np.zeros((self.n_pop, self.n_var))
        self.y_pop_position = np.zeros((self.n_pop, self.n_var))
        self.pop_cost = np.zeros((self.n_pop))
        self.best_scheduled = np.zeros((self.n_pop)) 
        self.best_flow_scheduled = np.zeros((self.n_pop)) 
        self.solution_init()
        print(self.pop_cost)

        self.x_best_sol =  self.x_pop_position[0, :]
        self.y_best_sol =  self.x_pop_position[0, :]
        self.best_flow_scheduled_cost = list()          # Array to Hold Best flow scheduled Costs
        self.best_scheduled_cost = list()               # Array to Hold Best scheduled Costs
        self.best_cost = list()                         # Array to Hold Best Costs
        self.iteration_times = list()                   

    def solution_init(self):
        init_mul = 10
        temp_x = np.zeros((self.n_pop*init_mul, self.n_var))
        temp_y = np.zeros((self.n_pop*init_mul, self.n_var))
        temp_cost = np.zeros((self.n_pop*init_mul))
        temp_scheduled = np.zeros((self.n_pop*init_mul))
        temp_flow_scheduled = np.zeros((self.n_pop*init_mul))
        for iter in range(0, self.n_pop*init_mul):
            temp_x[iter, :] = np.random.uniform(self.var_min, self.var_max, self.var_size)
            for f_idx in range(self.n_var):
                src = self.sch.flows[f_idx]['src']
                dst = self.sch.flows[f_idx]['dst']
                init_max = self.sch.route_init_space[src][dst] * self.var_max
                temp_y[iter, f_idx] = np.random.uniform(self.var_min, init_max)
            temp_flow_scheduled[iter], temp_scheduled[iter], temp_cost[iter] = self.cost_function(temp_x[iter, :], temp_y[iter, :])
        temp_flow_scheduled, temp_scheduled, temp_x, temp_y, temp_cost = self.pop_sort(temp_flow_scheduled, temp_scheduled, temp_x, temp_y, temp_cost)

        self.x_pop_position = temp_x[: self.n_pop]
        self.y_pop_position = temp_y[: self.n_pop]
        self.pop_cost = temp_cost[: self.n_pop]
        self.best_scheduled = temp_scheduled[: self.n_pop]
        self.best_flow_scheduled = temp_flow_scheduled[: self.n_pop]

    def cost_function(self, x_solution, y_solution):
        self.sch.routing(y_solution/(len(y_solution)-1))
        self.sch.calScore(x_solution%self.offset_resolution/self.offset_resolution, x_solution.argsort())
        return0, return1, return2 = self.sch.getLatency()
        return return0, return1, return2

    def pop_sort(self, flow_scheduled, scheduled, x_pos, y_pos, cost):
        sort_order = np.argsort(cost, axis=0)
        cost = cost[sort_order]
        x_pos = x_pos[sort_order]
        y_pos = y_pos[sort_order]
        scheduled = scheduled[sort_order]
        flow_scheduled = flow_scheduled[sort_order]
        return flow_scheduled, scheduled, x_pos, y_pos, cost

    def optimization(self):
        start = self.start
        for iter in range(self.max_it):
            new_pop_cost = self.pop_cost.copy()
            new_best_scheduled = self.best_scheduled.copy()
            new_best_flow_scheduled = self.best_flow_scheduled.copy()
            x_new_pop_pos = self.x_pop_position.copy()
            y_new_pop_pos = self.y_pop_position.copy()

            for i in range(self.n_pop):
                for k in range(self.n_var):
                    if np.random.uniform() < self.mr[i]: # immigrate?
                        random_number = np.random.uniform() * np.sum(self.mu)
                        select = self.mu[0]
                        select_index = 0
                        while (random_number > select) and (select_index < self.n_pop - 1):
                            select_index += 1
                            select += self.mu[select_index]
                        x_new_pop_pos[i][k] =  self.x_pop_position[select_index][k]
                        y_new_pop_pos[i][k] =  self.y_pop_position[select_index][k]

                    # Mutation
                    if np.random.rand() <= self.p_mutation:
                        sep = np.random.rand()
                        if sep < 1/3:
                            x_new_pop_pos[i, k] += self.sigma * np.random.randn(1)
                        elif sep < 2/3:
                            y_new_pop_pos[i, k] = np.random.uniform(self.var_min, self.var_max)
                        else:
                            x_new_pop_pos[i, k] += self.sigma * np.random.randn(1)
                            y_new_pop_pos[i, k] = np.random.uniform(self.var_min, self.var_max)
                        
                    if x_new_pop_pos[i, k] > self.var_max or x_new_pop_pos[i, k] < 0:
                        x_new_pop_pos[i, k] = np.random.rand() * self.var_max

                # Evaluation
                new_best_flow_scheduled[i], new_best_scheduled[i], new_pop_cost[i] = self.cost_function(x_new_pop_pos[i, :], y_new_pop_pos[i, :])
            
            # Sort New Population
            new_best_flow_scheduled, new_best_scheduled, x_new_pop_pos, y_new_pop_pos, new_pop_cost = self.pop_sort(new_best_flow_scheduled, new_best_scheduled, x_new_pop_pos, y_new_pop_pos, new_pop_cost)
            
            # Select Next Iteration Population
            self.x_pop_position = np.concatenate((self.x_pop_position[0:self.n_elite, :], x_new_pop_pos[0:self.migrant, :]), axis=0)
            self.y_pop_position = np.concatenate((self.y_pop_position[0:self.n_elite, :], y_new_pop_pos[0:self.migrant, :]), axis=0)
            self.pop_cost = np.concatenate((self.pop_cost[0:self.n_elite], new_pop_cost[0:self.migrant]), axis = 0)
            self.best_scheduled = np.concatenate((self.best_scheduled[0:self.n_elite], new_best_scheduled[0:self.migrant]), axis = 0)
            self.best_flow_scheduled = np.concatenate((self.best_flow_scheduled[0:self.n_elite], new_best_flow_scheduled[0:self.migrant]), axis = 0)
            
            for v in range(self.n_invader):
                self.x_pop_position[-(v+1), :] = np.random.uniform(self.var_min, self.var_max, self.var_size)
                self.y_pop_position[-(v+1), :] = np.random.uniform(self.var_min, self.var_max, self.var_size)
                self.best_flow_scheduled[-(v+1)], self.best_scheduled[-(v+1)], self.pop_cost[-(v+1)] = self.cost_function(self.x_pop_position[-(v+1), :], self.y_pop_position[-(v+1), :])
            
            # Sort Population
            self.best_flow_scheduled, self.best_scheduled, self.x_pop_position, self.y_pop_position, self.pop_cost = self.pop_sort(self.best_flow_scheduled, self.best_scheduled, self.x_pop_position, self.y_pop_position, self.pop_cost)

            # Update Best Solution Ever Found
            self.x_best_sol = self.x_pop_position[0, :]
            self.y_best_sol = self.y_pop_position[0, :]

            # Store Best Cost Ever Found
            self.best_flow_scheduled_cost.append(self.best_flow_scheduled[0])
            self.best_scheduled_cost.append(self.best_scheduled[0])
            self.best_cost.append(self.pop_cost[0])
            self.iteration_times.append(time.time() - start)
                        
            # Log Iteration Info
            print(f"Iteration {iter}: Best Cost = {self.best_cost[iter]} & Best TX Scheduled = {self.best_scheduled_cost[iter]} & Best Flow Scheduled Cost = {self.best_flow_scheduled_cost[iter]} & Time = {self.iteration_times[iter]}") 


            if time.time() - start > self.termination_time:
                print("Time up!")
                self.cost_function(self.x_pop_position[0, :], self.y_pop_position[0, :])
                #pprint.pprint(self.sch.flows)
                default_name = f"BBTO_{self.graph_name}_{len(self.sch.flows)}_simul{1}"

                df_flows = pd.DataFrame(self.sch.flows)
                df_flows.to_csv(default_name + 'flow_schedule.csv')
                df_cost = pd.DataFrame(self.best_cost)
                df_cost.to_csv(default_name + 'iter_cost.csv')
                df_scheduled_cost = pd.DataFrame(self.best_scheduled_cost)
                df_scheduled_cost.to_csv(default_name + 'iter_tx_scheduled_cost.csv')
                df_flow_scheduled_cost = pd.DataFrame(self.best_flow_scheduled_cost)
                df_flow_scheduled_cost.to_csv(default_name + 'iter_flow_scheduled_cost.csv')
                df_time = pd.DataFrame(self.iteration_times)
                df_time.to_csv(default_name + 'iter_time.csv') 
                # routing time
                df_routing_time = pd.DataFrame([self.routing_time])
                df_routing_time.to_csv(default_name + "routing_time.csv")

                break


def main(argv, args) : 
    print('#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#')
    print(f'argv : ', argv)
    print(f'args : ', args)
    
    print('Topology : ', args.input_topology)
    print('N_flows : ', args.input_n_flows)
    print('Seed(G) : ', args.input_seed_G)
    print('Seed(F) : ', args.input_seed_F)
    print('Pops : ', args.input_n_pop)
    print('Elites : ', args.input_n_elite)
    print('Invaders : ', args.input_n_invader)
    print('P_mu : ', args.input_mutation)
    print('#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#')
    print('\n')

    g_name = args.input_topology
    graph_set = makeG(args.input_seed_G)
    input_graph_type = None
    if g_name == "CEV":
        input_graph_type = 0
    elif g_name == "DFC":
        input_graph_type = 1
    else:
        input_graph_type = 2
    
    n_flows = int(args.input_n_flows/3)
    flow_set = makeFlowSet(n_flows, n_flows, n_flows, graph_set[g_name], graph_type = input_graph_type, f_seed = args.input_seed_F)

    sp_biased = 1 - math.log2(len(graph_set[g_name]['G'].edges)/len(graph_set[g_name]['nodes']))/10
    if sp_biased < 0.5: 
        sp_biased = 0.5

    mu_size = args.input_mutation
    if mu_size == -1:
        mu_size = 1.25 # util. based value, do hardcoding (max.- min. | threshold  1~2)

    # run
    print("Scheduler configuration.....")
    sch = Scheduler(flow_set, graph_set[g_name], sp_biased, 0.00512, p_num=10000)
    print("Optimization start!")
    optimizer = BBTO(
        schedule = sch, 
        n_pop = args.input_n_pop, 
        n_elite = args.input_n_elite, 
        n_invader = args.input_n_invader, 
        p_mu = mu_size/len(flow_set), 
        mutation_distance = 0.0001, 
        offset_resolution = 100, 
        termination_time = 3600, 
        graph_name = g_name
    )
    optimizer.optimization()    
    
if __name__ == '__main__' :
    argv = sys.argv
    main(argv, args)