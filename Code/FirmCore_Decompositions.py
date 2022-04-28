import heapq
import pickle
from math import sqrt
from objective_function import objective_function, log_objective_function

def FirmCore(multilayer_graph, nodes_iterator, layers_iterator, threshold, information, save=False):
    # degree of each node in each layer
    delta = {}

    # nodes with the same top lambda degree
    delta_map = {}

    # set of neighbors that we need to update
    neighbors = set()

    k_max = 0
    k_start = 0

    if threshold == 1:
        k_start = 1

    information.start_algorithm()

    for node in nodes_iterator:
        delta[node] = [len([neighbor for neighbor in multilayer_graph[node][layer]]) for layer in layers_iterator]
        delta_map[node] = heapq.nlargest(threshold, delta[node])[-1]

    k_max = max(list(delta_map.values()))
    # bin-sort for removing a vertex
    B = [set() for i in range(k_max + 1)]

    for node in nodes_iterator:
        B[delta_map[node]].add(node)

    print("maximum k = ", k_max)
    for k in range(k_start, k_max + 1):
        while B[k]:
            node = B[k].pop()
            delta_map[node] = k
            neighbors = set()

            for layer, layer_neighbors in enumerate(multilayer_graph[node]):
                for neighbor in layer_neighbors:
                    if delta_map[neighbor] > k:
                        delta[neighbor][layer] -= 1
                        if delta[neighbor][layer] + 1 == delta_map[neighbor]:
                            neighbors.add(neighbor)

            for neighbor in neighbors:
                B[delta_map[neighbor]].remove(neighbor)
                delta_map[neighbor] = heapq.nlargest(threshold, delta[neighbor])[-1]
                B[max(delta_map[neighbor], k)].add(neighbor)
        
    # end of the algorithm
    information.end_algorithm(max(delta_map.values()))

    if save:
        a_file = open("../output/" + information.dataset + "_" + str(threshold) + "_FirmCore_decomposition.pkl", "wb")
        pickle.dump(delta_map, a_file)
        a_file.close()

    return




def FirmDcore(multilayer_graph_in, multilayer_graph_out, nodes_iterator, layers_iterator, threshold, information, save=False, count=False):
    # in degree of each node in each layer
    delta_in = {}

    # in degree of each node in each layer
    delta_out = {}

    # nodes with the same top lambda degree
    delta_in_map = {}

    # nodes with the same top lambda degree
    delta_out_map = {}

    
    if threshold == 1:
        k_start = 1
        s_start = 1

    for node in nodes_iterator:
        delta_out[node] = [len([neighbor for neighbor in multilayer_graph_out[node][layer]]) for layer in layers_iterator]
        delta_out_map[node] = heapq.nlargest(threshold, delta_out[node])[-1]

    k_out_max = max(list(delta_out_map.values()))

    information.start_algorithm()
    
    for k in range(k_start, k_out_max):
        # calculate degrees
        for node in nodes_iterator:
            delta_in[node] = [len([neighbor for neighbor in multilayer_graph_in[node][layer]]) for layer in layers_iterator]
            delta_out[node] = [len([neighbor for neighbor in multilayer_graph_out[node][layer]]) for layer in layers_iterator]
            delta_in_map[node] = heapq.nlargest(threshold, delta_in[node])[-1]
            delta_out_map[node] = heapq.nlargest(threshold, delta_out[node])[-1]
        
        s_in_max = max(list(delta_in_map.values()))

        # build Buckets
        B_in = [set() for i in range(s_in_max + 1)]
        for node in nodes_iterator:
            B_in[delta_in_map[node]].add(node)

        
        for s in range(s_start, s_in_max):
            while B_in[s]:
                node = B_in[s].pop()
                delta_in_map[node] = s
                neighbors = set()

                for layer, layer_neighbors in enumerate(multilayer_graph_in[node]):
                    for neighbor in layer_neighbors:
                        if delta_out_map[neighbor] >= k:
                            delta_out[neighbor][layer] -= 1
                            if delta_out[neighbor][layer] + 1 == delta_out_map[neighbor]:
                                neighbors.add(neighbor)
                
                for neighbor in neighbors:
                    neighbors2 = set()
                    delta_out_map[neighbor] = heapq.nlargest(threshold, delta_out[neighbor])[-1]
                    if delta_out_map[neighbor] < k:
                        for layer, layer_neighbors in enumerate(multilayer_graph_out[neighbor]):
                            for neighbor2 in layer_neighbors:
                                if delta_in_map[neighbor2] > s:
                                    delta_in[neighbor2][layer] -= 1
                                    if delta_in[neighbor2][layer] + 1 == delta_in_map[neighbor2]:
                                        neighbors2.add(neighbor2)

                        for neighbor2 in neighbors2:
                            B_in[delta_in_map[neighbor2]].remove(neighbor2)
                            delta_in_map[neighbor2] = heapq.nlargest(threshold, delta_in[neighbor2])[-1]
                            B_in[max(delta_in_map[neighbor2], s)].add(neighbor2)

            # At the end of the for loop, here, we can save all the nodes with delta_out_map == k as S and all the nodes with delta_in_map == s as T
            if save or count:
                exist_tuples_core = []
                core = {'S':[], 'T':[]}
                for node in nodes_iterator:
                    if delta_in_map[node] >= s:
                        core['T'].append(node)
                    if delta_out_map[node] >= k:
                        core['S'].append(node)

                if save:
                    if len(core['S']) * len(core['T']) > 0:
                        a_file = open("../output/" + information.dataset + "_" + str(k) + "_" + str(s) + str(threshold) + "_FirmCore_decomposition.pkl", "wb")
                        pickle.dump(core, a_file)
                        a_file.close()
                
                if count:
                    if len(core['S']) * len(core['T']) > 0:
                        exist_tuples_core.append((k, s))
                
    # end of the algorithm
    information.end_algorithm(0)
    if count:
        with open("../output/" + information.dataset + "directed_none_empty_cores" ".txt", 'w') as f:
            for item in exist_tuples_core:
                f.write("%s " % item)
    return




############################################################
############################################################
############################################################
##################### Decomposition ########################
############################################################
############################################################
############################################################


def FirmCore_decomposition(multilayer_graph, nodes_iterator, layers_iterator, information, save=False):
    for threshold in range(1, max(layers_iterator) + 2):
        print("-------------- threshold = %d --------------"%threshold)
        # degree of each node in each layer
        delta = {}

        # nodes with the same top lambda degree
        delta_map = {}

        # set of neighbors that we need to update
        neighbors = set()

        k_max = 0
        k_start = 0

        # distinct cores
        dist_cores = 0

        if threshold == 1:
            k_start = 1

        information.start_algorithm()

        for node in nodes_iterator:
            delta[node] = [len([neighbor for neighbor in multilayer_graph[node][layer]]) for layer in layers_iterator]
            delta_map[node] = heapq.nlargest(threshold, delta[node])[-1]

        if save:
            a_file = open("../output/" + information.dataset + "_" + str(threshold) + "_FirmCore_decomposition_degree.pkl", "wb")
            pickle.dump(delta_map, a_file)
            a_file.close()

        k_max = max(list(delta_map.values()))
        # bin-sort for removing a vertex
        B = [set() for i in range(k_max + 1)]

        for node in nodes_iterator:
            B[delta_map[node]].add(node)

        print("maximum k = ", k_max)
        for k in range(k_start, k_max + 1):
            if B[k]:
                dist_cores += 1
            while B[k]:
                node = B[k].pop()
                delta_map[node] = k
                neighbors = set()

                for layer, layer_neighbors in enumerate(multilayer_graph[node]):
                    for neighbor in layer_neighbors:
                        if delta_map[neighbor] > k:
                            delta[neighbor][layer] -= 1
                            if delta[neighbor][layer] + 1 == delta_map[neighbor]:
                                neighbors.add(neighbor)

                for neighbor in neighbors:
                    B[delta_map[neighbor]].remove(neighbor)
                    delta_map[neighbor] = heapq.nlargest(threshold, delta[neighbor])[-1]
                    B[max(delta_map[neighbor], k)].add(neighbor)
            
        # end of the algorithm
        information.end_algorithm(max(delta_map.values()))
        information.print_end_algorithm()

        print("Number of Distinct cores = %s"%dist_cores)

        if save:
            a_file = open("../output/" + information.dataset + "_" + str(threshold) + "_FirmCore_decomposition.pkl", "wb")
            pickle.dump(delta_map, a_file)
            a_file.close()





def FirmDcore_decomposition(multilayer_graph_in, multilayer_graph_out, nodes_iterator, layers_iterator, information, save=False, count=False):
    for threshold in range(1, max(layers_iterator) + 2):
        print("-------------- threshold = %d --------------"%threshold)
        # in degree of each node in each layer
        delta_in = {}

        # out degree of each node in each layer
        delta_out = {}

        # nodes with the same top lambda degree
        delta_in_map = {}

        # nodes with the same top lambda degree
        delta_out_map = {}

        # non-epmty cores
        exist_tuples_core = []

        # distinct cores
        dist_cores = 0

        
        if threshold == 1:
            k_start = 1
            s_start = 1

        for node in nodes_iterator:
            delta_out[node] = [len([neighbor for neighbor in multilayer_graph_out[node][layer]]) for layer in layers_iterator]
            delta_out_map[node] = heapq.nlargest(threshold, delta_out[node])[-1]

        k_out_max = max(list(delta_out_map.values()))

        information.start_algorithm()

        
        print("maximum k = ", k_out_max)
        for k in range(k_start, k_out_max):
            flag = False
            active_S_nodes = set(nodes_iterator)
            # calculate degrees
            for node in active_S_nodes.copy():
                delta_out[node] = [len([neighbor for neighbor in multilayer_graph_out[node][layer]]) for layer in layers_iterator]
                delta_out_map[node] = heapq.nlargest(threshold, delta_out[node])[-1]
                if delta_out_map[node] < k:
                    active_S_nodes.remove(node)
               
            
            for node in nodes_iterator:
                delta_in[node] = [len([neighbor for neighbor in multilayer_graph_in[node][layer] if neighbor in active_S_nodes]) for layer in layers_iterator]
                delta_in_map[node] = heapq.nlargest(threshold, delta_in[node])[-1]
            
            s_in_max = max(list(delta_in_map.values()))

            # build Buckets
            B_in = [set() for i in range(s_in_max + 1)]
            for node in nodes_iterator:
                B_in[delta_in_map[node]].add(node)


            for s in range(s_start, s_in_max):
                if B_in[s]:
                    dist_cores += 1
                while B_in[s]:
                    node = B_in[s].pop()
                    delta_in_map[node] = s
                    neighbors = set()

                    for layer, layer_neighbors in enumerate(multilayer_graph_in[node]):
                        for neighbor in layer_neighbors:
                            if delta_out_map[neighbor] >= k:
                                delta_out[neighbor][layer] -= 1
                                if delta_out[neighbor][layer] + 1 == delta_out_map[neighbor]:
                                    neighbors.add(neighbor)
                    
                    for neighbor in neighbors:
                        neighbors2 = set()
                        delta_out_map[neighbor] = heapq.nlargest(threshold, delta_out[neighbor])[-1]
                        if delta_out_map[neighbor] < k:
                            for layer, layer_neighbors in enumerate(multilayer_graph_out[neighbor]):
                                for neighbor2 in layer_neighbors:
                                    if delta_in_map[neighbor2] > s:
                                        delta_in[neighbor2][layer] -= 1
                                        if delta_in[neighbor2][layer] + 1 == delta_in_map[neighbor2]:
                                            neighbors2.add(neighbor2)

                            for neighbor2 in neighbors2:
                                B_in[delta_in_map[neighbor2]].remove(neighbor2)
                                delta_in_map[neighbor2] = heapq.nlargest(threshold, delta_in[neighbor2])[-1]
                                B_in[max(delta_in_map[neighbor2], s)].add(neighbor2)

                # At the end of the for loop, here, we can save all the nodes with delta_out_map == k as S and all the nodes with delta_in_map == s as T
                if save or count:
                    core = {'S':[], 'T':[]}
                    for node in nodes_iterator:
                        if delta_in_map[node] >= s:
                            core['T'].append(node)
                        if delta_out_map[node] >= k:
                            core['S'].append(node)
                        
                    if save:
                        if len(core['S']) * len(core['T']) > 0:
                            flag = True
                            a_file = open("../output/" + information.dataset + "_" + str(k) + "_" + str(s) + "_" + str(threshold) + "_FirmCore_decomposition.pkl", "wb")
                            pickle.dump(core, a_file)
                            a_file.close()
        
                    if count:
                        if len(core['S']) * len(core['T']) > 0:
                            flag = True
                            exist_tuples_core.append((k, s))
            if not flag:
                break

        # end of the algorithm
        print("Number of Distinct cores = %s"%dist_cores)
        information.end_algorithm(0)
        information.print_end_algorithm()
        if count:
            with open("../output/" + information.dataset + "_" + str(threshold) + "_directed_none_empty_cores" ".txt", 'w') as f:
                for item in exist_tuples_core:
                    f.write("%s " % str(item))






############################################################
############################################################
############################################################
################### Densest Subgraph #######################
############################################################
############################################################
############################################################


def Densest_subgraph(multilayer_graph, nodes_iterator, layers_iterator, beta, information):
    # maximum value of objective function given threshold
    max_obj = [0, 0, 0]

    # densest subgraph
    dense_subgraph = 0

    # dense core number
    dense_core_number = (0, 0)

    information.start_algorithm()

    output = set()

    for threshold in range(1, max(layers_iterator) + 2):
        # degree of each node in each layer
        delta = {}

        # nodes with the same top lambda degree
        delta_map = {}

        # set of neighbors that we need to update
        neighbors = set()

        k_max = 0
        k_start = 0

        active_nodes = set(nodes_iterator)

        for node in nodes_iterator:
            delta[node] = [len([neighbor for neighbor in multilayer_graph[node][layer]]) for layer in layers_iterator]
            delta_map[node] = heapq.nlargest(threshold, delta[node])[-1]

        k_max = max(list(delta_map.values()))
        # bin-sort for removing a vertex
        B = [set() for i in range(k_max + 1)]

        for node in nodes_iterator:
            B[delta_map[node]].add(node)
        
        if threshold == 1:
            k_start = 1
            active_nodes.difference_update(list(B[0]))

        for k in range(k_start, k_max + 1):
            temporal_active_nodes = []
            while B[k]:
                node = B[k].pop()
                temporal_active_nodes.append(node)
                delta_map[node] = k
                neighbors = set()

                for layer, layer_neighbors in enumerate(multilayer_graph[node]):
                    for neighbor in layer_neighbors:
                        if delta_map[neighbor] > k:
                            delta[neighbor][layer] -= 1
                            if delta[neighbor][layer] + 1 == delta_map[neighbor]:
                                neighbors.add(neighbor)

                for neighbor in neighbors:
                    B[delta_map[neighbor]].remove(neighbor)
                    delta_map[neighbor] = heapq.nlargest(threshold, delta[neighbor])[-1]
                    B[max(delta_map[neighbor], k)].add(neighbor)
            
            number_nodes = len(active_nodes)
            if number_nodes > 0:
                # compute the number of edges of each layer from delta
                number_of_edges_layer_by_layer = {}
                for layer in layers_iterator:
                    number_of_edges_layer_by_layer[layer] = sum([delta[node][layer] for node in active_nodes]) / 2

                # compute core objective function
                core_objective_function = objective_function(number_nodes, number_of_edges_layer_by_layer, beta)

                if core_objective_function[0] >= max_obj[0]:
                    max_obj = core_objective_function
                    dense_subgraph = number_nodes
                    dense_core_number = (k + 1, threshold)
                    output = set(active_nodes)

            active_nodes.difference_update(temporal_active_nodes)

    # end of the algorithm
    information.end_algorithm(k_max=None)
    information.print_densest_subgraph(beta, max_obj[0], dense_subgraph, max_obj[1], dense_core_number, max_obj[2])

    a_file = open("../output/" + information.dataset + "_" + "densest_subgraph.pkl", "wb")
    pickle.dump(output, a_file)
    a_file.close()


    return 


def Directed_densest_subgraph(multilayer_graph_in, multilayer_graph_out, nodes_iterator, layers_iterator, beta, information):
    # maximum value of objective function given threshold
    max_obj = [0, 0, 0]

    # densest subgraph
    dense_subgraph = 0

    # dense core number
    dense_core_number = (0, 0, 0)

    information.start_algorithm()

    for threshold in range(1, max(layers_iterator) + 2):
        # in degree of each node in each layer
        delta_in = {}

        # in degree of each node in each layer
        delta_out = {}

        # nodes with the same top lambda degree
        delta_in_map = {}

        # nodes with the same top lambda degree
        delta_out_map = {}

        
        if threshold == 1:
            k_start = 1
            s_start = 1

        for node in nodes_iterator:
            delta_out[node] = [len([neighbor for neighbor in multilayer_graph_out[node][layer]]) for layer in layers_iterator]
            delta_out_map[node] = heapq.nlargest(threshold, delta_out[node])[-1]

        k_out_max = max(list(delta_out_map.values()))

        
        for k in range(k_start, k_out_max):
            if k > 1:
                flag = False
            else:
                flag = True
            # calculate degrees
            active_S_nodes = set(nodes_iterator)
            # calculate degrees
            for node in active_S_nodes.copy():
                delta_out[node] = [len([neighbor for neighbor in multilayer_graph_out[node][layer]]) for layer in layers_iterator]
                delta_out_map[node] = heapq.nlargest(threshold, delta_out[node])[-1]
                if delta_out_map[node] < k:
                    active_S_nodes.remove(node)
               
            
            for node in nodes_iterator:
                delta_in[node] = [len([neighbor for neighbor in multilayer_graph_in[node][layer] if neighbor in active_S_nodes]) for layer in layers_iterator]
                delta_in_map[node] = heapq.nlargest(threshold, delta_in[node])[-1]
            
            s_in_max = max(list(delta_in_map.values()))

            # build Buckets
            B_in = [set() for i in range(s_in_max + 1)]
            for node in nodes_iterator:
                B_in[delta_in_map[node]].add(node)
            
            active_nodes_S = set(nodes_iterator)
            active_nodes_T = set(nodes_iterator)

            for s in range(s_start, s_in_max):
                while B_in[s]:
                    node = B_in[s].pop()
                    delta_in_map[node] = s
                    active_nodes_T.remove(node)
                    neighbors = set()

                    for layer, layer_neighbors in enumerate(multilayer_graph_in[node]):
                        for neighbor in layer_neighbors:
                            if delta_out_map[neighbor] >= k:
                                delta_out[neighbor][layer] -= 1
                                if delta_out[neighbor][layer] + 1 == delta_out_map[neighbor]:
                                    neighbors.add(neighbor)
                    
                    for neighbor in neighbors:
                        neighbors2 = set()
                        delta_out_map[neighbor] = heapq.nlargest(threshold, delta_out[neighbor])[-1]
                        if delta_out_map[neighbor] < k:
                            active_nodes_S.remove(neighbor)
                            for layer, layer_neighbors in enumerate(multilayer_graph_out[neighbor]):
                                for neighbor2 in layer_neighbors:
                                    if delta_in_map[neighbor2] > s:
                                        delta_in[neighbor2][layer] -= 1
                                        if delta_in[neighbor2][layer] + 1 == delta_in_map[neighbor2]:
                                            neighbors2.add(neighbor2)

                            for neighbor2 in neighbors2:
                                B_in[delta_in_map[neighbor2]].remove(neighbor2)
                                delta_in_map[neighbor2] = heapq.nlargest(threshold, delta_in[neighbor2])[-1]
                                B_in[max(delta_in_map[neighbor2], s)].add(neighbor2)

                number_nodes = sqrt(len(active_nodes_S) * len(active_nodes_T))
                if number_nodes > 0:
                    flag = True
                    # compute the number of edges of each layer from delta
                    number_of_edges_layer_by_layer = {}
                    for layer in layers_iterator:
                        number_of_edges_layer_by_layer[layer] = sum([delta_out[node][layer] for node in active_nodes_S])

                    # compute core objective function
                    core_objective_function = objective_function(number_nodes, number_of_edges_layer_by_layer, beta)

                    if core_objective_function[0] > max_obj[0]:
                        max_obj = core_objective_function
                        dense_subgraph = (len(active_nodes_S), len(active_nodes_T))
                        dense_core_number = (k, s + 1, threshold)

            if not flag:
                break    

    # end of the algorithm
    information.end_algorithm(0)
    information.print_densest_subgraph(beta, max_obj[0], dense_subgraph, max_obj[1], dense_core_number, max_obj[2])

    return
