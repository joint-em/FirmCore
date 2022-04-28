from FirmCore_Decompositions import FirmCore, FirmCore_decomposition, FirmDcore, FirmDcore_decomposition, Densest_subgraph, Directed_densest_subgraph
from multilayer_graph import MultilayerGraph, DirectedMultilayerGraph
from Information import Information
from time import time
import argparse


if __name__ == '__main__':
    # create a parser
    parser = argparse.ArgumentParser(description='FirmCore Decomposition and Densest Subgraph in Multilayer Networks')

    # arguments
    parser.add_argument('d', help='dataset')
    parser.add_argument('m', help='method')
    parser.add_argument('g', help='type of graph')
    parser.add_argument('-b', help='beta', type=float)
    parser.add_argument('-l', help='lambda', type=int)
    # options
    parser.add_argument('--dic', dest='dic', action='store_true', default=True ,help='dicomposition')
    parser.add_argument('--save', dest='save', action='store_true', default=False ,help='save results')
    parser.add_argument('--count', dest='count', action='store_true', default=False ,help='count D-cores')

    # read the arguments
    args = parser.parse_args()

    # dataset path
    dataset_path = "../datasets/" + args.d

    information = Information(args.d)

    information.print_dataset_name(dataset_path)

    # create the input graph and print its name
    start = int(round(time() * 1000))
    if args.g != "directed":
        multilayer_graph = MultilayerGraph(dataset_path)
    else:
        multilayer_graph = DirectedMultilayerGraph(dataset_path)
    end = int(round(time() * 1000))
    print(" >>>> Preprocessing Time: ", (end - start)/1000.00, " (s)\n")
    
    # FirmCore decomposition algorithms
    if args.m == 'core':
        if args.g != 'directed':
            if args.dic:
                print('---------- FirmCore Dicomposition ----------')
                FirmCore_decomposition(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator, multilayer_graph.layers_iterator, information, save=args.save)
            else:
                print('---------- FirmCore lambda = %d ----------'%args.l)
                FirmCore(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator, multilayer_graph.layers_iterator, args.l, information, save=args.save)
    
        else:
            if args.dic:
                print('---------- FirmD-Core Dicomposition ----------')
                FirmDcore_decomposition(multilayer_graph.in_adjacency_list, multilayer_graph.out_adjacency_list, multilayer_graph.nodes_iterator, multilayer_graph.layers_iterator, information, save=args.save, count=args.count)
            else:
                print('---------- FirmD-Core lambda = %d ----------'%args.l)
                FirmDcore(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator, multilayer_graph.layers_iterator, args.l, information, save=args.save, count=args.count)

    # Densest Subgraph
    elif args.m == 'densest' and args.b:
        if args.g == 'directed':
            print('---------- Directed Densest Subgraph ----------')
            Directed_densest_subgraph(multilayer_graph.in_adjacency_list, multilayer_graph.out_adjacency_list, multilayer_graph.nodes_iterator, multilayer_graph.layers_iterator, args.b, information)
            information.print_end_algorithm()

        else:
            print('---------- Densest Subgraph ----------')
            Densest_subgraph(multilayer_graph.adjacency_list, multilayer_graph.nodes_iterator, multilayer_graph.layers_iterator, args.b, information)
            information.print_end_algorithm()
 

    # dataset information
    elif args.m == 'info':
        information.print_dataset_info(multilayer_graph)

    # unknown input
    else:
        parser.print_help()









