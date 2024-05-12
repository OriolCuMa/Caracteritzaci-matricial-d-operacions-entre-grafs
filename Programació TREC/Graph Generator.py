import networkx as nx
import matplotlib.pyplot as plt
# import numpy as np
import random
import copy


class Union:
    @staticmethod
    def vertex_names(graph: list[list[int]]) -> list:
        # Assign names to all the vertices of a given graph.
        while True:
            vertex_names_list = input(
                "Enter the properly ordered vertex names list: ").split(", ")
            if len(vertex_names_list) != len(graph):
                print("The vertex names list length must the same as the \
number of vertices in the graph.")
                continue
            break
        return vertex_names_list

    @staticmethod
    def vertices_union_graph(dict1, dict2):
        # Boolean list, the elements of which are 1 if they belong both to the
        # graph1 and graph2 and 0 if they don't.
        vertices_union_graph = list(copy.deepcopy(dict2))
        for i in set(dict1).intersection(dict2):
            vertices_union_graph.remove(i)
        return [1 if name in dict1 and name in dict2 else 0
                for name in dict1 + vertices_union_graph]

    @staticmethod
    def relative_complement(graph2: list[list[int]],
                            dict1, dict2) -> list[list[int]]:
        # Returns the graph2 without the elements that belong both to graph1
        # and graph2.
        relative_complement = copy.deepcopy(graph2)
        intersection = set(dict1).intersection(dict2)
        intersection_indexes = [list(dict2).index(name)
                                for name in intersection]
        for i in sorted(intersection_indexes, reverse=True):
            relative_complement.pop(i)
            for j in range(len(relative_complement)):
                relative_complement[j].pop(i)
        return relative_complement

    @staticmethod
    def intersection(graph2: list[list], dict1, dict2) -> list[list[int]]:
        # Returns the coincident vertex matrix slices from the graph2
        intersection_list = [1 if name in set(dict1).intersection(dict2) else 0
                             for name in dict2]
        intersection_graph: list[list] = [
            [] for _ in range(sum(intersection_list))]
        for i in range(len(intersection_list)):
            for j in range(len(intersection_list)):
                if bool(intersection_list[i]
                        ) and not bool(intersection_list[j]):
                    intersection_graph[i].append(graph2[i][j])
        return intersection_graph

    @staticmethod
    def union(graph1: list[list[int]],
              graph2: list[list[int]]) -> list[list[int]]:
        union_graph = []
        dict1 = Union.vertex_names(graph1)
        dict2 = Union.vertex_names(graph2)
        if dict1 == dict2:
            return graph1
        intersection = Union.vertices_union_graph(dict1, dict2)
        # Builds the first set of rows of the resulting matrix. It has a
        # length of the number of vertices of the graph1.
        relative_complement = Union.relative_complement(graph2, dict1, dict2)
        intersection_graph = Union.intersection(graph2, dict1, dict2)
        count_1 = 0
        for graph1_row in range(len(graph1)):
            if bool(intersection[graph1_row]):
                side_vertices_1 = intersection_graph[count_1]
                count_1 += 1
            else:
                side_vertices_1 = [
                    0 for _ in range(len((relative_complement)))]
            union_rows_1 = graph1[graph1_row] + [i for i in side_vertices_1]
            union_graph.append(union_rows_1)
        # Builds the second set of rows of the resulting matrix. It has a
        # length of the number of vertices of the relative complement of the
        # graph2 related to graph1.
        for j in range(len(relative_complement)):
            count_2 = 0
            side_vertices_2 = []
            for k in range(len(graph1)):
                if bool(intersection[k]):
                    side_vertices_2.append(intersection_graph[count_2][j])
                    count_2 += 1
                else:
                    side_vertices_2.append(0)
            union_rows_2 = side_vertices_2 + relative_complement[j]
            union_graph.append(union_rows_2)
        if sum(intersection) > 1:
            for i in range(len(graph1)):
                for j in range(len(graph1[i])):
                    if intersection[i] and intersection[j] and graph1[
                            i][j] < graph2[i][j]:
                        union_graph[i][j] = graph2[i][j]
        print(union_graph)
        return union_graph


def sum_graph(graph1: list[list[int]],
              graph2: list[list[int]]) -> list[list[int]]:
    sum_graph: list[list[int]] = []  # Initialitze the sum graph
    # Iterate over the original graphs and add as many 1 to each graph row as
    # the other graph's length to have an edge between all the vertices of
    # graph1 and graph2
    for graph1_row in graph1:
        sum_graph_row_1 = graph1_row + [1 for _ in range(len(graph2))]
        sum_graph.append(sum_graph_row_1)
    for graph2_row in graph2:
        sum_graph_row_2 = [1 for _ in range(len(graph1))] + graph2_row
        sum_graph.append(sum_graph_row_2)
    return sum_graph


def complement(graph: list[list[int]]) -> list[list[int]]:
    graph_order = len(graph)  # Number of vertices
    # Initialize the complement graph with all zeros
    complement_graph = [[0] * graph_order for _ in range(graph_order)]
    # Iterate over the adjacency matrix and set matrix element to 1 if there's
    # no edge in the original graph (row != element to avoid self-loops)
    for row in range(graph_order):
        for element in range(graph_order):
            if row != element and graph[row][element] == 0:
                complement_graph[row][element] = 1
    return complement_graph


def random_graph(order: int, probability: float) -> list[list[int]]:
    G = nx.Graph()
    G.add_nodes_from(vertex for vertex in range(order))
    maximum_edges = (order * (order - 1)) / 2
    edges_added = set()
    for _ in range(int(maximum_edges)):
        vertex1 = random.randint(0, order - 1)
        vertex2 = random.randint(0, order - 1)
        if random.random() < probability and vertex1 != vertex2 and ((
                vertex1, vertex2) not in edges_added and (
                vertex2, vertex1) not in edges_added):
            G.add_edge(vertex1, vertex2)
            edges_added.add((vertex1, vertex2))
            edges_added.add((vertex2, vertex1))
    return nx.to_numpy_array(G).astype(int).tolist()


def input_matrix():
    while True:
        print("You must enter a square matrix without missing elements. \
Enter 'quit' to end the program.")
        userinput = input("Adjacency matrix: ")
        if userinput == "quit":
            exit("You quitted the program.")
        if "|" in userinput:
            matrix = [[int(element) for element in row.split()]
                      for row in userinput.split("|")]
            if any(len(row) != len(matrix[0]) for row in matrix):
                print("The matrix entered is missing elements.")
                continue
            if any(len(row) != len(matrix) for row in matrix):
                print("The matrix entered isn't square.")
                continue
            return matrix
        print("The matrix entered isn't bidirectional.")


def node_naming(graph: list[list[int]]):
    while True:
        name_chosen = input("Enter the properly ordered node names list: ")
        dictionary = name_chosen.split(", ")
        if len(dictionary) > len(graph):
            print("The node names list length should be equal or lower than \
the number of nodes in the graph.")
            continue
        break
    if name_chosen != "":
        for i in range(len(dictionary)):
            nx.relabel_nodes(graph, {i: dictionary[i]}, False)


def degree(graph):
    while True:
        node_input = input("Enter a node to know its degree (Enter 'quit' \
if you aren't interested in it): ")
        nx.convert_node_labels_to_integers(graph)
        if node_input == "quit":
            break
        try:
            node = int(node_input)
        except ValueError:
            node = node_input
        if node not in nx.nodes(graph):
            print("Enter a valid node name, please!")
            continue
        subgraph = nx.Graph()
        for i in graph.neighbors(node):
            subgraph.add_edge(node, i)
        print(f"The degree of '{node}' is: {len(subgraph)-1}")
        nx.draw_planar(subgraph, with_labels=True)
        plt.show()


def vertex_cut(graph):
    vertex_cut_list = []
    nodes = nx.nodes(graph)
    for n in nodes:
        subgraph = nx.induced_subgraph(graph, nodes-{n})
        if nx.number_connected_components(
                graph) < nx.number_connected_components(subgraph):
            vertex_cut_list.append(n)
    print(f"This graph's cut nodes are: {vertex_cut_list}")
    while True:
        vertex_cut_input = input("Enter a cut node to see the subgraph \
without it. (Enter 'quit' if you aren't interested in it): ")
        nx.convert_node_labels_to_integers(graph)
        if vertex_cut_input == "quit":
            break
        try:
            cut_node = int(vertex_cut_input)
        except ValueError:
            cut_node = vertex_cut_input
        if cut_node in vertex_cut_list:
            nx.draw_planar(nx.induced_subgraph(graph, nodes-{cut_node}),
                           with_labels=True)
            plt.show()
        else:
            print("Enter a valid input.")
