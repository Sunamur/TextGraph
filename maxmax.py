import networkx as nx
import numpy as np


def maxmax_clustering(graph):
    surviving_edges=[]
    for i in graph.nodes:
        sorted_edges=sorted([[x[1],x[2]['weight']] for x in graph.out_edges(i,data=True)],key=lambda x: x[1],reverse=True)
        surviving_edges.append((sorted_edges[0][0], i))
    g = nx.from_edgelist(surviving_edges, create_using=nx.DiGraph)
    nx.set_node_attributes(g, {x:True for x in g.nodes}, 'root')
    for i in g.nodes:
        if g.nodes[i]['root']:
            nx.set_node_attributes(g, {x:False for x in nx.descendants(g, i)}, 'root')
    root_nodes = [x for x in g.nodes if g.nodes[x]['root']]
    clusters = {x:root for root in root_nodes for x in nx.descendants(g,root)}
    clusters.update({x:x for x in root_nodes})
    return clusters
