# coding: utf-8

# # Assignment 2: Graph Search
# In this assignment you will be implementing a variety of graph search
# algorithms, with the eventual goal of solving tri-directional search.
#
# Before you start, you will need:
#
# 1. [networkx](http://networkx.github.io/), which is a package for processing
# networks. This assignment will be easier if you take some time to test out
# and get familiar with the [basic methods](https://networkx.github.io/examples.html)
# of networkx. We have provided a version of networkx for you to use. It is in
# the lib folder. Please only use that version. If you have installed networkx
# already, run this code on a virtualenv without networkx installed. What's a
# virtualenv you say? See [this](http://docs.python-guide.org/en/latest/dev/virtualenvs/).
#
# 2. [matplotlib](http://matplotlib.org/downloads.html) for basic network
# visualization. You're free to use your own version :)
#
# 3. [Python 2.7.x](https://www.python.org/downloads/) (in case you're on
# Python 3).
#
# We will be using two undirected networks for this assignment: a simplified
# [map](http://robotics.cs.tamu.edu/dshell/cs420/images/map.jpg) of Romania
# (from Russell and Norvig) and a full street map of Atlanta.
#
# A read-only version of this notebook can be found [here](https://github.gatech.edu/omscs6601/assignment_2/blob/master/search_notebook.ipynb).

from __future__ import division

import pickle
import random

import matplotlib.pyplot as plt
import networkx

from ExplorableGraph import ExplorableGraph
from osm2networkx import *
from search_submission import *
from visualize_graph import plot_search

"""Romania map data from Russell and Norvig, Chapter 3."""
romania = pickle.load(open('romania_graph.pickle', 'rb'))
romania = ExplorableGraph(romania)
romania.reset_search()
print romania['a']


def check_pq():
    pq = PriorityQueue()
    temp_list = []

    for i in range(10):
        a = random.randint(0, 10000)
        pq.append((a, 'a'))
        temp_list.append(a)

    temp_list = sorted(temp_list)

    for i in temp_list:
        j = pq.pop()
        if not i == j[0]:
            return False

    return True


check_pq()


# This function exists to help you visually debug your code.
# Feel free to modify it in any way you like.
# graph should be an ExplorableGraph which contains a networkx graph
# node_positions should be a dictionary mapping nodes to x,y coordinates
# IMP - This function may modify the graph you pass to it.
def draw_graph(graph, node_positions={}, start=None, goal=None, path=[]):
    explored = list(graph.explored_nodes)

    labels = {}
    for node in graph:
        labels[node] = node

    if not node_positions:
        node_positions = networkx.spring_layout(graph)

    networkx.draw_networkx_nodes(graph, node_positions)
    networkx.draw_networkx_edges(graph, node_positions, style='dashed')
    networkx.draw_networkx_labels(graph, node_positions, labels)

    networkx.draw_networkx_nodes(graph, node_positions, nodelist=explored,
                                 node_color='g')

    if path:
        edges = [(path[i], path[i + 1]) for i in range(0, len(path) - 1)]
        networkx.draw_networkx_edges(graph, node_positions, edgelist=edges,
                                     edge_color='b')

    if start:
        networkx.draw_networkx_nodes(graph, node_positions, nodelist=[start],
                                     node_color='b')

    if goal:
        networkx.draw_networkx_nodes(graph, node_positions, nodelist=[goal],
                                     node_color='y')

    plt.plot()
    plt.show()


"""Testing and visualizing breadth-first search
in the notebook."""
start = 'a'
goal = 'u'

node_positions = {n: romania.node[n]['pos'] for n in romania.node.keys()}

romania.reset_search()
path = breadth_first_search(romania, start, goal)

draw_graph(romania, node_positions=node_positions, start=start, goal=goal,
           path=path)


"""Loading Atlanta map data."""
atlanta = pickle.load(open('atlanta_osm.pickle', 'rb'))
atlanta = ExplorableGraph(atlanta)
atlanta.reset_search()


# Visualizing search results
# ---
# When using a geographic network, you may want to visualize your searches. We
# can do this by converting the search results to a [GeoJSON](https://en.wikipedia.org/wiki/GeoJSON)
# file which we then visualize on [Gist](https://gist.github.com/) by
# [importing](https://github.com/blog/1576-gist-meets-geojson) the file.
#
# We provide a method for doing this in visualize_graph.py called plot_search()
# which takes as parameters the graph, the name of the file to write, the nodes
# on the path, and the set of all nodes explored. This produces a GeoJSON file
# named as specified, which you can upload to Gist to visualize the search path
# and explored nodes.


"""Example of how to visualize search results
with two sample nodes in Atlanta."""
# NOTE: *** Please complete the  bidirectional_ucs before this.***
# You can try visualization with any other search methods completed above too.
atlanta.reset_search()
path = bidirectional_ucs(atlanta, '69244359', '557989279')
all_explored = atlanta.get_explored_nodes()
plot_search(atlanta, 'atlanta_search.json', path, all_explored)
# then upload 'atlanta_search.json' to Gist


# Race!
# ---
# Here's your chance to show us your best stuff. This part is mandatory if you want to compete in the race for extra credit. Implement custom_search() using whatever strategy you like. Your search should be tri-directional and it'll be tested on the Atlanta map only.

# In[ ]:


def custom_search(graph, goals):
    """Run your best tridirectional search between
    goals, and return the path."""
    raise NotImplementedError
    # return path
