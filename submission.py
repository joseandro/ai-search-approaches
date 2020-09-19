# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

import heapq
import itertools
import os
import pickle
import math


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""
        # Heapq implementation was inspired by Python's documentation on Heapq:
        # https://docs.python.org/3.5/library/heapq.html#priority-queue-implementation-notes

        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = '<removed-task>'  # placeholder for a removed task
        self.counter = itertools.count()  # unique sequence count
        self.queue = []  # our queue

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """
        while self.queue:
            priority, count, task = heapq.heappop(self.queue)
            if task is not self.REMOVED:
                self.entry_finder[(priority, task)].pop(0)
                return [priority, task]
        raise KeyError('pop from an empty priority queue')

    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """
        entry = self.entry_finder[node].pop(0)
        entry[-1] = self.REMOVED

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __getitem__(self, item):
        return self.queue[0]

    def __str__(self):
        """Priority Queue to string."""
        str = ''
        for i in self.queue:
            str += f'[{i[0]}, {i[-1].state}] '
        return 'PQ:%s ' % str

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """
        priority = node[0]
        task = node[1]

        count = next(self.counter)
        entry = [priority, count, task]
        if node not in self.entry_finder:
            self.entry_finder[node] = []

        self.entry_finder[node].append(entry)
        heapq.heappush(self.queue, entry)

    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def contains_state(self, key):
        return key.state in [n[-1].state for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def get(self, item):
        res = None
        for n in self.queue:
            if n[-1].state == item:
                res = n[-1]
                break
        return res

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """

        return self.queue[0]


class Node:
    def __init__(self, state, action, cost, parent):
        self.state = state
        self.action = action
        self.cost = cost
        self.parent = parent

    def __str__(self):
        return '{cost:' + str(self.cost) + ', state:' + str(self.state) + '}'

    def __contains__(self, key):
        return key == self.state


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []

    explored = []
    tree = []
    root = Node(state=start,
                action=None,
                cost=0,
                parent=None)
    frontier = PriorityQueue()
    frontier.append((0, root))
    best_cost = float("inf")
    while True:
        if frontier.size() == 0:
            raise KeyError('frontier is empty, search failed')
        element = frontier.pop()
        weight = element[0]
        node = element[-1]
        S = node.state
        if S in explored:
            continue

        explored.append(S)
        if S == goal:
            path = []
            while node.parent is not None:
                path.append(node.state)
                node = node.parent
            path.append(start)
            path.reverse()
            return path
        w = weight + 1
        if w < best_cost:
            actions = sorted(graph[S])
            for a in actions:
                leaf = Node(state=a,
                            action=S + a,
                            cost=w,
                            parent=node)
                if leaf not in frontier and a not in explored:
                    frontier.append((w, leaf))
                    tree.append(leaf)
                    if a == goal:
                        best_cost = w


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if start == goal:
        return []

    explored = []
    tree = []
    root = Node(state=start,
                action=None,
                cost=0,
                parent=None)
    frontier = PriorityQueue()
    frontier.append((0, root))
    best_cost = float("inf")
    while True:
        if frontier.size() == 0:
            raise KeyError('frontier is empty, search failed')
        element = frontier.pop()
        weight = element[0]
        node = element[-1]
        S = node.state
        if S in explored:
            continue
        explored.append(S)
        if S == goal:
            path = []
            while node.parent is not None:
                path.append(node.state)
                node = node.parent
            path.append(start)
            path.reverse()
            return path

        if weight < best_cost:
            actions = graph[S]
            for a in actions:
                w = weight + graph.get_edge_weight(S, a)
                if w < best_cost:
                    leaf = Node(state=a,
                                action=S + a,
                                cost=w,
                                parent=node)
                    if leaf not in frontier and a not in explored:
                        frontier.append((w, leaf))
                        tree.append(leaf)
                        if a == goal:
                            best_cost = w


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """
    return math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(graph.nodes[v]['pos'], graph.nodes[goal]['pos'])))


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if start == goal:
        return []

    explored = []
    tree = []
    root = Node(state=start,
                action=None,
                cost=heuristic(graph, start, goal),
                parent=None)
    frontier = PriorityQueue()
    frontier.append((0, root))
    best_cost = float("inf")
    while True:
        if frontier.size() == 0:
            raise KeyError('frontier is empty, search failed')
        element = frontier.pop()
        node = element[-1]
        S = node.state
        if S in explored:
            continue
        explored.append(S)
        if S == goal:
            path = []
            while node.parent is not None:
                path.append(node.state)
                node = node.parent
            path.append(start)
            path.reverse()
            return path
        weight = element[0] - heuristic(graph, S, goal)
        if weight < best_cost:
            actions = graph[S]
            for a in actions:
                w = (weight + graph.get_edge_weight(S, a))
                if w < best_cost:
                    w += heuristic(graph, a, goal)
                    leaf = Node(state=a,
                                action=S + a,
                                cost=w,
                                parent=node)
                    if leaf not in frontier and a not in explored:
                        frontier.append((w, leaf))
                        tree.append(leaf)
                        if a == goal:
                            best_cost = w - heuristic(graph, a, goal)


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.
    See README.md for exercise description.
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []

    explored_agent_1 = []
    explored_agent_2 = []
    tree = []
    beginning = Node(state=start,
                     action=None,
                     cost=0,
                     parent=None)

    frontier_agent_1 = PriorityQueue()
    frontier_agent_1.append((0, beginning))

    frontier_agent_2 = PriorityQueue()
    end = Node(state=goal,
               action=None,
               cost=0,
               parent=None)
    frontier_agent_2.append((0, end))
    best_cost = float("inf")
    path = []
    while True:
        if frontier_agent_1.size() == 0 or frontier_agent_2.size() == 0:
            print('Frontier is zero!', frontier_agent_1, frontier_agent_2)
            return path

        element = frontier_agent_1.pop()
        weight = element[0]
        node = element[-1]
        S = node.state
        if S in explored_agent_2:
            return path

        if S in explored_agent_1:
            continue
        explored_agent_1.append(S)
        intersection = frontier_agent_2.get(S)
        if intersection is not None:
            if (best_cost >= (intersection.cost + node.cost)):
                best_cost = intersection.cost + node.cost
                path = []
                aux_node = node
                while aux_node.parent is not None:
                    path.append(aux_node.state)
                    aux_node = aux_node.parent
                path.append(start)
                path.reverse()
                if intersection.parent is not None:
                    intersection = intersection.parent
                    while intersection.parent is not None:
                        path.append(intersection.state)
                        intersection = intersection.parent
                    path.append(goal)

        if weight < best_cost:
            actions = graph[S]
            for a in actions:
                w = weight + graph.get_edge_weight(S, a)
                if w < best_cost:
                    leaf = Node(state=a,
                                action=S + a,
                                cost=w,
                                parent=node)
                    if leaf not in frontier_agent_1 and a not in explored_agent_1:
                        frontier_agent_1.append((w, leaf))
                        tree.append(leaf)
                        if a == goal:
                            best_cost = w

        element = frontier_agent_2.pop()
        weight = element[0]
        node = element[-1]
        S = node.state
        if S in explored_agent_1:
            return path

        if S in explored_agent_2:
            continue
        explored_agent_2.append(S)

        intersection = frontier_agent_1.get(S)
        if intersection is not None:
            if (best_cost >= (intersection.cost + node.cost)):
                best_cost = intersection.cost + node.cost
                path = []
                aux_node = node
                while intersection.parent is not None:
                    path.append(intersection.state)
                    intersection = intersection.parent
                path.append(start)
                path.reverse()
                if aux_node.parent is not None:
                    aux_node = aux_node.parent
                    while aux_node.parent is not None:
                        path.append(aux_node.state)
                        aux_node = aux_node.parent
                    path.append(goal)

        if weight < best_cost:
            actions = graph[S]
            for a in actions:
                w = weight + graph.get_edge_weight(S, a)
                if w < best_cost:
                    leaf = Node(state=a,
                                action=S + a,
                                cost=w,
                                parent=node)
                    if leaf not in frontier_agent_2 and a not in explored_agent_2:
                        frontier_agent_2.append((w, leaf))
                        tree.append(leaf)
                        if a == goal:
                            best_cost = w

def bi_a_star_process_intersection(node, explored, frontier, best_cost):
    S = node.state
    cost = float('inf')
    path = []
    explored.append(node)
    intersection = frontier.get(S)
    if intersection is not None:
        if best_cost >= (intersection.cost + node.cost):
            cost = intersection.cost + node.cost
            aux_node = node
            while aux_node.parent is not None:
                path.append(aux_node.state)
                aux_node = aux_node.parent
            path.append(aux_node.state)
            path.reverse()
            if intersection.parent is not None:
                intersection = intersection.parent
                while intersection.parent is not None:
                    path.append(intersection.state)
                    intersection = intersection.parent
                path.append(intersection.state)
    return cost, path

def bi_a_star_expand_node(graph, tree, node, frontier, explored, goal, best_cost, heuristic):
    weight = node.cost
    cost = None
    if weight < best_cost:
        actions = graph[node.state]
        for a in actions:
            w = weight + graph.get_edge_weight(node.state, a)
            if w < best_cost:
                leaf = Node(state=a,
                            action=node.state + a,
                            cost=w,
                            parent=node)
                i = heuristic(graph, a, goal)
                w += i
                if leaf not in frontier and get_node_from_explored_set(node=leaf, explored=explored) is None:
                    frontier.append((w, leaf))
                    tree.append(leaf)
                    if a == goal:
                        cost = leaf.cost
    return cost

def get_node_from_explored_set(node, explored):
    intersection = None
    for n in explored:
        if n.state == node.state:
            intersection = n
            break
    return intersection

def bi_a_star_get_path_from_intersection(node, explored):
    intersection = get_node_from_explored_set(node, explored)
    if intersection is None:
        return float('inf'), []
    path = []
    cost = 0
    aux_node = node
    cost = aux_node.cost
    while aux_node.parent is not None:
        path.append(aux_node.state)
        aux_node = aux_node.parent
    path.append(aux_node.state)
    path.reverse()

    cost += intersection.cost
    intersection = intersection.parent
    if intersection is not None:
        while intersection.parent is not None:
            path.append(intersection.state)
            intersection = intersection.parent
        path.append(intersection.state)

    return cost, path

def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if start == goal:
        return []

    explored_agent_1 = []
    explored_agent_2 = []
    tree = []
    beginning = Node(state=start,
                     action=None,
                     cost=0,
                     parent=None)

    frontier_agent_1 = PriorityQueue()
    frontier_agent_1.append((0, beginning))

    frontier_agent_2 = PriorityQueue()
    end = Node(state=goal,
               action=None,
               cost=0,
               parent=None)
    frontier_agent_2.append((0, end))
    best_cost = float("inf")
    path = []
    while True:
        if frontier_agent_1.size() == 0 or frontier_agent_2.size() == 0:
            print('Frontier is zero!', frontier_agent_1, frontier_agent_2)
            return path

        element = frontier_agent_1.pop()
        node = element[-1]
        S = node.state
        cost, ret = bi_a_star_get_path_from_intersection(node=node,
                                                   explored=explored_agent_2)
        if best_cost > cost:
            return ret
        if len(ret) > 0:
            return path

        if S not in explored_agent_1:
            cost, aux_p = bi_a_star_process_intersection(node=node,
                                                         explored=explored_agent_1,
                                                         frontier=frontier_agent_2,
                                                         best_cost=best_cost)
            if best_cost >= cost:
                best_cost = cost
                path = aux_p

            cost = bi_a_star_expand_node(graph=graph,
                                         tree=tree,
                                         node=node,
                                         goal=goal,
                                         frontier=frontier_agent_1,
                                         explored=explored_agent_1,
                                         best_cost=best_cost,
                                         heuristic=heuristic)
            if cost is not None:
                best_cost = cost


        element = frontier_agent_2.pop()
        node = element[-1]
        S = node.state
        cost, ret = bi_a_star_get_path_from_intersection(node=node,
                                                         explored=explored_agent_1)
        if best_cost > cost:
            ret.reverse()
            return ret
        if len(ret) > 0:
            return path

        if S not in explored_agent_2:
            cost, aux_p = bi_a_star_process_intersection(node=node,
                                                         explored=explored_agent_2,
                                                         frontier=frontier_agent_1,
                                                         best_cost=best_cost)
            if best_cost >= cost:
                best_cost = cost
                aux_p.reverse()
                path = aux_p

            cost = bi_a_star_expand_node(graph=graph,
                                         tree=tree,
                                         node=node,
                                         goal=start,
                                         frontier=frontier_agent_2,
                                         explored=explored_agent_2,
                                         best_cost=best_cost,
                                         heuristic=heuristic)
            if cost is not None:
                best_cost = cost


def get_path(found_paths):
    print('get_path =  ', found_paths)
    ordered_paths = {k: v for k, v in sorted(found_paths.items(), key=lambda item: item[1]['cost'])}
    iterator = iter(ordered_paths.items())
    path_a = next(iterator)[1]['path']
    path_b = next(iterator)[1]['path']
    print('new a', path_a)
    print('new b', path_b)
    if path_a[-1] == path_b[0]:
        print('first: ', path_a + path_b[1:])
        return path_a + path_b[1:]

    if path_b[-1] == path_a[0]:
        print('second: ', path_b + path_a[1:])
        return path_b + path_a[1:]

    path_a_rev = path_a.copy()
    path_a_rev.reverse()
    if path_a_rev[-1] == path_b[0]:
        print('third: ', path_a_rev + path_b[1:])
        return path_a_rev + path_b[1:]

    if path_b[-1] == path_a_rev[0]:
        print('fourth: ', path_b + path_a_rev[1:])
        return path_b + path_a_rev[1:]

    path_b_rev = path_b.copy()
    path_b_rev.reverse()
    if path_a[-1] == path_b_rev[0]:
        print('fifth: ', path_a + path_b_rev[1:])
        return path_a + path_b_rev[1:]

    if path_b_rev[-1] == path_a[0]:
        print('sixth: ', path_b_rev + path_a[1:])
        return path_b_rev + path_a[1:]

    return []


def process_node(frontier, node, id, start, goal, found_paths):
    if found_paths[id]['reached_goal'] is True:
        return

    S = node.state
    intersection = frontier.get(S)
    if intersection is not None:
        cost = node.cost + intersection.cost
        if cost <= found_paths[id]['cost']:  # we have still not found this goal
            found_paths[id]['cost'] = intersection.cost + node.cost
            # we reached our goal
            if node.state == goal:
                found_paths[id]['reached_goal'] = True
                print('goal found in ', id)

            path = []
            aux_node = node
            while aux_node.parent is not None:
                path.append(aux_node.state)
                aux_node = aux_node.parent
            path.append(start)
            path.reverse()
            if intersection.parent is not None:
                intersection = intersection.parent
                while intersection.parent is not None:
                    path.append(intersection.state)
                    intersection = intersection.parent
            path.append(goal)
            found_paths[id]['path'] = path


def node_expansion(graph, tree, node, frontier, explored, weight, found_paths, id_a, id_b, goal_a, goal_b):
    if found_paths[id_a]['reached_goal'] is True and found_paths[id_b]['reached_goal'] is True:
        print('path is already fully explored!')
        return

    S = node.state
    if found_paths[id_a]['cost'] > found_paths[id_b]['cost']:
        max_cost = found_paths[id_a]['cost']
    else:
        max_cost = found_paths[id_b]['cost']
    if weight < max_cost:
        actions = graph[S]
        for a in actions:
            w = weight + graph.get_edge_weight(S, a)
            if w < max_cost:
                leaf = Node(state=a,
                            action=S + a,
                            cost=w,
                            parent=node)
                if leaf not in frontier and leaf not in explored:
                    frontier.append((w, leaf))
                    tree.append(leaf)
                    if a == goal_a:
                        found_paths[id_a]['cost'] = w
                        found_paths[id_a]['reached_goal'] = True
                    elif a == goal_b:
                        found_paths[id_b]['cost'] = w
                        found_paths[id_b]['reached_goal'] = True


def inspect_explored_intersection(explored, node, path_id, found_paths):
    res = []
    for n in explored:
        if n.state == node.state:
            res.append(n)
    if len(res) > 0:
        intersection = res.pop()
        if found_paths[path_id]['cost'] >= intersection.cost + node.cost:
            found_paths[path_id]['cost'] = intersection.cost + node.cost
            path = []
            aux_node = node
            while aux_node.parent is not None:
                path.append(aux_node.state)
                aux_node = aux_node.parent
            path.append(aux_node.state)
            path.reverse()
            if intersection.parent is not None:
                intersection = intersection.parent
            while intersection.parent is not None:
                if intersection.state not in path:
                    path.append(intersection.state)
                intersection = intersection.parent
            if intersection.state not in path:
                path.append(intersection.state)
            found_paths[path_id]['path'] = path
            found_paths[path_id]['reached_goal'] = True


def is_state_unexplored(explored, state):
    for n in explored:
        if n.state == state:
            return False
    return True


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    print('Original: ', goals)
    if goals[0] == goals[1] and goals[1] == goals[2]:
        return []

    explored_agent_1 = []
    explored_agent_2 = []
    explored_agent_3 = []
    start_agent_1 = goals[0]
    start_agent_2 = goals[1]
    start_agent_3 = goals[2]
    tree = []
    start_node_agent_1 = Node(state=start_agent_1,
                              action=None,
                              cost=0,
                              parent=None)

    frontier_agent_1 = PriorityQueue()
    frontier_agent_1.append((0, start_node_agent_1))

    frontier_agent_2 = PriorityQueue()
    start_node_agent_2 = Node(state=start_agent_2,
                              action=None,
                              cost=0,
                              parent=None)
    frontier_agent_2.append((0, start_node_agent_2))

    frontier_agent_3 = PriorityQueue()
    start_node_agent_3 = Node(state=start_agent_3,
                              action=None,
                              cost=0,
                              parent=None)
    frontier_agent_3.append((0, start_node_agent_3))

    found_paths = {
        '12': {
            'path': [],
            'cost': float('inf'),
            'reached_goal': False
        },
        '13': {
            'path': [],
            'cost': float('inf'),
            'reached_goal': False
        },
        '23': {
            'path': [],
            'cost': float('inf'),
            'reached_goal': False
        }
    }

    while True:
        if (found_paths['12']['reached_goal'] is True and found_paths['13']['reached_goal'] is True) or \
           (found_paths['23']['reached_goal'] is True and found_paths['13']['reached_goal'] is True) or \
           (found_paths['12']['reached_goal'] is True and found_paths['23']['reached_goal'] is True):
            print('All best paths were found!')
            return get_path(found_paths)

        if frontier_agent_1.size() == 0 or \
                frontier_agent_2.size() == 0 or \
                frontier_agent_3.size() == 0:
            print('Frontier is zero!', frontier_agent_1, frontier_agent_2, frontier_agent_3)
            return get_path(found_paths)

        # Agent 1 logic
        element = frontier_agent_1.pop()
        node = element[-1]
        weight = element[0]
        inspect_explored_intersection(explored=explored_agent_2,
                                      node=node,
                                      path_id='12',
                                      found_paths=found_paths)

        inspect_explored_intersection(explored=explored_agent_3,
                                      node=node,
                                      path_id='13',
                                      found_paths=found_paths)

        if is_state_unexplored(explored_agent_1, node.state):
            explored_agent_1.append(node)
            # Frontier processing
            process_node(frontier=frontier_agent_2,
                         node=node,
                         id='12',
                         start=start_agent_1,
                         goal=start_agent_2,
                         found_paths=found_paths)

            process_node(frontier=frontier_agent_3,
                         node=node,
                         id='13',
                         start=start_agent_1,
                         goal=start_agent_3,
                         found_paths=found_paths)

            # Node expansion
            node_expansion(graph=graph,
                           tree=tree,
                           node=node,
                           frontier=frontier_agent_1,
                           explored=explored_agent_1,
                           weight=weight,
                           found_paths=found_paths,
                           id_a='12',
                           id_b='13',
                           goal_a=start_node_agent_2,
                           goal_b=start_node_agent_3)

        # Agent 2 logic
        element = frontier_agent_2.pop()
        node = element[-1]
        weight = element[0]
        inspect_explored_intersection(explored=explored_agent_1,
                                      node=node,
                                      path_id='12',
                                      found_paths=found_paths)

        inspect_explored_intersection(explored=explored_agent_3,
                                      node=node,
                                      path_id='23',
                                      found_paths=found_paths)

        if is_state_unexplored(explored_agent_2, node.state):
            explored_agent_2.append(node)
            # Frontier processing
            process_node(frontier=frontier_agent_1,
                         node=node,
                         id='12',
                         start=start_agent_2,
                         goal=start_agent_1,
                         found_paths=found_paths)

            process_node(frontier=frontier_agent_3,
                         node=node,
                         id='23',
                         start=start_agent_2,
                         goal=start_agent_3,
                         found_paths=found_paths)

            # Node expansion
            node_expansion(graph=graph,
                           tree=tree,
                           node=node,
                           frontier=frontier_agent_2,
                           explored=explored_agent_2,
                           weight=weight,
                           found_paths=found_paths,
                           id_a='12',
                           id_b='23',
                           goal_a=start_node_agent_1,
                           goal_b=start_node_agent_3)

        # Agent 3 logic
        element = frontier_agent_3.pop()
        node = element[-1]
        weight = element[0]
        inspect_explored_intersection(explored=explored_agent_2,
                                      node=node,
                                      path_id='23',
                                      found_paths=found_paths)

        inspect_explored_intersection(explored=explored_agent_1,
                                      node=node,
                                      path_id='13',
                                      found_paths=found_paths)

        if is_state_unexplored(explored_agent_3, node.state):
            explored_agent_3.append(node)
            # Frontier processing
            process_node(frontier=frontier_agent_2,
                         node=node,
                         id='23',
                         start=start_agent_3,
                         goal=start_agent_2,
                         found_paths=found_paths)

            process_node(frontier=frontier_agent_1,
                         node=node,
                         id='13',
                         start=start_agent_3,
                         goal=start_agent_1,
                         found_paths=found_paths)

            # Node expansion
            node_expansion(graph=graph,
                           tree=tree,
                           node=node,
                           frontier=frontier_agent_3,
                           explored=explored_agent_3,
                           weight=weight,
                           found_paths=found_paths,
                           id_a='23',
                           id_b='13',
                           goal_a=start_node_agent_2,
                           goal_b=start_node_agent_1)


def node_expansion_with_heuristic(graph, tree, node, frontier, explored, found_paths, id_a, id_b, goal_a,
                                  goal_b, heuristic):
    if found_paths[id_a]['reached_goal'] is True and found_paths[id_b]['reached_goal'] is True:
        print('path is already fully explored! ', found_paths[id_a]['path'], found_paths[id_b]['path'])
        return

    S = node.state
    actions = graph[S]
    for a in actions:
        for weight_id in [id_a, id_b]:
            w = node.cost + graph.get_edge_weight(S, a)
            if found_paths[weight_id]['reached_goal'] is True:
                continue

            if weight_id == id_a:
                goal = goal_a
            else:
                goal = goal_b

            if w < found_paths[weight_id]['cost']:
                leaf = Node(state=a,
                            action=S + a,
                            cost=w,
                            parent=node)
                if leaf not in frontier and leaf not in explored:# is_state_unexplored(explored, a):
                    w += heuristic(graph, a, goal.state)
                    frontier.append((w, leaf))
                    tree.append(leaf)
                    if a == goal.state:
                        found_paths[weight_id]['cost'] = w - heuristic(graph, a, goal.state)

def improved_inspect_explored_intersection(explored, node, path_id, found_paths):
    res = []
    for n in explored:
        if n.state == node.state:
            res.append(n)
    if len(res) > 0:
        intersection = res.pop()
        if found_paths[path_id]['cost'] >= intersection.cost + node.cost:
            found_paths[path_id]['cost'] = intersection.cost + node.cost
            path = []
            aux_node = node
            while aux_node.parent is not None:
                path.append(aux_node.state)
                aux_node = aux_node.parent
            path.append(aux_node.state)
            path.reverse()
            if intersection.parent is not None:
                intersection = intersection.parent
            while intersection.parent is not None:
                if intersection.state not in path:
                    path.append(intersection.state)
                intersection = intersection.parent
            if intersection.state not in path:
                path.append(intersection.state)
            found_paths[path_id]['path'] = path
            found_paths[path_id]['reached_goal'] = True

def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic, landmarks=None):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
        landmarks: Iterable containing landmarks pre-computed in compute_landmarks()
            Default: None

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    print('Goals given: ', goals)
    if goals[0] == goals[1] and goals[1] == goals[2]:
        return []

    explored_agent_1 = []
    explored_agent_2 = []
    explored_agent_3 = []
    start_agent_1 = goals[0]
    start_agent_2 = goals[1]
    start_agent_3 = goals[2]
    tree = []
    start_node_agent_1 = Node(state=start_agent_1,
                              action=None,
                              cost=0,
                              parent=None)

    frontier_agent_1 = PriorityQueue()
    frontier_agent_1.append((0, start_node_agent_1))

    frontier_agent_2 = PriorityQueue()
    start_node_agent_2 = Node(state=start_agent_2,
                              action=None,
                              cost=0,
                              parent=None)
    frontier_agent_2.append((0, start_node_agent_2))

    frontier_agent_3 = PriorityQueue()
    start_node_agent_3 = Node(state=start_agent_3,
                              action=None,
                              cost=0,
                              parent=None)
    frontier_agent_3.append((0, start_node_agent_3))

    found_paths = {
        '12': {
            'path': [],
            'cost': float('inf'),
            'reached_goal': False
        },
        '13': {
            'path': [],
            'cost': float('inf'),
            'reached_goal': False
        },
        '23': {
            'path': [],
            'cost': float('inf'),
            'reached_goal': False
        }
    }

    while True:
        if (found_paths['12']['reached_goal'] is True and found_paths['13']['reached_goal'] is True) or \
                (found_paths['23']['reached_goal'] is True and found_paths['13']['reached_goal'] is True) or \
                (found_paths['12']['reached_goal'] is True and found_paths['23']['reached_goal'] is True):
        # if (found_paths['12']['reached_goal'] is True and found_paths['13']['reached_goal'] is True and found_paths['23']['reached_goal'] is True):
            print('All best paths were found!')
            return get_path(found_paths)

        if frontier_agent_1.size() == 0 or \
                frontier_agent_2.size() == 0 or \
                frontier_agent_3.size() == 0:
            print('Frontier is zero!', frontier_agent_1, frontier_agent_2, frontier_agent_3)
            return get_path(found_paths)

        # Agent 1 logic
        element = frontier_agent_1.pop()
        node = element[-1]
        improved_inspect_explored_intersection(explored=explored_agent_2,
                                      node=node,
                                      path_id='12',
                                      found_paths=found_paths)
        improved_inspect_explored_intersection(explored=explored_agent_3,
                                      node=node,
                                      path_id='13',
                                      found_paths=found_paths)
        if is_state_unexplored(explored_agent_1, node.state):
            explored_agent_1.append(node)
            # Frontier processing
            process_node(frontier=frontier_agent_2,
                         node=node,
                         id='12',
                         start=start_agent_1,
                         goal=start_agent_2,
                         found_paths=found_paths)
            process_node(frontier=frontier_agent_3,
                         node=node,
                         id='13',
                         start=start_agent_1,
                         goal=start_agent_3,
                         found_paths=found_paths)
            # Node expansion
            node_expansion_with_heuristic(graph=graph,
                                          tree=tree,
                                          node=node,
                                          frontier=frontier_agent_1,
                                          explored=explored_agent_1,
                                          found_paths=found_paths,
                                          id_a='12',
                                          id_b='13',
                                          goal_a=start_node_agent_2,
                                          goal_b=start_node_agent_3,
                                          heuristic=heuristic)
        # Agent 2 logic
        element = frontier_agent_2.pop()
        node = element[-1]
        improved_inspect_explored_intersection(explored=explored_agent_1,
                                      node=node,
                                      path_id='12',
                                      found_paths=found_paths)
        improved_inspect_explored_intersection(explored=explored_agent_3,
                                      node=node,
                                      path_id='23',
                                      found_paths=found_paths)
        if is_state_unexplored(explored_agent_2, node.state):
            explored_agent_2.append(node)
            # Frontier processing
            process_node(frontier=frontier_agent_1,
                         node=node,
                         id='12',
                         start=start_agent_2,
                         goal=start_agent_1,
                         found_paths=found_paths)
            process_node(frontier=frontier_agent_3,
                         node=node,
                         id='23',
                         start=start_agent_2,
                         goal=start_agent_3,
                         found_paths=found_paths)
            # Node expansion
            node_expansion_with_heuristic(graph=graph,
                                          tree=tree,
                                          node=node,
                                          frontier=frontier_agent_2,
                                          explored=explored_agent_2,
                                          found_paths=found_paths,
                                          id_a='12',
                                          id_b='23',
                                          goal_a=start_node_agent_1,
                                          goal_b=start_node_agent_3,
                                          heuristic=heuristic)

        # Agent 3 logic
        element = frontier_agent_3.pop()
        node = element[-1]
        improved_inspect_explored_intersection(explored=explored_agent_2,
                                      node=node,
                                      path_id='23',
                                      found_paths=found_paths)
        improved_inspect_explored_intersection(explored=explored_agent_1,
                                      node=node,
                                      path_id='13',
                                      found_paths=found_paths)
        if is_state_unexplored(explored_agent_3, node.state):
            explored_agent_3.append(node)
            # Frontier processing
            process_node(frontier=frontier_agent_2,
                         node=node,
                         id='23',
                         start=start_agent_3,
                         goal=start_agent_2,
                         found_paths=found_paths)
            process_node(frontier=frontier_agent_1,
                         node=node,
                         id='13',
                         start=start_agent_3,
                         goal=start_agent_1,
                         found_paths=found_paths)
            # Node expansion
            node_expansion_with_heuristic(graph=graph,
                                          tree=tree,
                                          node=node,
                                          frontier=frontier_agent_3,
                                          explored=explored_agent_3,
                                          found_paths=found_paths,
                                          id_a='23',
                                          id_b='13',
                                          goal_a=start_node_agent_2,
                                          goal_b=start_node_agent_1,
                                          heuristic=heuristic)


def return_your_name():
    """Return your name from this function"""
    return "Joseandro Marques Oliveira Luiz"


def compute_landmarks(graph):
    """
    Feel free to implement this method for computing landmarks. We will call
    tridirectional_upgraded() with the object returned from this function.

    Args:
        graph (ExplorableGraph): Undirected graph to search.

    Returns:
    List with not more than 4 computed landmarks. 
    """
    return None


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """
    pass


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to Gradescope, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None


def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    # Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    # Now we want to execute portions of the formula:
    constOutFront = 2 * 6371  # Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0] - vLatLong[0]) / 2)) ** 2  # First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0]) * math.cos(goalLatLong[0]) * (
            (math.sin((goalLatLong[1] - vLatLong[1]) / 2)) ** 2)  # Second term
    return constOutFront * math.asin(math.sqrt(term1InSqrt + term2InSqrt))  # Straight application of formula
