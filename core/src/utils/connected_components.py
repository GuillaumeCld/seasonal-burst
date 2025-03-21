from collections import defaultdict, deque

import numpy as np
from utils.operators import overlapping


def neighbors(i: int, j: int, nx: int, ny: int):
    """
    Find the indices of all neighbors of a given cell in a grid.

    Parameters:
        i: integer, row index
        j: integer, column index
        nx: integer, number of rows
        ny: integer, number of columns

    Returns:
        list of tuples with neighbors' indices
    """
    neighbors = []
    for x in range(i - 1, i + 2):
        for y in range(j - 1, j + 2):
            if 0 <= x < nx and 0 <= y < ny and (x != i or y != j):
                neighbors.append((x, y))

    return neighbors


def find_connected_components(vertices: list, edges: list, directed: bool = False):
    """
    Compute the connected components of a graph using a BFS algorithm  given its vertices and edges.

    Parameters:
        vertices (list): list of vertices
        edges (list): list of edges
        directed (bool): directed or undirected graph

    Returns:
        list of connected components
    """
    # Step 1: Build the adjacency list
    graph = defaultdict(list)
    for v1, v2 in edges:
        graph[v1].append(v2)
        # We do not add v2 -> v1 for directed graphs
        if not directed:
            graph[v2].append(v1)

    # Step 2: Initialize data structures
    visited = set()
    components = []

    # Step 3: Define a BFS to explore each component
    def bfs(start_vertex):
        component = []
        queue = deque([start_vertex])
        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                visited.add(vertex)
                component.append(vertex)
                for neighbor in graph[vertex]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        return component

    # Step 4: Find all connected components
    for vertex in vertices:
        if vertex not in visited:
            component = bfs(vertex)
            components.append(component)

    return components


def connected_components_bursts(
    grid_bursts: np.array, min_overlap: int = 1, directed: bool = False
):
    """
    Compute the connected components of the bursts in a grid.

    Parameters:
        grid_bursts (np.array): grid of bursts
        min_overlap (int): minimum number of overlapping observations
        directed (bool): directed or undirected graph
    """

    nx, ny = grid_bursts.shape
    grid_mark = np.full((nx, ny), False, dtype=bool)

    edges = []
    vertices = []

    for i in range(nx):
        for j in range(ny):
            for i_burst, burst in enumerate(grid_bursts[i, j]):
                for neighbor in neighbors(i, j, nx, ny):
                    for i_nn_burst, neighbor_burst in enumerate(grid_bursts[neighbor]):
                        if grid_mark[neighbor]:
                            continue

                        if overlapping(burst, neighbor_burst, min_overlap):
                            e1 = (i, j, i_burst)
                            e2 = (*neighbor, i_nn_burst)
                            if directed:
                                # case e2 <-> e1
                                if burst[0] == neighbor_burst[0]:
                                    edges.append((e1, e2))
                                    edges.append((e2, e1))
                                # case:  e1 -> e2
                                elif burst[0] < neighbor_burst[0]:
                                    edges.append((e1, e2))

                                # case e2 -> e1
                                else:
                                    edges.append((e2, e1))

                            else:
                                edges.append((e1, e2))
                vertices.append((i, j, i_burst))
            grid_mark[i, j] = True

    connected_components = find_connected_components(vertices, edges, directed)

    return connected_components
