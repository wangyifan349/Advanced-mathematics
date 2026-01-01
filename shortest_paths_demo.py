import heapq
from collections import deque

def dijkstra_shortest_paths(adj_list, source_node):
    """
    Dijkstra算法：单源最短路，适用于无负权图
    :param adj_list: 邻接表，{节点: [(相邻节点, 权重), ...], ...}
    :param source_node: 源点节点编号
    :return: 源点到其它各点的最短距离字典
    """
    shortest_distances = {node: float('inf') for node in adj_list}
    shortest_distances[source_node] = 0
    visited_nodes = set()
    priority_queue = [(0, source_node)]

    while priority_queue:
        current_dist, current_node = heapq.heappop(priority_queue)
        if current_node in visited_nodes:
            continue
        visited_nodes.add(current_node)
        neighbors = adj_list[current_node]
        for neighbor, edge_weight in neighbors:
            new_distance = current_dist + edge_weight
            if new_distance < shortest_distances[neighbor]:
                shortest_distances[neighbor] = new_distance
                heapq.heappush(priority_queue, (new_distance, neighbor))
    return shortest_distances

def bellman_ford_shortest_paths(edge_list, num_nodes, source_node):
    """
    Bellman-Ford算法：单源最短路，可有负权边，不可有负权环
    :param edge_list: 边集合 [(起点, 终点, 权重), ...]
    :param num_nodes: 节点总数量
    :param source_node: 源点编号
    :return: 源点到其它节点的最短距离列表（索引代表节点编号）
    """
    distances = [float('inf')] * num_nodes
    distances[source_node] = 0

    for i in range(num_nodes - 1):
        for start_node, end_node, edge_weight in edge_list:
            if distances[start_node] != float('inf'):
                temp_distance = distances[start_node] + edge_weight
                if temp_distance < distances[end_node]:
                    distances[end_node] = temp_distance

    for start_node, end_node, edge_weight in edge_list:
        if distances[start_node] != float('inf') and distances[start_node] + edge_weight < distances[end_node]:
            raise ValueError("Graph contains negative weight cycle")
    return distances

def floyd_warshall_shortest_paths(adjacency_matrix):
    """
    Floyd-Warshall算法：多源最短路径所有对
    :param adjacency_matrix: 邻接矩阵，adjacency_matrix[i][j]表示i到j的权重, 无边为float('inf')
    :return: 各点对间最短距离矩阵
    """
    node_count = len(adjacency_matrix)
    distance_matrix = [row[:] for row in adjacency_matrix]  # 深复制
    for passing_node in range(node_count):
        for start_node in range(node_count):
            for end_node in range(node_count):
                candidate_dist = distance_matrix[start_node][passing_node] + distance_matrix[passing_node][end_node]
                if candidate_dist < distance_matrix[start_node][end_node]:
                    distance_matrix[start_node][end_node] = candidate_dist
    return distance_matrix

def bfs_min_steps(adj_list, source_node):
    """
    BFS无权图最短路径：最短步数（单源）
    :param adj_list: 邻接表（无权图），{节点: [邻接点, ...], ...}
    :param source_node: 起始节点编号
    :return: 源点到其它各点的最短步数字典（-1为不可达）
    """
    steps_to_node = {node: -1 for node in adj_list}
    steps_to_node[source_node] = 0
    node_queue = deque([source_node])
    while node_queue:
        current_node = node_queue.popleft()
        neighbors = adj_list[current_node]
        for neighbor in neighbors:
            if steps_to_node[neighbor] == -1:
                steps_to_node[neighbor] = steps_to_node[current_node] + 1
                node_queue.append(neighbor)
    return steps_to_node

# ============================== 算法演示区 ==============================

def main():
    print("-------- Dijkstra 演示 --------")
    # 0->1(2), 0->2(4), 1->2(1), 1->3(7), 2->3(3)
    adj_list = {
        0: [(1, 2), (2, 4)],
        1: [(2, 1), (3, 7)],
        2: [(3, 3)],
        3: []
    }
    distances = dijkstra_shortest_paths(adj_list, 0)
    print("Dijkstra最短距离:", distances)
    
    print("\n----- Bellman-Ford 演示 -----")
    edges = [ (0,1,2), (0,2,4), (1,2,1), (1,3,7), (2,3,3) ]
    distances = bellman_ford_shortest_paths(edges, 4, 0)
    print("Bellman-Ford最短距离:", distances)

    print("\n----- Floyd-Warshall 演示 -----")
    INF = float('inf')
    adjacency_matrix = [
        [0, 2, 4, INF],
        [INF, 0, 1, 7],
        [INF, INF, 0, 3],
        [INF, INF, INF, 0]
    ]
    all_pair_distances = floyd_warshall_shortest_paths(adjacency_matrix)
    for row in all_pair_distances:
        print("Floyd-Warshall最短距离:", row)

    print("\n-------- BFS 演示 --------")
    bfs_adj_list = {
        0: [1, 2],
        1: [2, 3],
        2: [3],
        3: []
    }
    min_steps = bfs_min_steps(bfs_adj_list, 0)
    print("BFS最短步数:", min_steps)

if __name__ == '__main__':
    main()
