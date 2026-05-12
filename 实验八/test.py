def bellman_ford(n, edges, start):
    """
    Bellman-Ford 算法实现
    :param n: 顶点总数
    :param edges: 边列表，格式 [(起点u, 终点v, 权重w), ...]
    :param start: 起点编号
    :return: 最短距离数组 / 负权环提示
    """
    # 1. 初始化距离数组：起点为0，其余为无穷大
    INF = float('inf')
    dist = [INF] * n
    dist[start] = 0

    # 2. 核心：松弛 n-1 轮（n个顶点最多松弛n-1轮就能得到最短路径）
    for i in range(n - 1):
        updated = False  # 标记本轮是否有更新（优化用）
        for u, v, w in edges:
            # 松弛操作：如果 u 可达，且 u→v 的路径更短，就更新 v 的距离
            if dist[u] != INF and dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                updated = True
        # 优化：如果本轮没有更新，提前结束（后续轮也不会更新）
        if not updated:
            break

    # 3. 检测负权环：再松弛1轮，如果还能更新，说明存在负权环
    has_negative_cycle = False
    for u, v, w in edges:
        if dist[u] != INF and dist[v] > dist[u] + w:
            has_negative_cycle = True
            break

    return dist, has_negative_cycle


# ===================== 代入你的题目数据 =====================
if __name__ == '__main__':
    # 顶点数量：0~9 共10个
    vertex_num = 10
    # 你的图的所有边（直接复制题目里的边）
    graph_edges = [
        (0,5,5) , (0,8,8) , (0,6,6) ,(0,2,6) , (8,7,1) , (8,2,-4) , (2,7,4),
        (6,2,-1) , (8,6,-4) , (9,1,5) , (1,5,-5) , (6,5,2) , (1,3,-3) ,
        (3,4,3) , (5,3,-1) , (4,5,-2)
    ]
    # 起点：0
    start_vertex = 0

    # 运行算法
    shortest_dist, is_negative_cycle = bellman_ford(vertex_num, graph_edges, start_vertex)

    # 输出结果
    print("从顶点0出发到各顶点的最短距离：")
    for i in range(vertex_num):
        print(f"顶点 {i}: {shortest_dist[i] if shortest_dist[i] != float('inf') else '不可达'}")

    print("\n是否存在从起点可达的负权环：", "是" if is_negative_cycle else "否")