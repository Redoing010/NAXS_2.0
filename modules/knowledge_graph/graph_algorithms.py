# 图算法模块
# 实现各种图分析算法，包括中心性分析、社区发现、路径查找等

import logging
import math
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
from collections import defaultdict, deque
import heapq
import random

logger = logging.getLogger(__name__)

class CentralityType(Enum):
    """中心性类型"""
    DEGREE = "degree"              # 度中心性
    BETWEENNESS = "betweenness"    # 介数中心性
    CLOSENESS = "closeness"        # 接近中心性
    EIGENVECTOR = "eigenvector"    # 特征向量中心性
    PAGERANK = "pagerank"          # PageRank
    KATZ = "katz"                  # Katz中心性

class CommunityAlgorithm(Enum):
    """社区发现算法"""
    LOUVAIN = "louvain"            # Louvain算法
    LABEL_PROPAGATION = "label_propagation"  # 标签传播
    MODULARITY = "modularity"      # 模块度优化
    GIRVAN_NEWMAN = "girvan_newman"  # Girvan-Newman算法

@dataclass
class CentralityMetrics:
    """中心性指标"""
    node_id: str
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    pagerank: float = 0.0
    katz_centrality: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'node_id': self.node_id,
            'degree_centrality': self.degree_centrality,
            'betweenness_centrality': self.betweenness_centrality,
            'closeness_centrality': self.closeness_centrality,
            'eigenvector_centrality': self.eigenvector_centrality,
            'pagerank': self.pagerank,
            'katz_centrality': self.katz_centrality
        }

@dataclass
class Community:
    """社区"""
    id: str
    nodes: Set[str] = field(default_factory=set)
    size: int = 0
    density: float = 0.0
    modularity: float = 0.0
    
    def __post_init__(self):
        self.size = len(self.nodes)
    
    def add_node(self, node_id: str):
        """添加节点"""
        self.nodes.add(node_id)
        self.size = len(self.nodes)
    
    def remove_node(self, node_id: str):
        """移除节点"""
        self.nodes.discard(node_id)
        self.size = len(self.nodes)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'nodes': list(self.nodes),
            'size': self.size,
            'density': self.density,
            'modularity': self.modularity
        }

@dataclass
class CommunityDetection:
    """社区发现结果"""
    communities: List[Community] = field(default_factory=list)
    modularity: float = 0.0
    algorithm: str = ""
    processing_time: float = 0.0
    
    def get_community_for_node(self, node_id: str) -> Optional[Community]:
        """获取节点所属社区
        
        Args:
            node_id: 节点ID
            
        Returns:
            社区对象
        """
        for community in self.communities:
            if node_id in community.nodes:
                return community
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'communities': [community.to_dict() for community in self.communities],
            'community_count': len(self.communities),
            'modularity': self.modularity,
            'algorithm': self.algorithm,
            'processing_time': self.processing_time
        }

@dataclass
class PathFinding:
    """路径查找结果"""
    source: str
    target: str
    path: List[str] = field(default_factory=list)
    distance: float = float('inf')
    algorithm: str = ""
    
    @property
    def exists(self) -> bool:
        """路径是否存在"""
        return len(self.path) > 0 and self.distance != float('inf')
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'source': self.source,
            'target': self.target,
            'path': self.path,
            'distance': self.distance,
            'path_length': len(self.path),
            'exists': self.exists,
            'algorithm': self.algorithm
        }

@dataclass
class GraphMetrics:
    """图指标"""
    node_count: int = 0
    edge_count: int = 0
    density: float = 0.0
    average_degree: float = 0.0
    clustering_coefficient: float = 0.0
    diameter: int = 0
    average_path_length: float = 0.0
    connected_components: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'node_count': self.node_count,
            'edge_count': self.edge_count,
            'density': self.density,
            'average_degree': self.average_degree,
            'clustering_coefficient': self.clustering_coefficient,
            'diameter': self.diameter,
            'average_path_length': self.average_path_length,
            'connected_components': self.connected_components
        }

class GraphAlgorithms:
    """图算法集合
    
    实现各种图分析算法
    """
    
    def __init__(self):
        # 图数据结构
        self.adjacency_list: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.nodes: Set[str] = set()
        self.edges: Set[Tuple[str, str]] = set()
        
        # 缓存
        self.centrality_cache: Dict[str, Dict[str, float]] = {}
        self.path_cache: Dict[Tuple[str, str], PathFinding] = {}
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info("图算法模块初始化完成")
    
    def load_graph(self, nodes: List[str], edges: List[Tuple[str, str, float]] = None):
        """加载图数据
        
        Args:
            nodes: 节点列表
            edges: 边列表 (source, target, weight)
        """
        with self.lock:
            self.nodes = set(nodes)
            self.adjacency_list.clear()
            self.edges.clear()
            
            if edges:
                for source, target, weight in edges:
                    self.add_edge(source, target, weight)
            
            # 清空缓存
            self.centrality_cache.clear()
            self.path_cache.clear()
            
            logger.debug(f"图数据加载完成: {len(self.nodes)} 个节点, {len(self.edges)} 条边")
    
    def add_edge(self, source: str, target: str, weight: float = 1.0):
        """添加边
        
        Args:
            source: 源节点
            target: 目标节点
            weight: 权重
        """
        with self.lock:
            self.nodes.add(source)
            self.nodes.add(target)
            self.adjacency_list[source][target] = weight
            self.adjacency_list[target][source] = weight  # 无向图
            self.edges.add((source, target))
    
    def calculate_degree_centrality(self) -> Dict[str, float]:
        """计算度中心性
        
        Returns:
            节点度中心性字典
        """
        if 'degree' in self.centrality_cache:
            return self.centrality_cache['degree']
        
        centrality = {}
        n = len(self.nodes)
        
        if n <= 1:
            return {node: 0.0 for node in self.nodes}
        
        for node in self.nodes:
            degree = len(self.adjacency_list[node])
            centrality[node] = degree / (n - 1)  # 标准化
        
        self.centrality_cache['degree'] = centrality
        return centrality
    
    def calculate_betweenness_centrality(self) -> Dict[str, float]:
        """计算介数中心性
        
        Returns:
            节点介数中心性字典
        """
        if 'betweenness' in self.centrality_cache:
            return self.centrality_cache['betweenness']
        
        centrality = {node: 0.0 for node in self.nodes}
        
        for source in self.nodes:
            # 使用Brandes算法
            stack = []
            paths = {node: [] for node in self.nodes}
            sigma = {node: 0.0 for node in self.nodes}
            sigma[source] = 1.0
            distance = {node: -1 for node in self.nodes}
            distance[source] = 0
            queue = deque([source])
            
            # BFS
            while queue:
                current = queue.popleft()
                stack.append(current)
                
                for neighbor in self.adjacency_list[current]:
                    # 第一次访问邻居
                    if distance[neighbor] < 0:
                        queue.append(neighbor)
                        distance[neighbor] = distance[current] + 1
                    
                    # 最短路径
                    if distance[neighbor] == distance[current] + 1:
                        sigma[neighbor] += sigma[current]
                        paths[neighbor].append(current)
            
            # 累积
            delta = {node: 0.0 for node in self.nodes}
            while stack:
                current = stack.pop()
                for predecessor in paths[current]:
                    delta[predecessor] += (sigma[predecessor] / sigma[current]) * (1 + delta[current])
                if current != source:
                    centrality[current] += delta[current]
        
        # 标准化
        n = len(self.nodes)
        if n > 2:
            norm = 2.0 / ((n - 1) * (n - 2))
            for node in centrality:
                centrality[node] *= norm
        
        self.centrality_cache['betweenness'] = centrality
        return centrality
    
    def calculate_closeness_centrality(self) -> Dict[str, float]:
        """计算接近中心性
        
        Returns:
            节点接近中心性字典
        """
        if 'closeness' in self.centrality_cache:
            return self.centrality_cache['closeness']
        
        centrality = {}
        
        for node in self.nodes:
            distances = self._dijkstra(node)
            total_distance = sum(d for d in distances.values() if d != float('inf'))
            reachable_nodes = sum(1 for d in distances.values() if d != float('inf')) - 1  # 排除自己
            
            if total_distance > 0 and reachable_nodes > 0:
                centrality[node] = reachable_nodes / total_distance
            else:
                centrality[node] = 0.0
        
        self.centrality_cache['closeness'] = centrality
        return centrality
    
    def calculate_pagerank(self, damping: float = 0.85, max_iter: int = 100, tolerance: float = 1e-6) -> Dict[str, float]:
        """计算PageRank
        
        Args:
            damping: 阻尼系数
            max_iter: 最大迭代次数
            tolerance: 收敛容差
            
        Returns:
            节点PageRank值字典
        """
        cache_key = f'pagerank_{damping}_{max_iter}_{tolerance}'
        if cache_key in self.centrality_cache:
            return self.centrality_cache[cache_key]
        
        n = len(self.nodes)
        if n == 0:
            return {}
        
        # 初始化
        pagerank = {node: 1.0 / n for node in self.nodes}
        
        for iteration in range(max_iter):
            new_pagerank = {}
            
            for node in self.nodes:
                rank = (1 - damping) / n
                
                # 计算来自其他节点的贡献
                for neighbor in self.nodes:
                    if node in self.adjacency_list[neighbor]:
                        out_degree = len(self.adjacency_list[neighbor])
                        if out_degree > 0:
                            rank += damping * pagerank[neighbor] / out_degree
                
                new_pagerank[node] = rank
            
            # 检查收敛
            diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in self.nodes)
            if diff < tolerance:
                logger.debug(f"PageRank收敛于第{iteration + 1}次迭代")
                break
            
            pagerank = new_pagerank
        
        self.centrality_cache[cache_key] = pagerank
        return pagerank
    
    def calculate_eigenvector_centrality(self, max_iter: int = 100, tolerance: float = 1e-6) -> Dict[str, float]:
        """计算特征向量中心性
        
        Args:
            max_iter: 最大迭代次数
            tolerance: 收敛容差
            
        Returns:
            节点特征向量中心性字典
        """
        cache_key = f'eigenvector_{max_iter}_{tolerance}'
        if cache_key in self.centrality_cache:
            return self.centrality_cache[cache_key]
        
        n = len(self.nodes)
        if n == 0:
            return {}
        
        # 初始化
        centrality = {node: 1.0 for node in self.nodes}
        
        for iteration in range(max_iter):
            new_centrality = {node: 0.0 for node in self.nodes}
            
            # 计算新的中心性值
            for node in self.nodes:
                for neighbor in self.adjacency_list[node]:
                    new_centrality[node] += centrality[neighbor]
            
            # 标准化
            norm = math.sqrt(sum(v * v for v in new_centrality.values()))
            if norm > 0:
                for node in new_centrality:
                    new_centrality[node] /= norm
            
            # 检查收敛
            diff = sum(abs(new_centrality[node] - centrality[node]) for node in self.nodes)
            if diff < tolerance:
                logger.debug(f"特征向量中心性收敛于第{iteration + 1}次迭代")
                break
            
            centrality = new_centrality
        
        self.centrality_cache[cache_key] = centrality
        return centrality
    
    def detect_communities_louvain(self) -> CommunityDetection:
        """使用Louvain算法进行社区发现
        
        Returns:
            社区发现结果
        """
        start_time = datetime.now()
        
        # 初始化：每个节点为一个社区
        node_to_community = {node: i for i, node in enumerate(self.nodes)}
        communities = {i: {node} for i, node in enumerate(self.nodes)}
        
        improved = True
        iteration = 0
        
        while improved:
            improved = False
            iteration += 1
            
            for node in self.nodes:
                current_community = node_to_community[node]
                best_community = current_community
                best_gain = 0.0
                
                # 尝试将节点移动到邻居的社区
                neighbor_communities = set()
                for neighbor in self.adjacency_list[node]:
                    neighbor_communities.add(node_to_community[neighbor])
                
                for community_id in neighbor_communities:
                    if community_id == current_community:
                        continue
                    
                    # 计算模块度增益
                    gain = self._calculate_modularity_gain(
                        node, current_community, community_id,
                        node_to_community, communities
                    )
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_community = community_id
                
                # 移动节点
                if best_community != current_community:
                    communities[current_community].remove(node)
                    communities[best_community].add(node)
                    node_to_community[node] = best_community
                    improved = True
            
            logger.debug(f"Louvain算法第{iteration}次迭代完成")
        
        # 构建结果
        result_communities = []
        for i, (community_id, nodes) in enumerate(communities.items()):
            if nodes:  # 非空社区
                community = Community(
                    id=f"community_{i}",
                    nodes=nodes
                )
                result_communities.append(community)
        
        # 计算总模块度
        modularity = self._calculate_modularity(node_to_community)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = CommunityDetection(
            communities=result_communities,
            modularity=modularity,
            algorithm="louvain",
            processing_time=processing_time
        )
        
        logger.info(f"Louvain社区发现完成: {len(result_communities)} 个社区, 模块度: {modularity:.3f}")
        
        return result
    
    def detect_communities_label_propagation(self, max_iter: int = 100) -> CommunityDetection:
        """使用标签传播算法进行社区发现
        
        Args:
            max_iter: 最大迭代次数
            
        Returns:
            社区发现结果
        """
        start_time = datetime.now()
        
        # 初始化：每个节点的标签为自己
        labels = {node: node for node in self.nodes}
        
        for iteration in range(max_iter):
            changed = False
            nodes_list = list(self.nodes)
            random.shuffle(nodes_list)  # 随机顺序
            
            for node in nodes_list:
                # 统计邻居标签
                neighbor_labels = defaultdict(int)
                for neighbor in self.adjacency_list[node]:
                    neighbor_labels[labels[neighbor]] += 1
                
                if neighbor_labels:
                    # 选择最频繁的标签
                    most_frequent_label = max(neighbor_labels.items(), key=lambda x: x[1])[0]
                    
                    if labels[node] != most_frequent_label:
                        labels[node] = most_frequent_label
                        changed = True
            
            if not changed:
                logger.debug(f"标签传播算法收敛于第{iteration + 1}次迭代")
                break
        
        # 构建社区
        label_to_nodes = defaultdict(set)
        for node, label in labels.items():
            label_to_nodes[label].add(node)
        
        result_communities = []
        for i, (label, nodes) in enumerate(label_to_nodes.items()):
            community = Community(
                id=f"community_{i}",
                nodes=nodes
            )
            result_communities.append(community)
        
        # 计算模块度
        node_to_community = {}
        for i, community in enumerate(result_communities):
            for node in community.nodes:
                node_to_community[node] = i
        
        modularity = self._calculate_modularity(node_to_community)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = CommunityDetection(
            communities=result_communities,
            modularity=modularity,
            algorithm="label_propagation",
            processing_time=processing_time
        )
        
        logger.info(f"标签传播社区发现完成: {len(result_communities)} 个社区, 模块度: {modularity:.3f}")
        
        return result
    
    def find_shortest_path(self, source: str, target: str) -> PathFinding:
        """查找最短路径
        
        Args:
            source: 源节点
            target: 目标节点
            
        Returns:
            路径查找结果
        """
        cache_key = (source, target)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        if source not in self.nodes or target not in self.nodes:
            result = PathFinding(source=source, target=target, algorithm="dijkstra")
            self.path_cache[cache_key] = result
            return result
        
        # 使用Dijkstra算法
        distances = {node: float('inf') for node in self.nodes}
        distances[source] = 0
        previous = {}
        visited = set()
        
        heap = [(0, source)]
        
        while heap:
            current_distance, current = heapq.heappop(heap)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == target:
                break
            
            for neighbor, weight in self.adjacency_list[current].items():
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(heap, (distance, neighbor))
        
        # 重构路径
        path = []
        if target in previous or source == target:
            current = target
            while current is not None:
                path.append(current)
                current = previous.get(current)
            path.reverse()
        
        result = PathFinding(
            source=source,
            target=target,
            path=path,
            distance=distances[target],
            algorithm="dijkstra"
        )
        
        self.path_cache[cache_key] = result
        return result
    
    def calculate_graph_metrics(self) -> GraphMetrics:
        """计算图的整体指标
        
        Returns:
            图指标
        """
        n = len(self.nodes)
        m = len(self.edges)
        
        # 密度
        max_edges = n * (n - 1) // 2
        density = m / max_edges if max_edges > 0 else 0.0
        
        # 平均度
        total_degree = sum(len(neighbors) for neighbors in self.adjacency_list.values())
        average_degree = total_degree / n if n > 0 else 0.0
        
        # 聚类系数
        clustering_coefficient = self._calculate_clustering_coefficient()
        
        # 连通分量
        connected_components = self._count_connected_components()
        
        # 直径和平均路径长度
        diameter, average_path_length = self._calculate_path_metrics()
        
        return GraphMetrics(
            node_count=n,
            edge_count=m,
            density=density,
            average_degree=average_degree,
            clustering_coefficient=clustering_coefficient,
            diameter=diameter,
            average_path_length=average_path_length,
            connected_components=connected_components
        )
    
    def _dijkstra(self, source: str) -> Dict[str, float]:
        """Dijkstra最短路径算法
        
        Args:
            source: 源节点
            
        Returns:
            到各节点的最短距离
        """
        distances = {node: float('inf') for node in self.nodes}
        distances[source] = 0
        visited = set()
        
        heap = [(0, source)]
        
        while heap:
            current_distance, current = heapq.heappop(heap)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor, weight in self.adjacency_list[current].items():
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(heap, (distance, neighbor))
        
        return distances
    
    def _calculate_modularity_gain(self, node: str, from_community: int, to_community: int,
                                 node_to_community: Dict[str, int], communities: Dict[int, Set[str]]) -> float:
        """计算模块度增益
        
        Args:
            node: 节点
            from_community: 源社区
            to_community: 目标社区
            node_to_community: 节点到社区的映射
            communities: 社区字典
            
        Returns:
            模块度增益
        """
        # 简化的模块度增益计算
        ki_in_from = 0  # 节点在源社区内的边数
        ki_in_to = 0    # 节点在目标社区内的边数
        
        for neighbor in self.adjacency_list[node]:
            if node_to_community[neighbor] == from_community:
                ki_in_from += 1
            elif node_to_community[neighbor] == to_community:
                ki_in_to += 1
        
        # 简化计算
        return ki_in_to - ki_in_from
    
    def _calculate_modularity(self, node_to_community: Dict[str, int]) -> float:
        """计算模块度
        
        Args:
            node_to_community: 节点到社区的映射
            
        Returns:
            模块度值
        """
        m = len(self.edges)
        if m == 0:
            return 0.0
        
        modularity = 0.0
        
        for node1 in self.nodes:
            for node2 in self.nodes:
                if node1 == node2:
                    continue
                
                # 实际边权重
                actual_weight = self.adjacency_list[node1].get(node2, 0)
                
                # 期望边权重
                degree1 = len(self.adjacency_list[node1])
                degree2 = len(self.adjacency_list[node2])
                expected_weight = (degree1 * degree2) / (2 * m)
                
                # 同一社区的贡献
                if node_to_community[node1] == node_to_community[node2]:
                    modularity += actual_weight - expected_weight
        
        return modularity / (2 * m)
    
    def _calculate_clustering_coefficient(self) -> float:
        """计算聚类系数
        
        Returns:
            平均聚类系数
        """
        if len(self.nodes) == 0:
            return 0.0
        
        total_clustering = 0.0
        
        for node in self.nodes:
            neighbors = list(self.adjacency_list[node].keys())
            k = len(neighbors)
            
            if k < 2:
                continue
            
            # 计算邻居间的连接数
            connections = 0
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in self.adjacency_list[neighbors[i]]:
                        connections += 1
            
            # 局部聚类系数
            possible_connections = k * (k - 1) // 2
            local_clustering = connections / possible_connections if possible_connections > 0 else 0.0
            total_clustering += local_clustering
        
        return total_clustering / len(self.nodes)
    
    def _count_connected_components(self) -> int:
        """计算连通分量数
        
        Returns:
            连通分量数
        """
        visited = set()
        components = 0
        
        for node in self.nodes:
            if node not in visited:
                # DFS遍历连通分量
                stack = [node]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        stack.extend(self.adjacency_list[current].keys())
                components += 1
        
        return components
    
    def _calculate_path_metrics(self) -> Tuple[int, float]:
        """计算路径相关指标
        
        Returns:
            (直径, 平均路径长度)
        """
        if len(self.nodes) <= 1:
            return 0, 0.0
        
        all_distances = []
        max_distance = 0
        
        # 计算所有节点对之间的最短距离
        for source in self.nodes:
            distances = self._dijkstra(source)
            for target, distance in distances.items():
                if source != target and distance != float('inf'):
                    all_distances.append(distance)
                    max_distance = max(max_distance, distance)
        
        diameter = int(max_distance)
        average_path_length = sum(all_distances) / len(all_distances) if all_distances else 0.0
        
        return diameter, average_path_length

# 便捷函数
def create_graph_algorithms() -> GraphAlgorithms:
    """创建图算法实例
    
    Returns:
        图算法实例
    """
    return GraphAlgorithms()

def calculate_centrality(graph_algorithms: GraphAlgorithms, centrality_type: CentralityType) -> Dict[str, float]:
    """计算中心性
    
    Args:
        graph_algorithms: 图算法实例
        centrality_type: 中心性类型
        
    Returns:
        中心性值字典
    """
    if centrality_type == CentralityType.DEGREE:
        return graph_algorithms.calculate_degree_centrality()
    elif centrality_type == CentralityType.BETWEENNESS:
        return graph_algorithms.calculate_betweenness_centrality()
    elif centrality_type == CentralityType.CLOSENESS:
        return graph_algorithms.calculate_closeness_centrality()
    elif centrality_type == CentralityType.PAGERANK:
        return graph_algorithms.calculate_pagerank()
    elif centrality_type == CentralityType.EIGENVECTOR:
        return graph_algorithms.calculate_eigenvector_centrality()
    else:
        raise ValueError(f"不支持的中心性类型: {centrality_type}")

def detect_communities(graph_algorithms: GraphAlgorithms, algorithm: CommunityAlgorithm) -> CommunityDetection:
    """社区发现
    
    Args:
        graph_algorithms: 图算法实例
        algorithm: 社区发现算法
        
    Returns:
        社区发现结果
    """
    if algorithm == CommunityAlgorithm.LOUVAIN:
        return graph_algorithms.detect_communities_louvain()
    elif algorithm == CommunityAlgorithm.LABEL_PROPAGATION:
        return graph_algorithms.detect_communities_label_propagation()
    else:
        raise ValueError(f"不支持的社区发现算法: {algorithm}")

def find_shortest_path(graph_algorithms: GraphAlgorithms, source: str, target: str) -> PathFinding:
    """查找最短路径
    
    Args:
        graph_algorithms: 图算法实例
        source: 源节点
        target: 目标节点
        
    Returns:
        路径查找结果
    """
    return graph_algorithms.find_shortest_path(source, target)