import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
from sklearn.cluster import KMeans
import geatpy as ea
from src.models.tsp_problem import TSPProblem
from src.utils.metrics import route_distance

class LargeScaleTSPProblem:
    """
    大规模 TSP 问题建模与求解：
    1. 将原 TSP 数据扩展成 200 个客户（通过对原客户 X 坐标加 100 生成额外 100 个客户）。
    2. 对 200 个客户进行 KMeans 聚类，划分成若干区域。
    3. 分别求解每个区域的局部 TSP（采用 GA）。
    4. 按聚类中心顺序将各区域局部最优路径拼接为全局路径（区域间直接连接）。
    """
    def __init__(self, coords, n_clusters=4):
        """
        Parameters:
            coords (np.ndarray): 原 TSP 数据中的客户坐标，形状为 (100, 2)。
            n_clusters (int): 聚类的数量，默认 4。
        """
        self.original_coords = coords  # 原始 100 个客户
        self.n_clusters = n_clusters
        self.full_coords = self._generate_full_coords(coords)  # 扩展为 200 个客户
        self.cluster_labels = None
        self.cluster_centroids = None
        self.local_routes = {}  # 存储每个聚类的局部最优路径（全局索引）
    
    def _generate_full_coords(self, coords):
        """
        生成扩展数据：在原数据基础上，通过对 X 坐标加 100 得到另外 100 个客户，
        与原客户合并为 200 个客户。
        
        Returns:
            np.ndarray: 形状 (200,2)。
        """
        new_coords = coords.copy()
        new_coords[:, 0] += 100  # 新客户 X 坐标加 100
        full = np.vstack([coords, new_coords])
        return full
    
    def cluster_data(self):
        """
        对 200 个客户进行 KMeans 聚类，存储聚类标签和聚类中心。
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.cluster_labels = kmeans.fit_predict(self.full_coords)
        self.cluster_centroids = kmeans.cluster_centers_
    
    def _solve_local_tsp(self, local_coords):
        """
        对局部区域（某个聚类内）的客户求解 TSP 问题。
        
        Parameters:
            local_coords (np.ndarray): 局部客户坐标，形状 (n_local, 2)。
        
        Returns:
            tuple: (local_best_route, local_best_distance)
                local_best_route: 局部最优排列，索引范围 0~(n_local-1)
                local_best_distance: 对应的巡回总距离
        """
        # 创建 TSP 问题实例
        problem_local = TSPProblem(local_coords)
        # 构造种群，排列编码 "P" 对于局部问题无需 FieldD（与 task1_tsp 类似）
        population = ea.Population(Encoding='P', NIND=50)  # 局部问题可用较小种群
        # 使用简单遗传算法模板
        algorithm = ea.soea_SEGA_templet(problem_local, population)
        algorithm.MAXGEN = 200
        algorithm.logTras = 0
        algorithm.verbose = False
        algorithm.drawing = 0
        res = ea.optimize(algorithm, verbose=False, drawing=0, outputMsg=False, drawLog=False, saveFlag=False)
        local_route = res['Vars'][0].astype(int)
        local_distance = res['ObjV'][0][0]
        return local_route, local_distance
    
    def solve(self, pop_size_local=50, max_gen_local=200):
        """
        求解大规模 TSP 问题：
          1. 聚类划分区域；
          2. 对每个区域求解局部 TSP；
          3. 按聚类中心（例如 X 坐标）排序各区域，并将局部路径拼接为全局路径。
        
        Parameters:
            pop_size_local (int): 局部 TSP 求解时种群规模（默认 50）。
            max_gen_local (int): 局部 TSP 求解时最大进化代数（默认 200）。
        
        Returns:
            tuple: (global_route, global_distance)
                global_route: 全局最优巡回路线（全局索引，长度为 200）
                global_distance: 全局总巡回距离
        """
        # 1. 聚类
        self.cluster_data()
        
        global_route = []
        cluster_info = []
        
        # 2. 对每个聚类分别求解局部 TSP
        for cluster_id in range(self.n_clusters):
            # 获取属于当前聚类的全局索引
            indices = np.where(self.cluster_labels == cluster_id)[0]
            local_coords = self.full_coords[indices]
            if len(indices) == 0:
                continue
            local_route, local_distance = self._solve_local_tsp(local_coords)
            # 将局部解映射到全局索引：local_route 中的数字代表局部排序，映射：global_index = indices[local_route]
            global_local_route = [indices[i] for i in local_route]
            # 保存局部结果
            cluster_info.append({
                "cluster_id": cluster_id,
                "indices": indices,
                "centroid": self.cluster_centroids[cluster_id],
                "local_route": global_local_route,
                "local_distance": local_distance
            })
        
        # 3. 对聚类结果排序（例如按聚类中心的 X 坐标排序）
        cluster_info.sort(key=lambda x: x["centroid"][0])
        
        # 4. 拼接全局路线：简单拼接各区域的局部路线
        for info in cluster_info:
            global_route.extend(info["local_route"])
        
        # 5. 计算全局总距离（直接使用欧氏距离累加法，拼接时考虑各区域间连接）
        from src.utils.metrics import route_distance
        global_distance = route_distance(self.full_coords, global_route)
        return global_route, global_distance
