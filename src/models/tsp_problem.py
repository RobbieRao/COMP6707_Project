import geatpy as ea
import numpy as np

class TSPProblem(ea.Problem):
    """
    使用 Geatpy 框架定义的 TSP 问题类。
    目标：在 100 位顾客中找到总距离最短的巡回路线（闭合）。
    """
    def __init__(self, coords):
        """
        Parameters:
            coords (np.ndarray): 顾客坐标数组，形状为 (n, 2)。
        """
        self.coords = coords
        n = coords.shape[0]
        name = "TSP"              # 问题名称
        M = 1                     # 目标数：单目标
        maxormins = [1]           # 1 表示最小化问题
        Dim = n                   # 决策变量维度（即顾客数）
        varTypes = [1] * Dim      # 1 表示离散变量（用于排列编码）
        
        # 决策变量取值范围：每个位置只能是 [0, n-1]
        lb = [0] * Dim
        ub = [n - 1] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        
        # 调用父类构造方法完成初始化
        super().__init__(name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        self.lb = lb
        self.ub = ub
        self.lbin = lbin
        self.ubin = ubin
    
    def aimFunc(self, pop):
        """
        目标函数：计算每个个体（路线）的总距离，并赋值给 pop.ObjV。
        
        Parameters:
            pop (Population): Geatpy 种群对象，包含所有个体的表型信息 pop.Phen。
        """
        routes = pop.Phen.astype(int)  # 形状 (NIND, Dim)
        NIND, Dim = routes.shape
        total_distance = np.zeros((NIND, 1))
        
        for i in range(NIND):
            route = routes[i]
            dist = 0.0
            # 累计相邻顾客之间的距离
            for j in range(Dim - 1):
                dist += np.linalg.norm(self.coords[route[j]] - self.coords[route[j + 1]])
            # 闭合路径：最后一个顾客回到第一个顾客
            dist += np.linalg.norm(self.coords[route[-1]] - self.coords[route[0]])
            
            total_distance[i, 0] = dist
        
        pop.ObjV = total_distance
