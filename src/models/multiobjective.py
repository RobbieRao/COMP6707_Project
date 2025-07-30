# multiobjective.py
import numpy as np
import geatpy as ea

class MultiObjectiveTSPProblem(ea.Problem):
    """
    多目标 TSP 问题：
    - f1: 总巡回距离（最小化）
    - f2: 总销售利润，定义为每个客户的 intrinsic PROFIT 减去从起点到该客户的累计旅行距离，
          由于我们希望最大化利润，因此转换为最小化 -总利润。
          
    可选模式：
      * 如果 lambda_weight 为 None，则使用多目标形式，目标为 <min f1, min (-total_profit)>；
      * 如果 lambda_weight 给定，则采用加权方式，将问题转化为单目标：min (f1 - lambda_weight * total_profit)
    """
    def __init__(self, coords, profit, lambda_weight=None):
        """
        Parameters:
            coords (np.ndarray): 客户坐标数组，形状为 (n, 2)。
            profit (np.ndarray): 客户内在 PROFIT 数组，形状为 (n, )。
            lambda_weight (float or None): 加权参数。如果为 None，则采用多目标形式；否则采用加权单目标形式。
        """
        self.coords = coords
        self.profit = profit
        self.n = coords.shape[0]
        self.lambda_weight = lambda_weight
        
        if self.lambda_weight is None:
            M = 2
            maxormins = [1, 1]  # 都为最小化：f1 (distance) 和 -profit
        else:
            M = 1
            maxormins = [1]  # 单目标最小化
        
        name = "MultiObjTSP"
        Dim = self.n
        varTypes = [1] * Dim  # 使用排列编码（离散变量）
        lb = [0] * Dim
        ub = [self.n - 1] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        
        # 调用父类构造函数
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        
    def aimFunc(self, pop):
        """
        目标函数：对种群中每个个体（排列）计算目标值。
        
        对于每个个体，计算：
          f1 = 总巡回距离（包括返回起点）
          f2 = - (总销售利润)
          其中，总销售利润 = sum_{j=0}^{n-1} (profit[r[j]] - cumulative_distance_j)
          其中 cumulative_distance_j 为从起点到 r[j] 的累计距离（r[0]处累计距离为0）。
          
        如果采用加权模式，则返回单目标 f = f1 - lambda_weight * (总销售利润).
        """
        routes = pop.Phen.astype(int)  # shape (NIND, n)
        NIND, n = routes.shape
        ObjV = []
        
        for i in range(NIND):
            route = routes[i]
            # 计算 f1：总巡回距离
            distance = 0.0
            cumulative = 0.0  # 累计距离，从起点开始
            total_profit = 0.0
            # 遍历客户序列
            for j in range(n - 1):
                d = np.linalg.norm(self.coords[route[j]] - self.coords[route[j+1]])
                distance += d
                cumulative += d
                # 对当前客户计算利润：intrinsic profit - 累计距离
                total_profit += self.profit[route[j]] - cumulative
            # 加上最后一个客户返回起点的距离
            d_last = np.linalg.norm(self.coords[route[-1]] - self.coords[route[0]])
            distance += d_last
            cumulative += d_last
            total_profit += self.profit[route[-1]] - cumulative
            
            if self.lambda_weight is None:
                # 多目标形式：f1 和 f2 = - total_profit
                ObjV.append([distance, -total_profit])
            else:
                # 加权单目标：f = f1 - lambda_weight * total_profit
                ObjV.append([distance - self.lambda_weight * total_profit])
        
        pop.ObjV = np.array(ObjV)
