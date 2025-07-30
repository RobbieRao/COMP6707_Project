# src/models/bilevel.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import random
import copy
import geatpy as ea
from src.utils.metrics import route_distance

def compute_bilevel_cost(route, coords_customers, depot1_coord, depot2_coord):
    """
    计算给定路线在双层情境下的总距离：
      1) 起点：depot1_coord
      2) 前 50 个客户
      3) 中间停靠：depot2_coord
      4) 后 50 个客户
      5) 回到 depot1_coord
    
    Parameters:
        route (list[int]): 100 个客户的访问顺序（0~99）
        coords_customers (np.ndarray): 形状 (100, 2) 的客户坐标
        depot1_coord (np.ndarray): 形状 (2,) 的第一个仓库坐标
        depot2_coord (np.ndarray): 形状 (2,) 的第二个仓库坐标
    
    Returns:
        float: 总巡回距离
    """
    # 分成两段
    first_segment = route[:50]
    second_segment = route[50:]
    
    dist = 0.0
    # depot1 -> first_segment[0]
    dist += np.linalg.norm(depot1_coord - coords_customers[first_segment[0]])
    # first_segment 中相邻客户距离
    for i in range(len(first_segment) - 1):
        dist += np.linalg.norm(coords_customers[first_segment[i]] - coords_customers[first_segment[i+1]])
    # first_segment -> depot2
    dist += np.linalg.norm(coords_customers[first_segment[-1]] - depot2_coord)
    
    # depot2 -> second_segment[0]
    dist += np.linalg.norm(depot2_coord - coords_customers[second_segment[0]])
    # second_segment 中相邻客户距离
    for i in range(len(second_segment) - 1):
        dist += np.linalg.norm(coords_customers[second_segment[i]] - coords_customers[second_segment[i+1]])
    # second_segment -> depot1
    dist += np.linalg.norm(coords_customers[second_segment[-1]] - depot1_coord)
    
    return dist

# -----------------------------------------------------------------------------
# (A) 嵌套进化算法（Nested Evolutionary Algorithm）
# -----------------------------------------------------------------------------
class NestedBiLevelTSP:
    """
    上层：在 10 个候选仓库中选择 (Depot1, Depot2)；下层：给定 (Depot1, Depot2) 后，求解强制在第 50 个客户后
    访问第二仓库并最终返回第一个仓库的 TSP 路径。
    
    思路：
      - 上层染色体仅编码 (d1, d2)，即仓库对；上层采用遗传算法或枚举方式搜索。
      - 对于每个 (d1, d2)，通过下层遗传算法找到最优/近似最优路线，并将此路线的距离作为上层适应度。
      - 经过多代迭代后，上层 GA 找到最优仓库对及对应的下层最优路径。
    """
    def __init__(self, coords_customers, coords_depots, pop_size=10, max_gen=30, 
                 lower_pop_size=100, lower_max_gen=300):
        """
        Parameters:
            coords_customers (np.ndarray): 100 个客户的坐标，形状 (100, 2)。
            coords_depots (np.ndarray): 10 个仓库的坐标，形状 (10, 2)。
            pop_size (int): 上层种群规模（即选择 Depot 对的种群大小）。
            max_gen (int): 上层最大进化代数。
            lower_pop_size (int): 下层种群规模（求解 100 客户的 TSP 路径时）。
            lower_max_gen (int): 下层最大进化代数。
        """
        self.coords_customers = coords_customers
        self.coords_depots = coords_depots
        self.n_depots = coords_depots.shape[0]
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.lower_pop_size = lower_pop_size
        self.lower_max_gen = lower_max_gen
    
    def solve_lower_level(self, depot1_idx, depot2_idx):
        """
        下层：给定仓库对 (d1, d2)，使用一个简单的 GA 来求解带强制中点停靠的 TSP 问题。
        
        Returns:
            best_distance: 下层最小巡回距离
            best_route: 对应的路线 (0~99 的排列)
        """
        # 1) 构造 geatpy 问题类
        problem = LowerLevelTSPProblem(self.coords_customers,
                                       self.coords_depots[depot1_idx],
                                       self.coords_depots[depot2_idx])
        
        # 2) 构造种群
        population = ea.Population(Encoding='P', NIND=self.lower_pop_size)
        
        # 3) 选择算法模板（例如标准遗传算法）
        algorithm = ea.soea_EGA_templet(problem, population)
        algorithm.MAXGEN = self.lower_max_gen
        algorithm.logTras = 0  # 不打印中间日志
        algorithm.verbose = False
        algorithm.drawing = 0
        
        # 4) 运行优化
        res = ea.optimize(algorithm, verbose=False, drawing=0, outputMsg=False, saveFlag=False)
        
        # 提取结果
        best_route = res['Vars'][0].astype(int)
        best_distance = res['ObjV'][0, 0]
        return best_distance, best_route
    
    def run(self):
        """
        执行双层优化过程：
          - 上层编码：2 个整数，分别是仓库对 (d1, d2)，0 <= d1, d2 < 10, 且 d1 != d2。
          - 下层：对每个上层个体，调用 solve_lower_level() 求解并返回最小距离作为上层适应度。
        
        Returns:
            global_best: 字典，包含最优仓库对及其对应的下层最优路线和距离。
        """
        # -- 上层初始化 --
        # 染色体结构为 (depot1_idx, depot2_idx)，两个值均在 [0, 9]，且不相等
        upper_pop = []
        while len(upper_pop) < self.pop_size:
            d1 = random.randint(0, self.n_depots - 1)
            d2 = random.randint(0, self.n_depots - 1)
            if d2 != d1:
                upper_pop.append({'depots': (d1, d2), 'fitness': None, 'route': None})
        
        global_best = None
        
        # -- 上层迭代 --
        for g in range(self.max_gen):
            # 1) 评估
            for ind in upper_pop:
                if ind['fitness'] is None:  # 未评估
                    d1, d2 = ind['depots']
                    cost, route = self.solve_lower_level(d1, d2)
                    ind['fitness'] = cost
                    ind['route'] = route
            
            # 2) 记录当前最优
            best = min(upper_pop, key=lambda x: x['fitness'])
            if global_best is None or best['fitness'] < global_best['fitness']:
                global_best = copy.deepcopy(best)
            
            # 3) 生成后代（简单交叉或变异）
            new_pop = []
            # 简单精英保留
            upper_pop_sorted = sorted(upper_pop, key=lambda x: x['fitness'])
            new_pop.append(copy.deepcopy(upper_pop_sorted[0]))
            
            # 其余个体随机配对交叉
            while len(new_pop) < self.pop_size:
                p1 = random.choice(upper_pop)
                p2 = random.choice(upper_pop)
                d1_1, d1_2 = p1['depots']
                d2_1, d2_2 = p2['depots']
                # 交叉方式：随机对 depot1/depot2 交换
                if random.random() < 0.5:
                    child_depots = (d1_1, d2_2)
                else:
                    child_depots = (d2_1, d1_2)
                if child_depots[0] == child_depots[1]:
                    # 变异修正：随机重选 depot2
                    d_ = random.randint(0, self.n_depots - 1)
                    while d_ == child_depots[0]:
                        d_ = random.randint(0, self.n_depots - 1)
                    child_depots = (child_depots[0], d_)
                
                child = {'depots': child_depots, 'fitness': None, 'route': None}
                new_pop.append(child)
            
            upper_pop = new_pop
        
        return global_best


# -----------------------------------------------------------------------------
# 下层 TSP 问题类（给定特定 Depot 选择后使用）
# -----------------------------------------------------------------------------
class LowerLevelTSPProblem(ea.Problem):
    def __init__(self, coords_customers, depot1_coord, depot2_coord):
        """
        下层单目标问题：给定 depot1, depot2，求在第 50 个客户后访问 depot2，并最终返回 depot1 的最小总距离。
        这里我们直接在 aimFunc() 中实现 compute_bilevel_cost() 的计算即可。
        """
        self.coords_customers = coords_customers
        self.depot1_coord = depot1_coord
        self.depot2_coord = depot2_coord
        self.n = coords_customers.shape[0]
        name = "LowerLevelTSP"
        M = 1  # 单目标最小化
        maxormins = [1]  # 1 表示最小化
        Dim = self.n  # 100
        varTypes = [1] * Dim  # 排列编码
        lb = [0]*Dim
        ub = [self.n - 1]*Dim
        lbin = [1]*Dim
        ubin = [1]*Dim
        super().__init__(name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    
    def aimFunc(self, pop):
        routes = pop.Phen.astype(int)  # (NIND, 100)
        ObjV = []
        for route in routes:
            cost = compute_bilevel_cost(route, self.coords_customers, 
                                        self.depot1_coord, self.depot2_coord)
            ObjV.append([cost])
        pop.ObjV = np.array(ObjV)


# -----------------------------------------------------------------------------
# (B) 单层重构方法（Single-Level Reformulation）
# -----------------------------------------------------------------------------
class SingleLevelBiLevelTSPProblem(ea.Problem):
    """
    将 Depot 选择与 TSP 路径合并在一个染色体中（整数编码，Enc='I'）：
      - 前 2 个基因为 (d1, d2)，取值在 [0, 9] 且 d1 != d2。
      - 后 100 个基因为 TSP 路线，均在 [0, 99] 之间。
    
    在 aimFunc() 中强制：当访问完前 50 个客户时，必须访问 d2；最后必须返回 d1。
    如果不满足 d1 != d2 或出现其他约束冲突，可以使用惩罚或修正方式处理。
    """
    def __init__(self, coords_customers, coords_depots):
        self.coords_customers = coords_customers
        self.coords_depots = coords_depots
        self.n_depots = coords_depots.shape[0]   # 10
        self.n_customers = coords_customers.shape[0]  # 100
        
        name = "SingleLevelBiLevelTSP"
        M = 1                # 单目标
        maxormins = [1]      # 1表示最小化
        
        Dim = 2 + self.n_customers  # (d1, d2) + 100
        varTypes = [1]*Dim          # 全部为离散整数类型
        
        # 定义上下界
        lb = [0]*Dim
        ub = [0]*Dim
        # 前两个基因：Depot 选择 [0, 9]
        lb[0], ub[0] = 0, self.n_depots-1
        lb[1], ub[1] = 0, self.n_depots-1
        # 后 100 个基因：顾客索引 [0, 99]
        for i in range(2, Dim):
            lb[i] = 0
            ub[i] = self.n_customers - 1
        
        # 边界可以都设为闭区间 1
        lbin = [1]*Dim
        ubin = [1]*Dim
        
        # 调用父类构造函数，这里 geatpy 会利用 lb、ub、lbin、ubin 来生成 Field
        super().__init__(name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    
    def aimFunc(self, pop):
        X = pop.Phen.astype(int)  # (NIND, 102)
        ObjV = []
        for i in range(X.shape[0]):
            d1, d2 = X[i, 0], X[i, 1]
            route = X[i, 2:].copy()
            
            # 若 d1 == d2，直接判定不合法，给一个很大的惩罚
            if d1 == d2:
                ObjV.append([1e6])
                continue
            
            # 修正或惩罚重复顾客：保证后 100 基因构成一个无重复城市的排列
            route_unique, counts = np.unique(route, return_counts=True)
            if len(route_unique) < self.n_customers or np.any(counts > 1):
                # 执行修正，把重复城市换成缺失的城市
                missing_cities = set(range(self.n_customers)) - set(route_unique)
                missing_cities = list(missing_cities)
                idx_missing = 0
                used = set()
                route_fixed = []
                for city in route:
                    if city in used:
                        # 用缺失城市替换重复出现
                        city = missing_cities[idx_missing]
                        idx_missing += 1
                    route_fixed.append(city)
                    used.add(city)
                route = np.array(route_fixed)
            
            # 计算双层场景下的总距离
            cost = compute_bilevel_cost(route, 
                                        self.coords_customers,
                                        self.coords_depots[d1],
                                        self.coords_depots[d2])
            ObjV.append([cost])
        
        pop.ObjV = np.array(ObjV)
        
# -----------------------------------------------------------------------------
# (C) 单层重构方法-随机键（Single-Level Reformulation with Random Key）
# -----------------------------------------------------------------------------

class SingleLevelBiLevelTSPProblem_RK(ea.Problem):
    """
    单层重构方法 - 使用随机键 (Random Key) 编码:
      染色体长度: 2 + 100 = 102 维
        - 前 2 维: (d1, d2) ∈ [0,9], 实数，但会取 int(d+0.5) 作为 depot index
        - 后 100 维: [0,1] 区间的随机数，通过对这 100 个随机数的排序来确定客户访问顺序
    
    这样做能保证后 100 维自然构成一个无重复的排列(0..99)，无需在 aimFunc 中做去重修正。
    """
    def __init__(self, coords_customers, coords_depots):
        self.coords_customers = coords_customers
        self.coords_depots = coords_depots
        self.n_depots = coords_depots.shape[0]  # 10
        self.n_customers = coords_customers.shape[0]  # 100
        
        name = "SingleLevelBiLevelTSP_RK"
        M = 1                    # 单目标
        maxormins = [1]          # 1 -> 最小化
        Dim = 2 + self.n_customers  # 2 + 100 = 102
        varTypes = [0]*Dim       # 全部设为 0 (实数编码)
        
        # 构造上下界:
        lb = [0]*Dim
        ub = [1]*Dim
        
        # 前 2 维 depot 选择, 取值在 [0, 9], 虽然也用实数，但最后会取 int(d+0.5).
        # 所以可以把它们的上界设成 9.x:
        ub[0], ub[1] = 9, 9  # depot index ∈ [0..9]
        
        # 后 100 维: [0,1], 通过排序得到 0..99 的访问顺序
        # 故 lb[2..] = 0, ub[2..] = 1 即可
        
        # 边界均设置为闭区间
        lbin = [1]*Dim
        ubin = [1]*Dim
        
        # 调用父类构造函数
        super().__init__(name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    
    def aimFunc(self, pop):
        """
        对每个个体：
          1) 读取前 2 维 -> d1, d2 (取整后判定是否非法)
          2) 将后 100 维排序得到客户访问顺序 route
          3) compute_bilevel_cost(...) 计算总距离
        """
        X = pop.Phen  # shape (NIND, 102), 每个元素是实数 in [lb, ub]
        ObjV = []
        for i in range(X.shape[0]):
            # 1) 解析 depot
            #    由于 d1, d2 是实数, 用 round() 或 int()+0.5
            #    也可用 np.clip(...) 保证不越界
            d1 = int(round(X[i, 0]))
            d2 = int(round(X[i, 1]))
            
            # 若 d1 == d2 -> 惩罚
            if d1 == d2:
                ObjV.append([1e6])
                continue
            
            # 若 depot 超出 [0,9] -> 惩罚 (通常不应出现，但以防万一)
            if not (0 <= d1 < self.n_depots and 0 <= d2 < self.n_depots):
                ObjV.append([1e6])
                continue
            
            # 2) 构造 route: 对后 100 维做排序
            #    idx = np.argsort(X[i, 2:]) -> array([..., ...], dtype=int)
            #    route[k] = idx[k], 这会产生范围 [0..99], 不重复。
            random_keys = X[i, 2:]
            route = np.argsort(random_keys)  # 0..99
            # 注: np.argsort() 返回值本身就形如 [3,1,0,2,...], 不会有重复
            
            # 3) 计算双层 cost
            cost = compute_bilevel_cost(route, self.coords_customers,
                                        self.coords_depots[d1],
                                        self.coords_depots[d2])
            ObjV.append([cost])
        
        pop.ObjV = np.array(ObjV)