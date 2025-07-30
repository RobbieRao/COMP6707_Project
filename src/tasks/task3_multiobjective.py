import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import geatpy as ea
import numpy as np
from src.utils.data_loader import load_tsp_data
from src.utils.visualization import plot_tsp_route, plot_pareto_front
from src.models.multiobjective import MultiObjectiveTSPProblem

def run_multiobjective_tsp(tsp_file, mode="pareto", lambda_weight=0.1, pop_size=100, max_gen=500):
    """
    运行多目标 TSP 优化。
    
    Parameters:
        tsp_file (str): TSP.csv 文件路径，要求包含 "XCOORD", "YCOORD", "PROFIT" 列。
        mode (str): "pareto" 或 "weighted"。如果为 "weighted"，则采用加权单目标方法；否则采用多目标方法。
        lambda_weight (float): 当 mode=="weighted" 时使用的权重参数。
        pop_size (int): 种群规模。
        max_gen (int): 最大进化代数。
    
    Returns:
        dict: 包含以下信息：
            - "route": 最优解的排列（如果采用 weighted，则为单个解；如果采用 pareto，则为一组非支配解中的一解）
            - "ObjV": 对应的目标值
            - "problem": 问题实例
            - "res": 优化结果字典
    """
    # 1. 加载 TSP 数据
    tsp_data = load_tsp_data(tsp_file)
    coords = tsp_data["coords"]
    profit = tsp_data["profit"]
    
    # 2. 创建问题实例，根据 mode 设置 lambda_weight 参数
    if mode == "weighted":
        problem = MultiObjectiveTSPProblem(coords, profit, lambda_weight=lambda_weight)
    else:
        problem = MultiObjectiveTSPProblem(coords, profit, lambda_weight=None)
    
    # 3. 构造种群
    # 对于排列编码，不需要额外 FieldD
    population = ea.Population(Encoding='P', NIND=pop_size)
    
    # 4. 选择算法模板
    if mode == "weighted":
        # 加权单目标方法，使用单目标遗传算法模板
        algorithm = ea.soea_EGA_templet(problem, population)
    else:
        # 多目标方法，使用 NSGA-II 模板（或其他多目标模板）
        algorithm = ea.moea_NSGA2_templet(problem, population)
    
    algorithm.MAXGEN = max_gen
    algorithm.logTras = 1
    algorithm.verbose = False
    algorithm.drawing = 0
    
    # 5. 运行优化
    res = ea.optimize(algorithm, verbose=False, drawing=1, outputMsg=True, drawLog=False, saveFlag=False)
    
    # 6. 提取结果
    if mode == "weighted":
        best_route = res['Vars'][0].astype(int)
        best_ObjV = res['ObjV'][0]
        result_dict = {
            "route": best_route,
            "ObjV": best_ObjV,
            "problem": problem,
            "res": res
        }
    else:
        # 多目标情况：返回全部非支配解
        routes = res['Vars']
        ObjV = res['ObjV']
        result_dict = {
            "route": routes,
            "ObjV": ObjV,
            "problem": problem,
            "res": res
        }
    
    return result_dict

if __name__ == "__main__":
    # 构造 TSP.csv 的绝对路径（假设 data 目录在项目根目录下）
    tsp_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/TSP.csv"))
    
    # 选择模式： "weighted" 或 "pareto"
    # mode = "pareto"  # 尝试多目标方法
    mode = "weighted"  # 或加权方法，lambda_weight 可根据需要调整
    
    
    result = run_multiobjective_tsp(tsp_file, mode=mode, lambda_weight=0.1, pop_size=100, max_gen=500)
    
    if mode == "weighted":
        print(f"加权模式最优目标值: {result['ObjV']}")
        print("最优路线排列:", result["route"])
        # 可视化结果
        from src.utils.visualization import plot_tsp_route
        plot_tsp_route(result["problem"].coords, result["route"], title="Weighted Multiobjective TSP Route", show=True)
    else:
        # 多目标模式，输出非支配解个数，并绘制 Pareto 前沿图（使用目标空间的前两个维度）
        routes = result["route"]
        ObjV = result["ObjV"]
        print("多目标非支配解个数:", ObjV.shape[0])
        print("部分非支配解目标值:")
        print(ObjV[:10])
        # 绘制 Pareto 前沿（目标空间）
        from src.utils.visualization import plot_pareto_front
        plot_pareto_front(ObjV, title="Pareto Front (Multiobjective TSP)", show=True)
