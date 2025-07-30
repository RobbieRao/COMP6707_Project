import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import geatpy as ea
import numpy as np

from src.utils.data_loader import load_tsp_data
from src.utils.visualization import plot_tsp_route
from src.models.tsp_problem import TSPProblem

def run_task1_tsp(tsp_file, pop_size=100, max_gen=500):
    """
    运行经典 TSP 遗传算法求解，并返回最优解信息。
    
    Parameters:
        tsp_file (str): TSP.csv 文件路径。
        pop_size (int): 种群规模 NIND。
        max_gen (int): 最大进化代数 MAXGEN。
    
    Returns:
        dict: 包含以下信息：
            - "best_route": 最优路线的排列
            - "best_distance": 对应的总距离
            - "problem": 问题实例
            - "res": Geatpy 优化结果字典
    """
    # 1. 加载 TSP 数据（仅使用坐标）
    tsp_data = load_tsp_data(tsp_file)
    coords = tsp_data["coords"]  # 形状 (100, 2)
    
    # 2. 创建问题实例
    problem = TSPProblem(coords)
    
    # 3. 构造种群，不需要传入 FieldD（对于排列编码 "P"）
    population = ea.Population(Encoding='P', NIND=pop_size)
    
    # 4. 选择算法模板并设置参数（使用简单遗传算法模板）
    algorithm = ea.soea_SEGA_templet(problem, population)
    algorithm.MAXGEN = max_gen     # 最大进化代数
    algorithm.logTras = 0          # 日志记录间隔
    algorithm.verbose = False       # 是否打印中间信息
    algorithm.drawing = 1          # 设为 0 表示不在算法内部绘图
    
    # 5. 运行优化
    res = ea.optimize(algorithm, 
                      verbose=False, 
                      drawing=1,      # 1 表示绘制收敛图
                      outputMsg=True, 
                      drawLog=False, 
                      saveFlag=False)
    
    # 6. 提取最优解
    best_route = res['Vars'][0].astype(int)
    best_distance = res['ObjV'][0][0]
    
    return {
        "best_route": best_route,
        "best_distance": best_distance,
        "problem": problem,
        "res": res
    }

if __name__ == "__main__":
    tsp_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/TSP.csv"))
    result = run_task1_tsp(tsp_file, pop_size=100, max_gen=500)
    
    print(f"最优路线总距离: {result['best_distance']:.4f}")
    print("最优路线排列:", result["best_route"])
    
    # 可视化最优路线
    coords = result["problem"].coords
    plot_tsp_route(coords, result["best_route"], title="Best TSP Route", show=True)
