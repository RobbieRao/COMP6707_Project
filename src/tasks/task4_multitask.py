# src/tasks/task4_multitask.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
from src.utils.data_loader import load_tsp_data
from src.utils.visualization import plot_tsp_route
from src.models.multitasking import MFEA_TSP
import random

def run_multitasking_tsp(tsp_file, RMP=0.3, pop_size=100, max_gen=500):
    """
    运行多任务 TSP 问题的 MFEA 算法。
    
    Parameters:
        tsp_file (str): TSP.csv 文件路径，要求包含 "XCOORD" 和 "YCOORD" 列。
        RMP (float): 随机交配概率。
        pop_size (int): 种群规模。
        max_gen (int): 最大进化代数。
    
    Returns:
        best_task0: 原始 TSP 最优个体（字典包含 'gene' 和 'obj'）。
        best_task1: 新 TSP 最优个体（字典包含 'gene' 和 'obj'）。
        new_coords (np.ndarray): 新 TSP 城市坐标数组。
    """
    np.random.seed(None)
    random.seed(None)
    # 加载原始 TSP 数据
    tsp_data = load_tsp_data(tsp_file)
    coords = tsp_data["coords"]
    n = coords.shape[0]
    # 生成新 TSP 数据：对每个城市坐标分别加上 [0,50] 内的随机扰动
    new_coords = coords.copy()
    for i in range(n):
        new_coords[i, 0] = coords[i, 0] + np.random.uniform(0, 50)
        new_coords[i, 1] = coords[i, 1] + np.random.uniform(0, 50)
    # 实例化 MFEA_TSP（两个任务均包含 n 个城市）
    mfea = MFEA_TSP(coords, new_coords, pop_size=pop_size, max_gen=max_gen, RMP=RMP)
    best_task0, best_task1 = mfea.run()
    return best_task0, best_task1, new_coords

if __name__ == "__main__":
    tsp_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/TSP.csv"))
    RMP = 0.3
    pop_size = 100
    max_gen = 500
    best_task0, best_task1, new_coords = run_multitasking_tsp(tsp_file, RMP=RMP, pop_size=pop_size, max_gen=max_gen)
    
    print("【Task4】多任务 TSP 求解结果：")
    if best_task0 is not None:
        print("原始 TSP 最优巡回距离: {:.4f}".format(best_task0['obj']))
        print("原始 TSP 最优路线排列:", best_task0['gene'])
    if best_task1 is not None:
        print("新 TSP 最优巡回距离: {:.4f}".format(best_task1['obj']))
        print("新 TSP 最优路线排列:", best_task1['gene'])
    
    # 利用可视化工具展示最佳路线
    original_coords = load_tsp_data(tsp_file)["coords"]
    plot_tsp_route(original_coords, best_task0['gene'], title="Best Route for Original TSP", show=True)
    plot_tsp_route(new_coords, best_task1['gene'], title="Best Route for New TSP", show=True)
