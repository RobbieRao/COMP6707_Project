import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from src.utils.data_loader import load_tsp_data
from src.utils.visualization import plot_tsp_route
from src.models.large_scale import LargeScaleTSPProblem

def run_large_scale_tsp(tsp_file, pop_size_local=50, max_gen_local=200, n_clusters=4):
    """
    运行大规模 TSP 求解。
    
    Parameters:
        tsp_file (str): TSP.csv 文件路径（包含原 100 个客户数据）。
        pop_size_local (int): 局部 TSP 求解时的种群规模。
        max_gen_local (int): 局部 TSP 求解时的最大迭代代数。
        n_clusters (int): 聚类的数量。
    
    Returns:
        dict: 包含以下信息：
            - "global_route": 全局最优巡回路线（全局索引，长度为 200）
            - "global_distance": 全局巡回总距离
            - "full_coords": 200 个客户的坐标数组
    """
    # 1. 加载 TSP 数据（原 100 个客户）
    tsp_data = load_tsp_data(tsp_file)
    original_coords = tsp_data["coords"]
    
    # 2. 创建大规模 TSP 问题实例（内部会生成额外 100 个客户，合并成 200 个）
    large_tsp = LargeScaleTSPProblem(original_coords, n_clusters=n_clusters)
    
    # 3. 求解大规模 TSP 问题
    global_route, global_distance = large_tsp.solve(pop_size_local=pop_size_local, max_gen_local=max_gen_local)
    
    return {
        "global_route": global_route,
        "global_distance": global_distance,
        "full_coords": large_tsp.full_coords
    }

if __name__ == "__main__":
    # 构造 TSP.csv 的绝对路径（假设 data 目录在项目根目录下）
    tsp_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/TSP.csv"))
    result = run_large_scale_tsp(tsp_file, pop_size_local=50, max_gen_local=200, n_clusters=4)
    
    print(f"全局巡回总距离: {result['global_distance']:.4f}")
    print("全局最优路径（全局索引）:", result["global_route"])
    
    # 可视化大规模 TSP 全局最优路径
    plot_tsp_route(result["full_coords"], result["global_route"], title="Large-Scale TSP Global Route", show=True)
