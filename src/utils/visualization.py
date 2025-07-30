# visualization.py
import matplotlib.pyplot as plt
import numpy as np

def plot_tsp_route(coords, route, title="TSP Optimal Route", show=True, save_path=None):
    """
    绘制 TSP 问题的巡回路径。
    
    Parameters:
        coords (np.ndarray): 顾客坐标数组，形状为 (n, 2)。
        route (list 或 np.ndarray): 表示巡回顺序的顾客索引数组。
        title (str): 图形标题。
        show (bool): 是否显示图形（调用 plt.show()）。
        save_path (str): 如果不为 None，则保存图像至指定路径。
    """
    # 确保 route 为整数型数组
    route = np.array(route, dtype=int)
    # 根据排列获取对应坐标
    route_coords = coords[route]
    # 为闭合路径，追加起点到末尾
    route_coords = np.vstack([route_coords, route_coords[0]])
    
    plt.figure(figsize=(8, 6))
    plt.plot(route_coords[:, 0], route_coords[:, 1], marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_clusters(coords, labels, title="Clustered Customers", show=True, save_path=None):
    """
    绘制聚类后的顾客分布图，不同聚类用不同颜色显示。
    
    Parameters:
        coords (np.ndarray): 顾客坐标数组，形状为 (n, 2)。
        labels (np.ndarray 或 list): 每个顾客对应的聚类标签。
        title (str): 图形标题。
        show (bool): 是否显示图形（调用 plt.show()）。
        save_path (str): 如果不为 None，则保存图像至指定路径。
    """
    coords = np.array(coords)
    labels = np.array(labels)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis", s=50)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(scatter)
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_pareto_front(obj_values, title="Pareto Front", show=True, save_path=None):
    """
    绘制多目标优化问题的 Pareto 前沿（二维散点图）。
    
    Parameters:
        obj_values (np.ndarray): 目标值数组，形状为 (n, m)。
            对于双目标问题，绘制二维散点图；如果 m>2，则只绘制前两个目标。
        title (str): 图形标题。
        show (bool): 是否显示图形（调用 plt.show()）。
        save_path (str): 如果不为 None，则保存图像至指定路径。
    """
    obj_values = np.array(obj_values)
    plt.figure(figsize=(8, 6))
    # 默认绘制第一个和第二个目标
    plt.scatter(obj_values[:, 0], obj_values[:, 1], marker='o')
    plt.xlabel("Goal 1")
    plt.ylabel("Goal 2")
    plt.title(title)
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

# Test Code
# if __name__ == "__main__":
#     # 测试 plot_tsp_route
#     coords = np.array([[0, 0], [1, 2], [3, 1], [4, 4]])
#     route = [0, 1, 2, 3]
#     plot_tsp_route(coords, route, title="Test TSP Route", show=True)
    
#     # 测试 plot_clusters
#     labels = [0, 1, 0, 1]
#     plot_clusters(coords, labels, title="Test Cluster Plot", show=True)
    
#     # 测试 plot_pareto_front
#     obj_values = np.array([[1, 5], [2, 3], [3, 2], [4, 1]])
#     plot_pareto_front(obj_values, title="Test Pareto Front", show=True)
