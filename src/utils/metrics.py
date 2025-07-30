# metrics.py
import numpy as np

def route_distance(coords, route):
    """
    计算给定路线的总巡回距离（包括返回起点）。
    
    Parameters:
        coords (np.ndarray): 顾客坐标数组，形状为 (n, 2)。
        route (list or np.ndarray): 表示巡回顺序的顾客索引数组。
        
    Returns:
        float: 整个巡回路线的总距离。
    """
    route = np.array(route, dtype=int)
    distance = 0.0
    for i in range(len(route) - 1):
        distance += np.linalg.norm(coords[route[i]] - coords[route[i+1]])
    # 加上从最后一个城市回到起点的距离
    distance += np.linalg.norm(coords[route[-1]] - coords[route[0]])
    return distance

def calc_igd(front, reference_points):
    """
    计算 IGD（Inverted Generational Distance）。
    
    IGD 是评估非支配解集逼近真实 Pareto 前沿质量的指标之一。
    定义为参考集合中的每个点到当前前沿的最小欧氏距离的平均值。
    
    Parameters:
        front (np.ndarray): 当前解集的目标值，形状为 (n, m)，n 表示解数目，m 表示目标维度。
        reference_points (np.ndarray): 参考前沿点集，形状为 (k, m)。
        
    Returns:
        float: IGD 指标值，值越小表示越接近参考前沿。
    """
    total_dist = 0.0
    for ref_point in reference_points:
        dist_list = np.linalg.norm(front - ref_point, axis=1)
        total_dist += np.min(dist_list)
    return total_dist / len(reference_points)

def calc_hv(front, reference_point):
    """
    计算 Hypervolume（HV）。
    
    HV 指标度量的是前沿与一个参考点（通常是比前沿所有点都要劣的点）
    之间包围的体积。对于 2D 情况就是面积，3D 情况是体积，以此类推。
    
    这里采用一种简单的蒙特卡洛估计方法，仅适合演示或小规模场景。
    对于高维或精度要求高的情况，需使用更专业的 HV 算法。
    
    Parameters:
        front (np.ndarray): 当前解集的目标值，形状为 (n, m)。
        reference_point (np.ndarray): 形状为 (m,) 的参考点（坐标比前沿所有点都劣）。
        
    Returns:
        float: HV 指标的近似值。值越大表示前沿越优（在参考点固定的前提下）。
    """
    # 简易蒙特卡洛估计，可能需要自行调整采样数
    num_samples = 10000
    mins = np.min(front, axis=0)
    maxs = reference_point  # 假设 reference_point >= front 所有坐标
    
    # 如果任意维度上 front 超过 reference_point，则 HV 不可计算或需要修改
    if np.any(mins > maxs):
        raise ValueError("参考点必须比前沿所有点更劣。")
    
    # 随机采样
    samples = np.random.uniform(low=mins, high=maxs, size=(num_samples, front.shape[1]))
    # 判断每个采样点是否被前沿支配
    dominated_count = 0
    for s in samples:
        # 如果前沿中存在解支配点 s，则说明 s 落在超体积区域内
        # 多目标支配定义：front[i] <= s 并且至少有一维 < s
        # 这里用所有解进行检查，会较慢，适合演示
        for f in front:
            if np.all(f <= s) and np.any(f < s):
                dominated_count += 1
                break
    
    box_volume = np.prod(maxs - mins)  # 超长方体体积
    hv_estimate = (dominated_count / num_samples) * box_volume
    return hv_estimate

def calc_spacing(front):
    """
    计算 Spacing 指标，用于衡量多目标解在前沿上的分布均匀性。
    
    定义为：Spacing = sqrt( (1/(n-1)) * sum( (d_i - d_mean)^2 ) )
    其中 d_i 是每个解到其最近邻解的距离，d_mean 是所有 d_i 的平均值。
    
    Parameters:
        front (np.ndarray): 当前解集的目标值，形状为 (n, m)。
        
    Returns:
        float: Spacing 指标值，值越小表示解在前沿上分布越均匀。
    """
    n = front.shape[0]
    if n < 2:
        return 0.0
    
    # 计算每个解到最近邻解的距离 d_i
    dist_matrix = np.linalg.norm(front[:, None, :] - front[None, :, :], axis=-1)
    # 将对角线距离（解到自身距离）设为 np.inf，以便取最小值
    np.fill_diagonal(dist_matrix, np.inf)
    d_i = np.min(dist_matrix, axis=1)
    d_mean = np.mean(d_i)
    
    spacing_val = np.sqrt(np.sum((d_i - d_mean) ** 2) / (n - 1))
    return spacing_val

# Test Code
# if __name__ == "__main__":
#     # 测试 route_distance
#     coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
#     route = [0, 1, 2, 3]
#     dist = route_distance(coords, route)
#     print("测试 TSP 路径距离:", dist)
    
#     # 测试多目标指标
#     front = np.array([[1, 5], [2, 3], [3, 2], [4, 1]])
#     ref_points = np.array([[1.5, 4], [3, 3], [5, 1.5]])
#     igd_val = calc_igd(front, ref_points)
#     print("IGD:", igd_val)
    
#     hv_val = calc_hv(front, reference_point=np.array([5, 5]))
#     print("HV (估计):", hv_val)
    
#     spacing_val = calc_spacing(front)
#     print("Spacing:", spacing_val)
