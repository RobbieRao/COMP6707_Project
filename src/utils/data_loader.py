# data_loader.py
import pandas as pd
import numpy as np

def load_tsp_data(file_path):
    """
    Returns:
        dict: 包含以下键：
            - "cust_no": 顾客编号数组
            - "coords": 顾客坐标数组，形状为 (n, 2)
            - "profit": 利润数组
            - "ready_time": 准备时间数组
            - "due_time": 截止时间数组
            - "data": 原始 DataFrame 数据
    """
    data = pd.read_csv(file_path)
    cust_no = data["CUST NO."].values
    coords = data[["XCOORD", "YCOORD"]].values.astype(float)
    profit = data["PROFIT"].values.astype(float)
    ready_time = data["READY TIME"].values.astype(float)
    due_time = data["DUE TIME"].values.astype(float)
    
    return {
        "cust_no": cust_no,
        "coords": coords,
        "profit": profit,
        "ready_time": ready_time,
        "due_time": due_time,
        "data": data
    }

def load_depot_data(file_path):
    """
    Returns:
        dict: 包含以下键：
            - "depot_no": Depot 编号数组
            - "coords": Depot 坐标数组，形状为 (m, 2)
            - "data": 原始 DataFrame 数据
    """
    data = pd.read_csv(file_path)
    depot_no = data["DEPOT NO."].values
    coords = data[["XCOORD", "YCOORD"]].values.astype(float)
    
    return {
        "depot_no": depot_no,
        "coords": coords,
        "data": data
    }


#Test Code
# if __name__ == "__main__":
#     tsp_file = "data/TSP.csv"
#     depot_file = "data/Depot.csv"
#     tsp_info = load_tsp_data(tsp_file)
#     print("TSP 数据加载成功：")
#     print("顾客编号:", tsp_info["cust_no"])
#     print("顾客坐标形状:", tsp_info["coords"].shape)
#     print("利润数组形状:", tsp_info["profit"].shape)
#     print("准备时间数组形状:", tsp_info["ready_time"].shape)
#     print("截止时间数组形状:", tsp_info["due_time"].shape)
    
#     depot_info = load_depot_data(depot_file)
#     print("\nDepot 数据加载成功：")
#     print("Depot 编号:", depot_info["depot_no"])
#     print("Depot 坐标形状:", depot_info["coords"].shape)
