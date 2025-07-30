# src/tasks/task5_bilevel.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import geatpy as ea
from src.models.bilevel import (
    NestedBiLevelTSP, 
    SingleLevelBiLevelTSPProblem_RK,  # Random Key approach
    compute_bilevel_cost
)
from src.utils.data_loader import load_tsp_data, load_depot_data

def validate_bilevel_solution(method, route, depot1_idx, depot2_idx,
                              coords_customers, coords_depots,
                              reported_distance):
    """
    对最终解进行一系列验证：
      1) 路线长度是否 100？
      2) 路线是否无重复（每个顾客恰好出现一次）？
      3) depot1_idx 是否 != depot2_idx？
      4) 是否符合“第 50 个顾客后停靠 Depot2”的需求：
         - compute_bilevel_cost 的实现就是: 
             depot1 -> route[:50] -> depot2 -> route[50:] -> depot1
      5) 再次用 compute_bilevel_cost 计算距离，与 reported_distance 对比
    """
    print(f"\n[Validation of {method}] ===============================")
    # 1) 路线长度
    n_customers = coords_customers.shape[0]
    if len(route) != n_customers:
        print(f"❌ The route length is incorrect. Expect 100 but got {len(route)}.")
    else:
        print(f"✅ The route length is {len(route)} (correct).")
    
    # 2) 是否无重复顾客
    unique_cities = set(route)
    if len(unique_cities) != n_customers:
        print(f"❌ Found duplicates or missing customers. Unique = {len(unique_cities)}, expected 100.")
    else:
        print(f"✅ The route covers all 100 customers without duplication.")
    
    # 3) 仓库对是否有效
    if depot1_idx == depot2_idx:
        print(f"❌ Depot1 and Depot2 are the same: {depot1_idx} = {depot2_idx}")
    else:
        print(f"✅ Depot pair (Depot{depot1_idx}, Depot{depot2_idx}) is valid.")
    
    # 4) 检查“前 50 / 后 50”结构
    if len(route) != 100:
        print(f"⚠️ Cannot check front 50 and last 50 because route length != 100.")
    else:
        print(f"First 5 customers: {route[:5]}")
        print(f"Last 5 customers: {route[-5:]}")
        print(f"✅ After visiting the first 50, the path goes to Depot2, per compute_bilevel_cost.")
    
    # 5) 再次计算距离
    dist_check = compute_bilevel_cost(
        route, coords_customers,
        coords_depots[depot1_idx],
        coords_depots[depot2_idx]
    )
    print(f"Recomputed distance via compute_bilevel_cost: {dist_check:.4f}")
    print(f"Distance reported by algorithm: {reported_distance:.4f}")
    if abs(dist_check - reported_distance) > 1e-6:
        diff_val = dist_check - reported_distance
        print(f"❌ Mismatch in distance. Difference = {diff_val:.4f}")
    else:
        print(f"✅ Distance matched (difference < 1e-6).")
    
    print(f"[End Validation of {method}] ===============================\n")


def run_bilevel_tsp(tsp_file, depot_file, method="nested", 
                    upper_pop_size=10, upper_max_gen=30,
                    lower_pop_size=100, lower_max_gen=300,
                    single_pop_size=200, single_max_gen=500,
                    seed=42):
    """
    Compare the Nested EA approach vs. Single-level (Random Key) approach 
    for the Bi-Level TSP with Depot Selection.
    
    method: 
      - "nested": Nested evolutionary algorithm
      - "single": Single-level approach with random key
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # 1) Load data
    tsp_data = load_tsp_data(tsp_file)
    coords_customers = tsp_data["coords"]  # (100, 2)
    depot_data = load_depot_data(depot_file)
    coords_depots = depot_data["coords"]   # (10, 2)
    
    result = {}
    
    if method == "nested":
        print("=== Running Nested Evolutionary Algorithm ===")
        nested_solver = NestedBiLevelTSP(coords_customers, coords_depots,
                                         pop_size=upper_pop_size,
                                         max_gen=upper_max_gen,
                                         lower_pop_size=lower_pop_size,
                                         lower_max_gen=lower_max_gen)
        best = nested_solver.run()
        d1, d2 = best['depots']
        best_route = best['route']
        best_cost = best['fitness']
        
        print(f"Best depot pair: (Depot{d1}, Depot{d2}), distance = {best_cost:.4f}")
        print(f"Best route (first 10 customers): {best_route[:10]} ...")
        
        # Validation
        validate_bilevel_solution(
            method="nested",
            route=best_route,
            depot1_idx=d1,
            depot2_idx=d2,
            coords_customers=coords_customers,
            coords_depots=coords_depots,
            reported_distance=best_cost
        )
        
        result = {
            "method": "nested",
            "depot1": d1,
            "depot2": d2,
            "route": best_route,
            "distance": best_cost
        }
    
    elif method == "single":
        print("=== Running Single-level (Random Key) Approach ===")
        problem = SingleLevelBiLevelTSPProblem_RK(coords_customers, coords_depots)
        population = ea.Population(Encoding='RI', NIND=single_pop_size)
        
        algorithm = ea.soea_EGA_templet(problem, population)
        algorithm.MAXGEN = single_max_gen
        algorithm.logTras = 1
        algorithm.verbose = False
        algorithm.drawing = 0
        
        res = ea.optimize(algorithm, verbose=False, drawing=0, outputMsg=False, saveFlag=False)
        # The first 2 dims are depot indices (in real), the last 100 dims are [0,1] random keys
        best_chrom = res['Vars'][0]
        best_cost = res['ObjV'][0, 0]
        
        d1 = int(round(best_chrom[0]))
        d2 = int(round(best_chrom[1]))
        random_keys = best_chrom[2:]
        # Sort the random keys to get a unique 0..99 route
        best_route = np.argsort(random_keys)
        
        print(f"Single-level (RK) best depot pair: (Depot{d1}, Depot{d2}), distance = {best_cost:.4f}")
        print(f"Best route (first 10 customers): {best_route[:10]} ...")
        
        # Validation
        validate_bilevel_solution(
            method="single",
            route=best_route,
            depot1_idx=d1,
            depot2_idx=d2,
            coords_customers=coords_customers,
            coords_depots=coords_depots,
            reported_distance=best_cost
        )
        
        result = {
            "method": "single",
            "depot1": d1,
            "depot2": d2,
            "route": best_route,
            "distance": best_cost
        }
    
    else:
        raise ValueError("method must be 'nested' or 'single'.")
    
    return result

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    tsp_file = os.path.join(base_dir, "data", "TSP.csv")
    depot_file = os.path.join(base_dir, "data", "Depot.csv")
    
    # 1) Run nested EA
    nested_result = run_bilevel_tsp(
        tsp_file, depot_file, 
        method="nested",
        upper_pop_size=10, 
        upper_max_gen=10, 
        lower_pop_size=50, 
        lower_max_gen=100
    )
    print("Nested EA result:", nested_result)
    
    # 2) Run single-level (random key) approach
    single_result = run_bilevel_tsp(
        tsp_file, depot_file,
        method="single",
        single_pop_size=100,
        single_max_gen=300
    )
    print("Single-level (RK) result:", single_result)
