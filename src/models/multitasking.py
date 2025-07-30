# src/models/multitasking.py
import numpy as np
import random
import copy
from src.utils.metrics import route_distance

class MFEA_TSP:
    """
    基于多因子进化算法（MFEA）解决多任务 TSP 问题。
    
    两个任务：
      Task 0：原始 TSP（coords1）
      Task 1：新 TSP（coords2），通过对原始坐标添加随机扰动生成
    
    算法流程：
      1. 初始化全局种群，每个个体计算两个任务的适应度，并分配技能因子为表现较优的任务。
      2. Divide：按照技能因子将全局种群划分为两个子种群。
      3. 迭代过程：
         - 对每个子种群分别进行父代选择（选取优秀个体的一半）。
         - 父代配对生成子代：根据是否同任务或随机值小于 RMP，决定是否进行交叉操作，
           子代继承随机选择的父代技能因子，并只在对应任务上进行评价（选择性模仿）。
         - 合并父代与子代后，在各子种群内根据适应度排序选择出固定数量的优秀个体，形成下一代子种群。
      4. 返回两个任务上各自的最佳个体。
    """
    def __init__(self, coords1, coords2, pop_size=100, max_gen=500, RMP=0.3, mutation_rate=0.2):
        """
        Parameters:
            coords1 (np.ndarray): Task 0 中城市坐标，形状为 (n, 2)。
            coords2 (np.ndarray): Task 1 中城市坐标，形状为 (n, 2)。
            pop_size (int): 全局种群规模。
            max_gen (int): 最大进化代数。
            RMP (float): 随机交配概率，控制跨任务交叉的概率。
            mutation_rate (float): 变异概率。
        """
        self.coords1 = coords1
        self.coords2 = coords2
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.RMP = RMP
        self.mutation_rate = mutation_rate
        self.n_cities = coords1.shape[0]
        self.population = []  # 全局种群，每个个体为字典：{'gene', 'skill_factor', 'obj'}
    
    def evaluate(self, gene, task):
        """
        仅在指定任务上计算适应度（总巡回距离）。
        """
        if task == 0:
            return route_distance(self.coords1, gene)
        elif task == 1:
            return route_distance(self.coords2, gene)
        else:
            raise ValueError("任务编号错误。")
    
    def initialize(self):
        """
        初始化全局种群：对每个个体计算两个任务适应度，
        并根据表现较好的任务分配技能因子，选择对应任务的适应度作为个体目标值。
        """
        self.population = []
        for _ in range(self.pop_size):
            gene = np.random.permutation(self.n_cities)
            f1 = self.evaluate(gene, 0)
            f2 = self.evaluate(gene, 1)
            if f1 < f2:
                skill_factor = 0
                obj = f1
            elif f2 < f1:
                skill_factor = 1
                obj = f2
            else:
                skill_factor = random.choice([0, 1])
                obj = f1
            individual = {'gene': gene, 'skill_factor': skill_factor, 'obj': obj}
            self.population.append(individual)
    
    def divide_population(self):
        """
        按照技能因子将全局种群划分为两个子种群。
        Returns:
            dict: {0: [个体列表], 1: [个体列表]}
        """
        subpop = {0: [], 1: []}
        for ind in self.population:
            subpop[ind['skill_factor']].append(ind)
        return subpop
    
    def select_parents(self, subpop):
        """
        从子种群中选择一半的优秀个体作为父代（基于目标值排序，值越小越优）。
        """
        subpop_sorted = sorted(subpop, key=lambda ind: ind['obj'])
        num = len(subpop_sorted)
        selected = subpop_sorted[:max(1, num // 2)]
        return selected
    
    def order_crossover(self, p1, p2):
        """
        Order Crossover (OX) 操作。
        """
        n = len(p1)
        child1 = -np.ones(n, dtype=int)
        child2 = -np.ones(n, dtype=int)
        pt1, pt2 = sorted(random.sample(range(n), 2))
        child1[pt1:pt2] = p1[pt1:pt2]
        child2[pt1:pt2] = p2[pt1:pt2]
        current_index = pt2
        for gene in list(p2[pt2:]) + list(p2[:pt2]):
            if gene not in child1:
                if current_index >= n:
                    current_index = 0
                child1[current_index] = gene
                current_index += 1
        current_index = pt2
        for gene in list(p1[pt2:]) + list(p1[:pt2]):
            if gene not in child2:
                if current_index >= n:
                    current_index = 0
                child2[current_index] = gene
                current_index += 1
        return child1, child2
    
    def swap_mutation(self, gene):
        """
        交换变异：随机交换基因中两个位置的值。
        """
        gene = gene.copy()
        i, j = random.sample(range(len(gene)), 2)
        gene[i], gene[j] = gene[j], gene[i]
        return gene
    
    def reproduction(self, parents):
        """
        父代配对生成子代：
          - 随机配对；
          - 若父代技能因子相同或随机数 < RMP，则执行交叉，
            否则直接复制各自基因（再经变异）。
          - 子代继承随机选择的父代技能因子，并只在该任务上进行适应度评价（选择性模仿）。
        """
        offspring = []
        num_parents = len(parents)
        indices = list(range(num_parents))
        random.shuffle(indices)
        for i in range(0, num_parents - 1, 2):
            p1 = parents[indices[i]]
            p2 = parents[indices[i+1]]
            if (p1['skill_factor'] == p2['skill_factor']) or (random.random() < self.RMP):
                # 执行交叉操作
                child_gene1, child_gene2 = self.order_crossover(p1['gene'], p2['gene'])
                if random.random() < self.mutation_rate:
                    child_gene1 = self.swap_mutation(child_gene1)
                if random.random() < self.mutation_rate:
                    child_gene2 = self.swap_mutation(child_gene2)
                # 子代技能因子：随机选择父代之一
                sf1 = random.choice([p1['skill_factor'], p2['skill_factor']])
                sf2 = random.choice([p1['skill_factor'], p2['skill_factor']])
                child1 = {'gene': child_gene1, 'skill_factor': sf1}
                child2 = {'gene': child_gene2, 'skill_factor': sf2}
                child1['obj'] = self.evaluate(child_gene1, sf1)
                child2['obj'] = self.evaluate(child_gene2, sf2)
                offspring.extend([child1, child2])
            else:
                # 不交叉，直接复制并变异
                child_gene1 = p1['gene'].copy()
                child_gene2 = p2['gene'].copy()
                if random.random() < self.mutation_rate:
                    child_gene1 = self.swap_mutation(child_gene1)
                if random.random() < self.mutation_rate:
                    child_gene2 = self.swap_mutation(child_gene2)
                child1 = {'gene': child_gene1, 'skill_factor': p1['skill_factor']}
                child2 = {'gene': child_gene2, 'skill_factor': p2['skill_factor']}
                child1['obj'] = self.evaluate(child_gene1, p1['skill_factor'])
                child2['obj'] = self.evaluate(child_gene2, p2['skill_factor'])
                offspring.extend([child1, child2])
        if num_parents % 2 == 1:
            p = parents[indices[-1]]
            child_gene = p['gene'].copy()
            if random.random() < self.mutation_rate:
                child_gene = self.swap_mutation(child_gene)
            child = {'gene': child_gene, 'skill_factor': p['skill_factor']}
            child['obj'] = self.evaluate(child_gene, p['skill_factor'])
            offspring.append(child)
        return offspring
    
    def select_next_generation(self, subpop, desired_size):
        """
        对单个任务子种群进行排序选择，保留 desired_size 个个体。
        """
        sorted_subpop = sorted(subpop, key=lambda ind: ind['obj'])
        return sorted_subpop[:desired_size]
    
    def run(self):
        """
        执行 MFEA 进化过程，并返回两个任务上各自的最佳个体。
        """
        self.initialize()
        for gen in range(self.max_gen):
            subpop = self.divide_population()
            new_subpop = {0: [], 1: []}
            # 针对每个任务分别进行繁殖与选择
            for task in [0, 1]:
                if len(subpop[task]) == 0:
                    continue
                parents = self.select_parents(subpop[task])
                offspring = self.reproduction(parents)
                combined = subpop[task] + offspring
                new_subpop[task] = self.select_next_generation(combined, self.pop_size // 2)
            # 形成新的全局种群
            self.population = new_subpop[0] + new_subpop[1]
            # 补充不足的个体（若有）
            while len(self.population) < self.pop_size:
                gene = np.random.permutation(self.n_cities)
                sf = random.choice([0, 1])
                obj = self.evaluate(gene, sf)
                self.population.append({'gene': gene, 'skill_factor': sf, 'obj': obj})
            best_task0 = min(new_subpop[0], key=lambda ind: ind['obj']) if new_subpop[0] else None
            best_task1 = min(new_subpop[1], key=lambda ind: ind['obj']) if new_subpop[1] else None
            best_obj0 = best_task0['obj'] if best_task0 is not None else float('inf')
            best_obj1 = best_task1['obj'] if best_task1 is not None else float('inf')
            #print(f"Generation {gen+1}: Task0 Best = {best_obj0:.4f} | Task1 Best = {best_obj1:.4f}")
        final_subpop = self.divide_population()
        best_task0 = min(final_subpop[0], key=lambda ind: ind['obj']) if final_subpop[0] else None
        best_task1 = min(final_subpop[1], key=lambda ind: ind['obj']) if final_subpop[1] else None
        return best_task0, best_task1
