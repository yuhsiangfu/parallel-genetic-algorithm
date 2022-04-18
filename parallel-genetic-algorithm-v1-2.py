"""
Parallel Genetic Algorithm for TSP, Ver. 1.2

@auth: Yu-Hsiang Fu
@date: 2022/04/12
"""
# --------------------------------------------------------------------------------
# 1.Import modular
# --------------------------------------------------------------------------------
import copy
import multiprocessing as mp
import networkx as nx
import numpy as np
import os
import random
import sys
import time
import util.pickle_func as pf


# --------------------------------------------------------------------------------
# 2.Define function
# --------------------------------------------------------------------------------
def load_data_to_graph(file_path: str):
    try:
        matrix = list()

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f.readlines():
                matrix.append([int(i) for i in line.strip().split()])

        matrix_np = np.array(matrix)
        g = nx.from_numpy_array(matrix_np)
    except:
        g = nx.Graph()  # empty graph

    return g


def generate_path(g: nx.Graph):
    node_set = set(g.nodes())
    path = list()

    while len(node_set) > 0:
        selected_node = random.choice(list(node_set))  # random select
        path.append(selected_node)
        node_set -= {selected_node}

    return path


# --------------------------------------------------------------------------------
# 3.Define genetic-function
# --------------------------------------------------------------------------------
def calculate_cost(g: nx.Graph, p: list):
    # sum(cost[weight] of nodes on the path p)
    return p, sum([g.edges[p[i-1], p[i]]["weight"] for i in range(1, len(p))])


def order1_exchange(pi: list, pj: list, rate_crossover: float):
    if random.random() < rate_crossover:
        len_indv = len(pi)

        # two-points
        cut1 = random.randint(1, len_indv - 2)
        cut2 = random.randint(1, len_indv - 2)

        if cut1 != cut2:
            if cut1 > cut2:
                cut1, cut2 = cut2, cut1
            # else:
            #     pass  # do nothing
        else:
            cut2 += 1

        # combine new individual
        mi = pi[cut1: cut2]  # exchange mid-part
        mj = pj[cut1: cut2]

        # ci
        c1 = [j for j in pj if j not in mi]
        ci = c1[:cut1] + mi + c1[cut1:]

        # cj
        c2 = [i for i in pi if i not in mj]
        cj = c2[:cut1] + mj + c2[cut1:]

        return ci, cj
    else:
        return pi, pj


def mutate(indv: list, rate_mutation: float):
    # bit-by-bit mutation
    len_indv = len(indv)

    for i in range(len_indv):
        if random.random() < rate_mutation:
            # do exchange
            _j = random.randint(0, len_indv-1)
            while i == _j:
                _j = random.randint(0, len_indv-1)
            indv[i], indv[_j] = indv[_j], indv[i]
        # else:
        #     # do not exchange
        #     pass

    return indv


# --------------------------------------------------------------------------------
# 4.Serial genetic-operation
# --------------------------------------------------------------------------------
def initial_population(g: nx.Graph, size_population=100):
    return [generate_path(g) for _ in range(size_population)]


def fitness(g: nx.Graph, population: list):
    return sorted([calculate_cost(g, p) for p in population], key=lambda x: x[1])


def selection(population_fit: list, rate_selection=0.2):
    # cut the population and remove fitness
    return [indv for indv, _ in population_fit[: int(len(population_fit) * rate_selection)]]


def crossover(population: list, size_population=100, rate_crossover=0.8):
    # make odd number of population size to even number
    if len(population) % 2:
        random_fill = random.choice(population)
        population.append(pf.deepcopy(random_fill))

    index_start = len(population)
    index_end = size_population
    for _ in range(index_start, index_end, 2):
        pi, pj = pf.deepcopy(random.sample(population, 2))
        population += [*order1_exchange(pi, pj, rate_crossover)]

    return population


def mutation(population: list, rate_mutation=0.05):
    return [mutate(indv, rate_mutation) for indv in population]


# --------------------------------------------------------------------------------
# 5.Parallel genetic-operation
# --------------------------------------------------------------------------------
def p_fitness(p: mp.Pool, g: nx.Graph, population: list):
    return sorted(p.starmap(calculate_cost, [(g, indv) for indv in population]), key=lambda x: x[1])


def p_crossover(p: mp.Pool, population: list, size_population: int, rate_crossover: float):
    if len(population) % 2:  # size is odd number
        random_fill = random.choice(population)
        population.append(pf.deepcopy(random_fill))

    index_start = len(population)
    index_end = size_population
    exchanged_list = p.starmap(order1_exchange,
                               [(*pf.deepcopy(random.sample(population, 2)), rate_crossover)
                                for _ in range(index_start, index_end, 2)])
    population += [j for i in exchanged_list for j in i]  # flatten of exchanged_list (list of list)

    return population


def p_mutation(p: mp.Pool, population: list, rate_mutation=float):
    return p.starmap(mutate, [(indv, rate_mutation) for indv in population])


# --------------------------------------------------------------------------------
# 6.Genetic algorithm
# --------------------------------------------------------------------------------
def genetic_algorithm(param: dict, is_parallel=False, is_show_msg=True):
    # initial population
    g = param["graph"]
    population = initial_population(g, size_population=param["size_population"])

    # ga-optimization
    best_evo_fit = sys.maxsize
    best_evo_num = 0
    best_evo_path = None

    pool = mp.Pool(param["num_cpu"])
    with pool as p:
        for i in range(param["num_evolution"]):
            if is_show_msg:
                print(f"  - evo-{i}")

            best_fit = sys.maxsize
            best_path = None

            for j in range(param["num_iteration"]):
                if is_show_msg:
                    print(f"   - iter-{j}, ", end="")

                # evaluation
                if is_parallel:
                    population_fit = p_fitness(p, g, population)
                else:
                    population_fit = fitness(g, population)
                min_path, min_fit = pf.deepcopy(population_fit[0])

                if min_fit < best_fit:
                    best_fit = copy.copy(min_fit)
                    best_path = pf.deepcopy(min_path)

                # genetic operation
                population = selection(population_fit, param["rate_selection"])
                if is_parallel:
                    population = p_crossover(p, population, param["size_population"], param["rate_crossover"])
                    population = p_mutation(p, population, param["rate_mutation"])
                else:
                    population = crossover(population, param["size_population"], param["rate_crossover"])
                    population = mutation(population, param["rate_mutation"])

                if is_show_msg:
                    print(f"fit={best_fit}")

            # best-evo
            if best_fit < best_evo_fit:
                best_evo_fit = copy.copy(best_fit)
                best_evo_num = i + 1
                best_evo_path = pf.deepcopy(best_path)

        print(f"  - final evo-num:  {best_evo_num}")
        print(f"  - final evo-fit:  {best_evo_fit}")
        print(f"  - final evo-path: {best_evo_path}")

    return best_evo_path, best_evo_fit, best_evo_num


# --------------------------------------------------------------------------------
# 7.Main function
# --------------------------------------------------------------------------------
def main_function():
    start_time = time.time()

    print("Parallel-Genetic-Algorithm")
    print("0. load TSP dataset")
    filename_list = ["att48_d", "dantzig42_d", "fri26_d", "gr17_d", "p01_d"]
    pickle_graph = "tsp-graph.pickle"

    if not os.path.isfile(pickle_graph):
        graph_dict = dict()
        for filename in filename_list:
            print(f" - load: {filename}")
            g = load_data_to_graph(file_path=f"./data/{filename}.txt")
            graph_dict[filename] = g

        pf.save(pickle_graph, graph_dict)
    else:
        print(f" - load: {pickle_graph}")
        graph_dict = pf.load(pickle_graph)

    print("1. config of parameter")
    graph_name = "att48_d"
    num_cpu = 8
    num_evolution = 1
    num_iteration = 2000
    size_population = 2000
    rate_selection = 0.2
    rate_crossover = 0.8
    rate_mutation = 0.05

    # param
    param = dict()
    param["graph"] = graph_dict[graph_name]
    param["graph_name"] = graph_name
    param["num_cpu"] = num_cpu
    param["num_evolution"] = num_evolution
    param["num_iteration"] = num_iteration
    param["size_population"] = size_population
    param["rate_selection"] = rate_selection
    param["rate_crossover"] = rate_crossover
    param["rate_mutation"] = rate_mutation

    print(f" - graph: {graph_name}")
    print(f" - num. of cpu: {num_cpu}")
    print(f" - num. of evolution: {num_evolution}")
    print(f" - num. of iteration: {num_iteration}")
    print(f" - population size:   {size_population}")
    print(f" - selection rate:    {rate_selection}")
    print(f" - crossover rate:    {rate_crossover}")
    print(f" - mutation rate:     {rate_mutation}")

    print("2. genetic-algorithm")
    print(" - serial")
    serial_start = time.time()
    _ = genetic_algorithm(param, is_parallel=False, is_show_msg=False)
    print(" - {0} sec.".format(round(time.time() - serial_start, 4)))
    # - graph: att48_d
    #  - serial
    #   - final evo-num:  1
    #   - final evo-fit:  49926
    #   - final evo-path: [44, 34, 25, 3, 9, 23, 41, 1, 28, 4, 47, 38, 31, 20, 24, 22, 40, 21, 2, 0, 15, 33, 13, 12, 46, 32, 7, 8, 10, 11, 35, 16, 27, 30, 39, 14, 18, 29, 19, 45, 43, 17, 37, 6, 5, 26, 42, 36]
    #  - 396.6317 sec.

    print(" - parallel")
    parallel_start = time.time()
    _ = genetic_algorithm(param, is_parallel=True, is_show_msg=False)
    print(" - {0} sec.".format(round(time.time() - parallel_start, 4)))
    # - graph: att48_d
    #  - parallel
    #   - final evo-num:  1
    #   - final evo-fit:  1564
    #   - final evo-path: [1, 9, 4, 10, 2, 14, 13, 16, 5, 7, 6, 0, 12, 3, 8, 11, 15]
    #  - 26.3566 sec.

    print("\n{0} sec.".format(round(time.time() - start_time, 4)))


if __name__ == "__main__":
    main_function()
