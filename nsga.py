'''
Author: Jiaheng Hu
Partially based on https://github.com/haris989/NSGA-II
'''


import math
import random
import matplotlib.pyplot as plt
from toy_env import MultiAgentEnv
from scipy.special import softmax
from params import get_params
import numpy as np

params = get_params()
env = MultiAgentEnv(n_num_grids=params['env_grid_num'],
                    n_num_agents=params['n_agent_types'],
                    n_env_types=params['n_env_types'],
                    agent_num=params['agent_num'])

n_genes = params['n_agent_types'] * params['env_grid_num']  # 12  # number of variables
pop_size = 50  # 128  # population size
input_limits = np.array([-5, 5])
mutation_rate = 0.1
n_mutations = math.ceil((pop_size - 1) * n_genes * mutation_rate)
max_gen = 100

# both are to be maximized
# First function to optimize
def function1(x):
    x = np.asarray(x).reshape([3, 4])
    x = softmax(x, axis=1)
    terrain = [4, 1, 2, 3] #first region: undeployed
    return env.get_reward(x, terrain)

# Second function to optimize
def function2(x):
    x = np.asarray(x).reshape([3, 4])
    x = softmax(x, axis=1)
    return -env.get_deployment_cost(x)

# Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

# Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

def crossover(first_parent, sec_parent, crossover_pt):
    beta = (
        np.random.rand(1)[0]
    )

    p_new1 = first_parent[crossover_pt] - beta * (
            first_parent[crossover_pt] - sec_parent[crossover_pt]
    )

    return np.hstack(
        (first_parent[:crossover_pt], p_new1, sec_parent[crossover_pt + 1:])
    ) #seems that we only need one offspring


# Function to carry out the mutation operator
def mutation(population, n_mutations=n_mutations, input_limits=input_limits):
    mutation_rows = np.random.choice(
        np.arange(1, population.shape[0]), n_mutations, replace=True
    )

    mutation_columns = np.random.choice(
        population.shape[1], n_mutations, replace=True
    )

    new_population = np.random.uniform(
        input_limits[0], input_limits[1], size=population.shape
    )

    population[mutation_rows, mutation_columns] = new_population[mutation_rows, mutation_columns]

    return population


def initialize_population(pop_size, n_genes, input_limits):
    population = np.random.uniform(
      input_limits[0], input_limits[1], size=(pop_size, n_genes)
    )

    return population


if __name__ == '__main__':
    solution = initialize_population(pop_size, n_genes, input_limits).tolist()
    gen_no = 0
    while(gen_no<max_gen):
        function1_values = [function1(solution[i])for i in range(0,pop_size)]
        function2_values = [function2(solution[i])for i in range(0,pop_size)]
        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
        if(gen_no%10==0):
            print(gen_no)
        # # Disabled for now
        # print("The best front for Generation number ",gen_no, " is")
        # for valuez in non_dominated_sorted_solution[0]:
        #     print(round(solution[valuez],3),end=" ")
        # print("\n")
        crowding_distance_values=[]
        for i in range(0,len(non_dominated_sorted_solution)):
            crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
        solution2 = solution[:]
        # Generating offsprings
        while(len(solution2)!=2*pop_size):
            a1 = random.randint(0,pop_size-1)
            b1 = random.randint(0,pop_size-1)
            xp = np.random.randint(n_genes)
            solution2.append(np.array(crossover(solution[a1], solution[b1], xp)))
        # TODO: check mutate solution2
        solution2 = mutation(np.array(solution2))
        function1_values2 = [function1(solution2[i])for i in range(0,2*pop_size)]
        function2_values2 = [function2(solution2[i])for i in range(0,2*pop_size)]
        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
        crowding_distance_values2=[]
        for i in range(0, len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
        new_solution = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if(len(new_solution)==pop_size):
                    break
            if (len(new_solution) == pop_size):
                break
        solution = [solution2[i] for i in new_solution]
        gen_no = gen_no + 1

    #Lets plot the final front now
    function1 = [i for i in function1_values]
    function2 = [j * -1 for j in function2_values]
    plt.xlabel('Coverage', fontsize=15)
    plt.ylabel('Deployment Cost', fontsize=15)
    plt.scatter(function1, function2)
    plt.show()
    x = solution2[function2_values.index(max(function2_values))].tolist()
    # print()
    print(env.get_integer(x))
    # import ipdb
    # ipdb.set_trace()