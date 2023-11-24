from ioh import get_problem, ProblemClass
from ioh import logger
import sys
import numpy as np
import time
import random

random.seed(42)

# Declaration of problems to be tested.
# We obtain an interface of the OneMax problem here.
dimension = 100  # used to be 50

"""
1 (fid) : The function ID of the problem in the problem suite. OneMax is 1 defined within the PBO class. 2 would correspond to another problem.
dimension : The dimension of the problem, which we have set to 50.
instance: In benchmarking libraries, problems often have multiple instances. These instances may vary slightly (e.g., different random noise, shifts, etc.) 
            to allow algorithms to be tested on a variety of conditions.
om(x) return the fitness value of 'x'
"""
om = get_problem(19, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)
# We know the optimum of onemax
optimum = dimension

# Create default logger compatible with IOHanalyzer
# `root` indicates where the output files are stored.
# `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
l = logger.Analyzer(root="data", 
    folder_name="run", 
    algorithm_name="genetic_algorithm", 
    algorithm_info="Test of Lena and Emma")

om.attach_logger(l)


# Parameters setting
pop_size = 10
tournament_k = 5
mutation_rate = 1 / dimension  # minimum 1/l of bitstring
crossover_probability = 1 - mutation_rate  # to be tested


# one-point Crossover
def crossover(p1, p2, n_points=2):
    # n-point crossover
    new_p1 = p1.copy()
    new_p2 = p2.copy()
    # print(new_p1, new_p2)
    # pos = [random.randint(0, len(p1)) for n in range(n_points)]
    pos = []
    while len(pos) != n_points:
        new_digit = random.randint(0, (len(p1)-1))
        if new_digit not in pos:
            pos.append(new_digit)
    rand_digit = random.random()
    if rand_digit < crossover_probability:
        for i in pos:
            # performing N-point crossover
            # cross_pos = random.randint(0, len(p1))
            copy_p1 = list(new_p1.copy())
            copy_p2 = list(new_p2.copy())
            new_p1 = copy_p1[:i] + copy_p2[i:]
            new_p2 = copy_p2[:i] + copy_p1[i:]
    p1 = new_p1
    p2 = new_p2
    return p1, p2


def uniform_crossover(p1, p2):
    # chance of crossover is 50% (has to be between 0 and 1)
    new_p1 = p1.copy()
    new_p2 = p2.copy()
    for i in range(len(p1)):
        chance = random.random()
        if chance < 0.5:
            bit_p1 = new_p2[i]
            bit_p2 = new_p1[i]
            new_p1[i] = bit_p1
            new_p2[i] = bit_p2
    return new_p1, new_p2


# Standard bit mutation using mutation rate p
def mutation(p):
    # print("mutation func")
    for bit in range(len(p)):
        rand_digit = random.random()
        if rand_digit < mutation_rate:
            # print("Its about flipping time")
            if p[bit] == 0:
                p[bit] = 1
            else:
                p[bit] = 0
    return p

# Using the Fitness proportional selection
def mating_selection(parent, parent_f, top_n=10) :
    #TODO: maybe add a constant later
    list_parents = []
    for i in range(len(parent_f)):
        fitness = parent_f[i]
        prob_parent = int(fitness) * [i]
        list_parents += prob_parent

    indexes = []
    parents_res = []
    choice = False
    for i in range(top_n):
        rand_parent = len(list_parents) - 1
        rand_digit = random.randint(0, rand_parent)
        # print(rand_digit)
        parent1 = parent[list_parents[rand_digit]]
        parents_res.append(parent1)
        # if rand_digit not in parents_res:
        #     indexes.append(parents_res)

        # second_digit = random.randint(0, len(list_parents))
        # parent2 = parent[list_parents[second_digit]]
        # if list_parents[rand_digit] != list_parents[second_digit]:
        #     choice = True
    return parents_res


def genetic_algorithm(func, budget=5000):
    # budget of each run: 5.000
    if budget is None:
        budget = 5000
    
    # f_opt : Optimal function value
    # x_opt : Optimal solution
    f_opt = sys.float_info.min
    x_opt = None
    
    # parent: A list that holds the binary strings representing potential solutions or individuals in the current
    # population. parent_f: A list that holds the fitness values corresponding to each individual in the parent list.
    parent = []
    parent_f = []
    for i in range(pop_size):
        # Initialization
        parent.append(np.random.randint(2, size=func.meta_data.n_variables))
        # Evaluation
        parent_f.append(func(parent[i]))
        budget = budget - 1

    while f_opt < optimum and budget > 0:
        # print(parent)
        offspring = mating_selection(parent, parent_f)
        # print(offspring)
        chunks = [offspring[i:i + 2] for i in range(0, len(offspring), 2)]
        offspring_co = []
        for c in chunks:
            p1, p2 = crossover(c[0], c[1])
            offspring_co.append(p1)
            offspring_co.append(p2)

        for p in offspring_co:
            pmut = mutation(p)

        parent = offspring_co.copy()
        for i in range(pop_size):
            budget -= 1
            val = func(parent[i])
            parent_f[i] = val
            if val > f_opt:
                f_opt = val
                x_opt = parent[i].copy()
            if f_opt >= optimum:
                break
        # sys.exit()

        # Perform mating selection, crossover, and mutation to generate offspring
        
    # ioh function, to reset the recording status of the function.
    func.reset()
    print(f_opt,x_opt)
    return f_opt, x_opt

def main():
    # We run the algorithm 20 independent times.
    for _ in range(20):
        genetic_algorithm(om)

if __name__ == '__main__':
  start = time.time()
  main()
  end = time.time()
  print("The program takes %s seconds" % (end-start))
