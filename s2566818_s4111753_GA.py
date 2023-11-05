from ioh import get_problem, ProblemClass
from ioh import logger
import sys
import numpy as np
import time
import random

# Declaration of problems to be tested.
# We obtain an interface of the OneMax problem here.
dimension = 10  # used to be 50

"""
1 (fid) : The function ID of the problem in the problem suite. OneMax is 1 defined within the PBO class. 2 would correspond to another problem.
dimension : The dimension of the problem, which we have set to 50.
instance: In benchmarking libraries, problems often have multiple instances. These instances may vary slightly (e.g., different random noise, shifts, etc.) 
            to allow algorithms to be tested on a variety of conditions.
om(x) return the fitness value of 'x'
"""
om = get_problem(1, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)
# We know the optimum of onemax
optimum = dimension

# Create default logger compatible with IOHanalyzer
# `root` indicates where the output files are stored.
# `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
l = logger.Analyzer(root="data", 
    folder_name="run", 
    algorithm_name="genetic_algorithm", 
    algorithm_info="The lab session of the evolutionary algorithm course in LIACS")

om.attach_logger(l)


# Parameters setting
pop_size = 10
tournament_k = 5
mutation_rate = 1 / dimension  # minimum 1/l of bitstring
crossover_probability = 1 - mutation_rate  # to be tested


# Uniform Crossover
def crossover(p1, p2):
    rand_digit = random.random()
    if rand_digit < crossover_probability:
        # performing crossover
        # one point crossover
        cross_pos = random.randint(0, len(p1))
        copy_p1 = list(p1.copy())
        copy_p2 = list(p2.copy())
        new_p1 = copy_p1[:cross_pos] + copy_p2[cross_pos:]
        new_p2 = copy_p2[:cross_pos] + copy_p1[cross_pos:]
        return new_p1, new_p2
    return p1, p2

# Standard bit mutation using mutation rate p
def mutation(p):
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
def mating_seletion(parent, parent_f) :
    # print(parent)
    # print(parent_f)
    # total = sum(parent_f)
    #TODO: maybe add a constant later
    list_parents = []
    for i in range(len(parent_f)):
        fitness = parent_f[i]
        prob_parent = int(fitness) * [i]
        list_parents += prob_parent
    choice = False
    while choice != True:
        rand_digit = random.randint(0, len(list_parents))
        # print(list_parents)
        # print(list_parents[rand_digit])
        # print(parent)
        parent1 = parent[list_parents[rand_digit]]
        second_digit = random.randint(0, len(list_parents))
        # print(list_parents[second_digit])
        # print(parent[8])
        parent2 = parent[list_parents[second_digit]]
        if list_parents[rand_digit] != list_parents[second_digit]:
            choice = True
    return [parent1, parent2]
    # selected_p = []
    # print(parent)
    # for i in parent_f:
    #     if len(selected_p) == 2:
    #         return selected_p
    #     random_digit = random.random()
    #     prob = parent_f[i] / total
    #     if i == 0:
    #         if random_digit < prob:
    #             select = parent[i]
    #             selected_p.append(select)
    #     else:
    #         if random_digit > parent_f[i-1]/total and random_digit < prob:
    #             select = parent[i]
    #             selected_p.append(select)
    #


def genetic_algorithm(func, budget = None):
    
    # budget of each run: 10000
    if budget is None:
        budget = 10000
    
    # f_opt : Optimal function value.
    # x_opt : Optimal solution.
    f_opt = sys.float_info.min
    x_opt = None
    
    # parent : A list that holds the binary strings representing potential solutions or individuals in the current population.
    # parent_f : A list that holds the fitness values corresponding to each individual in the parent list.
    parent = []
    parent_f = []
    for i in range(pop_size):

        # Initialization
        parent.append(np.random.randint(2, size=func.meta_data.n_variables))
        parent_f.append(func(parent[i]))
        budget = budget - 1

    while (f_opt < optimum and budget > 0):
        # crossover(parent[0], parent[1])
        # mutation(parent[0])
        mating_seletion(parent, parent_f)
        sys.exit()
        # Perform mating selection, crossover, and mutation to generate offspring
        offspring = ...
        
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
