from ioh import get_problem, ProblemClass
from ioh import logger
import sys
import numpy as np
import time
import random
# genetic algorithm search of the one max optimization problem
import sys

from numpy.random import randint
from numpy.random import rand

random.seed(42)

# Declaration of problems to be tested.
# We obtain an interface of the OneMax problem here.
dimension = 50  # used to be 50

om = get_problem(1, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)
# We know the optimum of onemax
optimum = dimension

# Create default logger compatible with IOHanalyzer
# `root` indicates where the output files are stored.
# `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
l = logger.Analyzer(root="data",
                    folder_name="run",
                    algorithm_name="genetic_algorithm",
                    algorithm_info="Test of Lena and Emma")
# define the total iterations
n_iter = 5000
# bits
n_bits = 50
# define the population size
n_pop = 10
# mutation rate
r_mut = 1.0 / float(n_bits)
# crossover rate
r_cross = 1 - r_mut
om.attach_logger(l)

def prop_selection(parent, parent_f, k=3):
    list_parents = []
    for i in range(len(parent_f)):
        fitness = parent_f[i]
        prob_parent = int(fitness) * [i]
        list_parents += prob_parent
    rand_parent = len(list_parents) - 1
    rand_digit = random.randint(0, rand_parent)
    parent1 = parent[list_parents[rand_digit]]
    return parent1


# tournament selection
def tour_selection(parent, parent_f, k=5):
    score = 0
    indiv = 0
    tour = random.sample(range(0, len(parent) - 1), 5)
    for i in tour:
        if parent_f[i] > score:
            score = parent_f[i]
            indiv = i
    return parent[indiv]


# crossover two parents to create two children
def crossover(p1, p2, crossover_probability, n_points=2):
    new_p1 = p1.copy()
    new_p2 = p2.copy()
    # print(new_p1, new_p2)
    # pos = [random.randint(0, len(p1)) for n in range(n_points)]
    pos = []
    while len(pos) != n_points:
        new_digit = random.randint(1, (len(p1) - 2))
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
    return [p1, p2]


def uniform_crossover(p1, p2):
    # chance of crossover is 50% (has to be between 0 and 1)
    new_p1 = p1.copy()
    new_p2 = p2.copy()
    for i in range(len(p1)):
        chance = random.choice([0, 1])
        if chance == 0:
            bit_p1 = new_p2[i]
            bit_p2 = new_p1[i]
            new_p1[i] = bit_p1
            new_p2[i] = bit_p2
    p1 = new_p1
    p2 = new_p2
    return [p1, p2]


# mutation operator
def mutation(p, mutation_rate):
    for bit in range(len(p)):
        rand_digit = random.uniform(0, 1)
        if rand_digit < mutation_rate:
            p[bit] = 1 - p[bit]


# genetic algorithm
def genetic_algorithm(func, n_bits, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring
    pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best, best_eval = 0, func(pop[0])
    # enumerate generations
    for gen in range(n_iter):
        # evaluate all candidates in the population
        scores = [func(c) for c in pop]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))
        # select parents
        selected = [tour_selection(pop, scores) for _ in range(n_pop)]
        # print(selected)
        # sys.exit()
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i + 1]
            # crossover and mutation
            # for c in crossover(p1, p2, r_cross):
            for c in uniform_crossover(p1, p2):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    func.reset()
    return [best, best_eval]


def main():


    # We run the algorithm 20 independent times.
    for _ in range(20):
        best, score = genetic_algorithm(om)
        print('Done!')
        print('f(%s) = %f' % (best, score))


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("The program takes %s seconds" % (end - start))
