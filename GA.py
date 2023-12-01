# import required libraries
import numpy as np
# you need to install this package `ioh`. Please see documentations here: 
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
import sys
import time
import random
from numpy.random import randint


# amount of dimensions
dimension = 50
# define the population size
pop_size = 10
# mutation rate
mutation_rate = 1.0 / float(dimension)
# crossover rate
crossover_probability = 1 - mutation_rate
np.random.seed(42)


def prop_selection(parent, parent_f):
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


def s2566818_s4111753_GA(problem):
    budget = 5000
    f_opt = sys.float_info.min

    # create initial population
    pop = []
    for p in range(pop_size):
        pop.append(randint(0, 2, dimension).tolist())
        # we could also subtract from the budget here, but the problem isn't being called

    while problem.state.evaluations < budget:
        # evaluate candidates of the populations
        pop_f = []
        for fit in pop:
            pop_f.append(problem(fit))

        # check for new best solution and select parents
        offspring = []
        for i in range(pop_size):
            offspring.append(tour_selection(pop, pop_f))
            if pop_f[i] > f_opt:
                x_opt, f_opt = pop[i], pop_f[i]
                print(f"{pop[i]}, scores: {pop_f[i]}")

        # create the next generation
        offspring_c = list()
        for i in range(0, pop_size, 2):
            # for c in crossover(p1, p2, crossover_probability):
            for c in uniform_crossover(offspring[i], offspring[i + 1]):
                # mutation
                mutation(c, mutation_rate)

                # store for next generation
                offspring_c.append(c)

        # replace population
        pop = offspring_c
    # no return value needed


def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    F18, _logger = create_problem(18)
    for run in range(20): 
        s2566818_s4111753_GA(F18)
        F18.reset()
    _logger.close()

    F19, _logger = create_problem(19)
    for run in range(20): 
        s2566818_s4111753_GA(F19)
        F19.reset()
    _logger.close()