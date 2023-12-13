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
# number of offspring
lambda_ = 20
# mutation rate
mutation_rate = 0.05 #1.0 / float(dimension) #0,01-0,1
# crossover rate
crossover_probability = 0.9 #0.9 #0,1 -0,9
np.random.seed(42)
random.seed(42)


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

def proportionalselection(parent, parent_f, c): 
    offspring = []
    probability = []
    # c = min(parent_f) - 0.01
    # c = 0.1
    for f in parent_f:
        if sum(parent_f) > 0:
            p = (f-c)/(sum(parent_f) - (c*len(parent)))
            probability.append(p)
        else:
            p = -1
            probability.append(p)

    random_number = random.random() #generate a random number between 0 and 1
    kans = 0
    for i in range(len(parent)-1):
        kans += probability[i] + probability[i+1]
        if random_number <= probability[0]:
            offspring = parent[0]
            break
        elif random_number <= kans:
            offspring = parent[i]
            break

    return offspring


# tournament selection
def tour_selection(parent, parent_f, k=5):
    score = 0
    indiv = 0
    tour = random.sample(range(0, len(parent) - 1), k)
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


def s2566818_s4111753_GA(problem, selection_method, selection_size):
    budget = 5000
    f_opt = sys.float_info.min

    # create initial population
    pop = []
    for p in range(pop_size):
        pop.append(randint(0, 2, dimension).tolist())

    while problem.state.evaluations < budget:
        # evaluate candidates of the populations
        pop_f = []
        for fit in pop:
            pop_f.append(problem(fit))

        for i in range(pop_size):
            if pop_f[i] > f_opt:
                x_opt, f_opt = pop[i], pop_f[i]

        # check for new best solution and select parents
        offspring = []
        while len(offspring) <= lambda_: # <= or < 
            if selection_method:
                offspring.append(proportionalselection(pop, pop_f, c = 0.1)) #(min(pop_f)-0.01)
            else:
                offspring.append(proportionalselection(pop, pop_f, c = 0.01))

        # create the next generation
        offspring_c = []
        for i in range(0, lambda_, 2):
            for c in crossover(offspring[i], offspring[i + 1], crossover_probability, n_points=25):
                # mutation
                mutation(c, mutation_rate)

                # store for next generation
                offspring_c.append(c)

        offspring_f = []
        for fit in offspring_c:
            offspring_f.append(problem(fit))
        
        # plus selection
        if selection_size:
            offspring_f += pop_f
            offspring_c += pop

        pop = []
        pop_f = []
        rank = np.argsort(offspring_f)[::-1]
        for m in range(pop_size):
            pop.append(offspring_c[rank[m]])
            pop_f.append(offspring_f[rank[m]])

    return f_opt


def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="finalfinal",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="GA_F18",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    F18, _logger = create_problem(18)
    opt = []
    selection_method = True
    selection_size = True
    for run in range(20): 
        f_opt = s2566818_s4111753_GA(F18, selection_method, selection_size)
        opt.append(f_opt)
        F18.reset()
    _logger.close()
    mean = round(sum(opt)/len(opt), 2)
    print(f"The mean optimal value for F18 is: {mean}")

    F19, _logger = create_problem(19)
    opt = []
    selection_method = False
    selection_size = False
    for run in range(20): 
        f_opt =s2566818_s4111753_GA(F19, selection_method, selection_size)
        opt.append(f_opt)
        F19.reset()
    _logger.close()
    mean = sum(opt)/len(opt)
    print(f"The mean optimal value for F19 is: {mean}")