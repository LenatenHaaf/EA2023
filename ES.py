import numpy as np
from ioh import get_problem, logger, ProblemClass
import sys
import math
import random

budget = 5000
dimension = 50
mu = 5
lambda_ = 10
t0 = 1/math.sqrt(mu)
t_prime = 1/math.sqrt(2 * mu)
t = 1 / math.sqrt(2 * math.sqrt(mu))

np.random.seed(42)

#  maybe implement correlated mutation later if time is left
def mutation_common(parent, sigma):
    new_parent = []

    # mutate sigma
    mut_sigma = np.random.normal(0, 1)
    sigma_prime = sigma * np.exp(t0 * mut_sigma)

    # mutate parent
    for p in parent:
        mut_p = np.random.normal(0, 1)
        p_prime = p + sigma_prime * mut_p
        new_parent.append(p_prime)
    return new_parent, sigma_prime


def one_step_mutation(parent, sigma_ind):
    g = np.random.normal(0, 1)
    sigma_ind_prime = []
    new_parent = []

    # mutate sigma
    for s in range(len(sigma_ind)):
        sigma_prime = sigma_ind[s] * np.exp((t_prime * g) + (t * np.random.normal(0, 1)))
        sigma_ind_prime.append(sigma_prime)

    for p in range(len(parent)):
        mut_p = np.random.normal(0, 1)
        parent_prime = parent[p] + sigma_ind_prime[p] * mut_p
        new_parent.append(parent_prime)
    return new_parent, sigma_ind_prime


def discrete_recombination(p1, p2, s1, s2):
    offspring = []
    sigma = (s1 + s2) / 2

    for i in range(len(p1)):
        chance = random.choice([0, 1])
        if chance == 0:
            offspring.append(p1[i])
        else:
            offspring.append(p2[i])
    return offspring, sigma


def intermediate_recombination(p1, p2, s1, s2):
    offspring = (p1 + p2) / 2
    sigma = (s1 + s2) / 2
    return offspring, sigma


def global_discrete_recombination(parent, parent_s):
    offspring = []
    for i in range(len(parent[0])):
        pos = random.sample([*range(0, len(parent) - 1)], 1)
        offspring.append(parent[pos[0]][i])
    if type(parent_s[0]) == int():
        sigma = sum(parent_s) / len(parent_s)
    else:
        sigma = [sum(i)/len(parent_s) for i in zip(*parent_s)]
    return offspring, sigma


def global_intermediate_recombination(parent, parent_s):
    offspring = [sum(i)/len(parent) for i in zip(*parent)]
    if type(parent_s[0]) == int():
        sigma = sum(parent_s) / len(parent_s)
    else:
        sigma = [sum(i)/len(parent_s) for i in zip(*parent_s)]
    return offspring, sigma


def s2566818_s4111753_ES(problem):
    # hint: F18 and F19 are Boolean problems. Consider how to present bitstrings as real-valued vectors in ES
    lowerbound = -10
    upperbound = 10
    budget = 5000
    parent = []
    parent_s = []
    parent_s_ind = []
    parent_f = []
    f_opt = sys.float_info.min

    for i in range(mu):
        individual = np.random.uniform(low=lowerbound, high=upperbound, size=dimension)
        parent.append(individual)
        parent_s.append(0.05 * (upperbound - lowerbound))
        temp = []
        for d in range(dimension):
            temp.append(0.05 * (upperbound - lowerbound))
        parent_s_ind.append(temp)

    for i in range(mu):
        # data needs to be normalized first
        scaled = [0 if x < 0 else 1 for x in parent[i]]
        parent_f.append(problem(scaled))
        budget -= 1
        if parent_f[i] > f_opt:
            f_opt = parent_f[i]
            x_opt = parent[i].copy()

    while problem.state.evaluations < budget:
        offspring = []
        offspring_s = []
        for i in range(lambda_):
            [p1, p2] = random.sample([*range(0, mu - 1)], 2)
            # ind, new_sigma = discrete_recombination(parent[p1], parent[p2], parent_s[p1],  parent_s[p2])
            # ind, new_sigma = intermediate_recombination(parent[p1], parent[p2], parent_s[p1],  parent_s[p2])
            ind, new_sigma = global_discrete_recombination(parent, parent_s_ind)
            # ind, new_sigma = global_intermediate_recombination(parent, parent_s_ind)
            offspring.append(ind)
            offspring_s.append(new_sigma)

        # mutation
        for i in range(len(offspring)):
            # offspring[i], offspring_s[i] = mutation_common(offspring[i], offspring_s[i])
            offspring[i], offspring_s[i] = one_step_mutation(offspring[i], offspring_s[i])

        # comma selection
        offspring_f = []
        for i in range(mu):
            # data needs to be normalized first
            scaled = [0 if x < 0 else 1 for x in offspring[i]]
            offspring_f.append(problem(scaled))

        offspring_f += parent_f
        offspring_s += parent_s
        offspring += parent

        parent = []
        parent_s = []
        parent_f = []
        rank = np.argsort(offspring_f)[::-1]
        for m in range(mu):
            parent.append(offspring[rank[m]])
            parent_f.append(offspring_f[rank[m]])
            parent_s.append(offspring_s[rank[m]])

    # print(f"Optimum: {f_opt}")
    return f_opt



def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="evolution_strategies",  # name of your algorithm
        algorithm_info="Practical assignment of Lena and Emma",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    F18, _logger = create_problem(18)
    avg = []
    for run in range(20): 
        f_opt = s2566818_s4111753_ES(F18)
        avg.append(f_opt)
        F18.reset() # it is necessary to reset the problem after each independent run
    print(round(sum(avg)/20, 2))
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder
    # avg = []
    # F19, _logger = create_problem(19)
    # for run in range(20):
    #     f_opt = s2566818_s4111753_ES(F19)
    #     avg.append(f_opt)
    #     F19.reset()
    # print(round(sum(avg) / 20, 2))
    # _logger.close()