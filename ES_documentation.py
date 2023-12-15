import numpy as np
from ioh import get_problem, logger, ProblemClass
import sys
import math
import random

budget = 5000
dimension = 50

random.seed(42)
np.random.seed(42)


def mutation_common(parent, sigma, t0):
    """
    Function which performs mutation on 1 sigma with regard to the t0. Clipping is used to force the values to be in
    between a certain range.

    Parameters:
        parent (list): the values from the parent converted to a bit string.
        sigma (list): the values from the parent containing the sigma's.
        t0 (float): the calculated t0 value.

    Returns:
        new_parent (list): the new (mutated) parent.
        sigma_prime (float): the calculated sigma prime.
    """
    new_parent = []

    # mutate sigma
    mut_sigma = np.random.normal(0, sigma) #1
    sigma_prime = sigma * np.exp(t0 * mut_sigma)

    # mutate parent
    for p in parent:
        mut_p = np.random.normal(0, 1)
        p_prime = p + sigma_prime * mut_p
        if p_prime > 10.0:
            p_prime = 10.0
        elif p_prime < -10.0:
            p_prime = -10.0
        new_parent.append(p_prime)
    return new_parent, sigma_prime


def one_step_mutation(parent, sigma_ind, t_prime, t):
    """
        Function which performs mutation on n sigma with regard to the t and t_prime. Clipping is used to force the values to be in
        between a certain range.

        Parameters:
            parent (list): the values from the parent converted to a bit string.
            sigma_ind (list): the values from the parent containing the sigma's.
            t_prime (float): the calculated t_prime value.
            t (float): the calculated t value.

        Returns:
            new_parent (list): the new (mutated) parent.
            sigma_prime (float): the calculated sigma prime.
    """
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
        if parent_prime > 10.0:
            parent_prime = 10.0
        elif parent_prime < -10.0:
            parent_prime = -10.0
        new_parent.append(parent_prime)
    return new_parent, sigma_ind_prime


def discrete_recombination(p1, p2, s1, s2):
    """
    Function which performs the discrete recombination for two parents.

    Parameters:
        p1 (numpy.ndarray): real values of parent 1
        p2 (numpy.ndarray): real values of parent 2
        s1 (list): individual sigma values for parent 1
        s2 (list): individual sigma values for parent 2

    Returns:
        offspring (list): created offspring made by discrete recombination.
        sigma (list): list containing sigma value(s) of the individual.
    """
    offspring = []
    if isinstance(s1, float):
        sigma = sum([s1, s2]) / 2
    else:
        sigma = [sum(i)/2 for i in zip(s1, s2)]

    for i in range(len(p1)):
        chance = random.choice([0, 1])
        if chance == 0:
            offspring.append(p1[i])
        else:
            offspring.append(p2[i])
    return offspring, sigma


def intermediate_recombination(p1, p2, s1, s2):
    """
        Function which performs the intermediate recombination for two parents.

        Parameters:
            p1 (numpy.ndarray): real values of parent 1
            p2 (numpy.ndarray): real values of parent 2
            s1 (list): individual sigma values for parent 1
            s2 (list): individual sigma values for parent 2

        Returns:
            offspring (list): created offspring made by discrete recombination.
            sigma (list): list containing sigma value(s) of the individual.
        """
    offspring = [sum(i)/2 for i in zip(p1, p2)]
    if isinstance(s1, float):
        sigma = sum([s1, s2]) / 2
    else:
        sigma = [sum(i)/2 for i in zip(s1, s2)]
    return offspring, sigma


def global_discrete_recombination(parent, parent_s):
    """
        Function which performs the global discrete recombination for two parents.

        Parameters:
            parent (list): containing arrays of the real values of the parents.
            parent_s (list):  individual sigma values for the parents.

        Returns:
            offspring (list): created offspring made by discrete recombination.
            sigma (list): list containing sigma value(s) of the individual.
    """
    offspring = []
    for i in range(len(parent[0])):
        pos = random.sample([*range(0, len(parent) - 1)], 1)
        offspring.append(parent[pos[0]][i])
    if isinstance(parent_s[0], float):
        sigma = sum(parent_s) / len(parent_s)
    else:
        sigma = [sum(i)/len(parent_s) for i in zip(*parent_s)]
    return offspring, sigma


def global_intermediate_recombination(parent, parent_s):
    """
        Function which performs the global intermediate recombination for two parents.

        Parameters:
            parent (list): containing arrays of the real values of the parents.
            parent_s (list):  individual sigma values for the parents.

        Returns:
            offspring (list): created offspring made by discrete recombination.
            sigma (list): list containing sigma value(s) of the individual.
    """
    offspring = [sum(i)/len(parent) for i in zip(*parent)]
    if isinstance(parent_s[0], float):
        sigma = sum(parent_s) / len(parent_s)
    else:
        sigma = [sum(i)/len(parent_s) for i in zip(*parent_s)]
    return offspring, sigma


def s2566818_s4111753_ES(problem, crossover, mu, lambda_):
    """
    Function which performs the overall code of the ES. Function is calling all the operations, for example the
    recombination and mutation.
    It keeps track of the budget as well and performs the initialization and the evaluation of the ES.

    Parameters:
        problem (ioh.iohcpp.problem.LABS): the problem imported by IOH (is either F18 or F19).
        crossover (bool): boolean which decides the type of recombination used with regard to the problem loaded by IOH.
        mu (int): amount of parents that need to be created.
        lambda_ (int): amount of offspring that needs to be produced.

    Returns:
        f_opt (float): value which contains the most optimal result.
    """
    lowerbound = -10
    upperbound = 10
    budget = 5000
    parent = []
    parent_s = []
    parent_s_ind = []
    parent_f = []
    f_opt = sys.float_info.min
    t0 = 1/math.sqrt(mu)
    t_prime = 1/math.sqrt(2 * mu)
    t = 1 / math.sqrt(2 * math.sqrt(mu))

    for i in range(mu):
        individual = np.random.uniform(low=lowerbound, high=upperbound, size=dimension)
        parent.append(individual)
        parent_s.append(0.05 * (upperbound - lowerbound))
        temp = []
        for d in range(dimension):
            temp.append(0.05 * (upperbound - lowerbound))
        parent_s_ind.append(temp)

    for i in range(mu):
        scaled = [0 if x < 0 else 1 for x in parent[i]]
        parent_f.append(problem(scaled))
        if parent_f[i] > f_opt:
            f_opt = parent_f[i]
            x_opt = parent[i].copy()

    while problem.state.evaluations < budget:
        offspring = []
        offspring_s = []
        for i in range(lambda_):
            if crossover:
                [p1, p2] = random.sample([*range(0, mu - 1)], 2)
                ind, new_sigma = discrete_recombination(parent[p1], parent[p2], parent_s_ind[p1],  parent_s_ind[p2])
            else:
                ind, new_sigma = global_discrete_recombination(parent, parent_s_ind)
            offspring.append(ind)
            offspring_s.append(new_sigma)

        # mutation
        for i in range(len(offspring)):
            offspring[i], offspring_s[i] = one_step_mutation(offspring[i], offspring_s[i], t_prime, t)

        # comma selection
        offspring_f = []
        for i in range(lambda_):
            scaled = [0 if x < 0 else 1 for x in offspring[i]]
            offspring_f.append(problem(scaled))

        #plus selection
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
            if parent_f[m] > f_opt:
                f_opt = parent_f[m]
                x_opt = parent[m].copy()
            parent_s.append(offspring_s[rank[m]])

    return f_opt



def create_problem(fid: int):
    """
        Function which creates the 'problem' and uses it to get information from IOH and calling IOH.

        Parameters:
            fid (int): type of problem (18 or 19)

        Returns:
            problem (ioh.iohcpp.problem.LABS): the problem imported by IOH (is either F18 or F19).
            l (ioh.iohcpp.logger.Analyzer): the logger to observe all evaluations in the run.

    """
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",  # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="ES_F18_F19",  # name of your algorithm
        algorithm_info="Practical assignment of Lena and Emma",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


if __name__ == "__main__":
    F18, _logger = create_problem(18)
    avg = []
    crossover = True
    for run in range(20): 
        f_opt = s2566818_s4111753_ES(F18, crossover, mu=20, lambda_=30)
        avg.append(f_opt)
        F18.reset() # it is necessary to reset the problem after each independent run
    _logger.close() # after all runs, it is necessary to close the logger to make sure all data are written to the folder
    mean = round(sum(avg)/len(avg), 2)
    print(f"The mean optimal value for F18 is: {mean}")

    avg = []
    crossover = False
    F19, _logger = create_problem(19)
    for run in range(20):
        f_opt = s2566818_s4111753_ES(F19, crossover, mu=20, lambda_=26)
        avg.append(f_opt)
        F19.reset()
    _logger.close()
    # print(f"F19 avg: {avg}")
    mean = round(sum(avg)/len(avg), 2)
    print(f"The mean optimal value for F19 is: {mean}")
