# genetic algorithm search for continuous function optimization

# Should we use logit or should we use 0-1 w/ normalization?

from numpy.random import randint
from numpy.random import rand
from env import MultiAgentEnv
from params import get_params
import numpy as np
import math

# # objective function
# def objective(x):
#     return x[0] ** 2.0 + x[1] ** 2.0

def to_pop_format(x):
    x = np.asarray(x).reshape([4, 3])
    x /= x.sum(axis=0)
    x = x.T
    return x

# the function that we want to maximize
def objective(x):
    x = np.asarray(x).reshape([4, 3])
    x /= x.sum(axis=0)
    return - env.get_reward(x, env_type)

# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
    decoded = list()
    largest = 2 ** n_bits
    for i in range(len(bounds)):
        # extract the substring
        start, end = i * n_bits, (i * n_bits) + n_bits
        substring = bitstring[start:end]
        # convert bitstring to a string of chars
        chars = ''.join([str(s) for s in substring])
        # convert string to integer
        integer = int(chars, 2)
        # scale integer to desired range
        value = bounds[i][0] + (integer / largest) * (bounds[i][1] - bounds[i][0])
        # store
        decoded.append(value)
    return decoded


# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k - 1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1) - 2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]


#move this to evo2, so that it works for continuous data
def create_offspring(first_parent, sec_parent, crossover_pt, offspring_number):
    """
    Creates an offspring from 2 parents. It performs the crossover
    according the following rule:
    p_new = first_parent[crossover_pt] + beta * (first_parent[crossover_pt] - sec_parent[crossover_pt])
    offspring = [first_parent[:crossover_pt], p_new, sec_parent[crossover_pt + 1:]
    where beta is a random number between 0 and 1, and can be either positive or negative
    depending on if it's the first or second offspring
    :param first_parent: first parent's chromosome
    :param sec_parent: second parent's chromosome
    :param crossover_pt: point(s) at which to perform the crossover
    :param offspring_number: whether it's the first or second offspring from a pair of parents.

    :return: the resulting offspring.
    """

    beta = (
        np.random.rand(1)[0]
        if offspring_number == "first"
        else -np.random.rand(1)[0]
    )

    p_new = first_parent[crossover_pt] - beta * (
            first_parent[crossover_pt] - sec_parent[crossover_pt]
    )

    return np.hstack(
        (first_parent[:crossover_pt], p_new, sec_parent[crossover_pt + 1:])
    )


def mutate_population(population, n_mutations, input_limits):
    """
    Mutates the population by randomizing specific positions of the
    population individuals.
    :param population: the population at a given iteration
    :param n_mutations: number of mutations to be performed.
    :param input_limits: tuple containing the minimum and maximum allowed
     values of the problem space.

    :return: the mutated population
    """

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

# genetic algorithm
def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring
    pop = [randint(0, 2, n_bits * len(bounds)).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best, best_eval = 0, objective(decode(bounds, n_bits, pop[0]))
    # enumerate generations
    for gen in range(n_iter):
        # evaluate all candidates in the population
        scores = [objective(decode(bounds, n_bits, c)) for c in pop]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen, to_pop_format(pop[i], bounds, n_bits), scores[i]))
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i + 1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children

        #
    return [best, best_eval]

def evolve_one_gen(objective, bounds, n_bits, r_cross, r_mut, pop):
    n_pop = pop.shape[0]
    decode_pop = [decode(bounds, n_bits, c) for c in pop]

    # evaluate all candidates in the population
    scores = [objective(c) for c in pop]
    # select parents
    selected = [selection(decode_pop, scores) for _ in range(n_pop)]
    # create the next generation
    children = list()
    for i in range(0, n_pop, 2):
        # get selected parents in pairs
        p1, p2 = selected[i], selected[i + 1]
        # crossover and mutation
        for c in crossover(p1, p2, r_cross):
            # mutation
            mutation(c, r_mut)
            # store for next generation
            # TODO: next gen should be composed of normalized real number
            child = to_pop_format(c, bounds, n_bits)
            children.append(child)

    return children


# define range for input
bounds = np.array([[0, 1]]*12)
# define the total iterations
n_iter = 100
# bits per variable
n_bits = 16
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / (float(n_bits) * len(bounds))

selection_rate = 0.5
mutation_rate = 0.1

n_genes = 2  # number of variables
pop_size = 100  # population size
input_limits = np.array([[-5, 5]]*2)
pop_keep = math.floor(selection_rate * pop_size)  # number of individuals to keep on each iteration

n_matings = math.floor((pop_size - pop_keep) / 2)  # number of crossovers to perform
n_mutations = math.ceil((pop_size - 1) * n_genes * mutation_rate)  # number o mutations to perform

if __name__ == '__main__':
    params = get_params()
    env = MultiAgentEnv(n_num_grids=params['env_grid_num'],
                            n_num_agents=params['n_agent_types'],
                            n_env_types=params['n_env_types'])

    env_type = [0, 1, 2, 3]

    # perform the genetic algorithm search
    best, score = genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
    print('Done!')
    decoded = decode(bounds, n_bits, best)
    print('f(%s) = %f' % (decoded, score))