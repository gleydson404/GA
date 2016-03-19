import numpy as np
from operator import itemgetter, add
from random import random, randint, uniform

# weights = np.array([0, 0], [0, 1], [1, 0], [1, 1])
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

RIGHT_ANSWER = [0, 1, 1, 1]
POP_LENGTH = 300
NUMBER_GENERATIONS = 300


def evaluate(value):
    if value <= 2:
        return 0
    else:
        return 1


def feed_forward(inputs, weights):
    sum = reduce(add, inputs*weights)
    return evaluate(sum)




def generate_float_individual(length):
    return np.random.uniform(-20, 20, length)


def generate_float_population(count, length):
    return [generate_float_individual(length) for item in range(count)]


def fitness_evaluation_individual(individual, right_answer):
    outs = []
    for item in inputs:
        outs.append(feed_forward(item, individual))

    out_vec = [np.abs(x - y) for x, y in zip(right_answer, outs)]
    return reduce(add, out_vec)


def fitness_evaluation_population(population):
    evaluated_individuals = []
    for individual in population:
        evaluated_individuals.append((fitness_evaluation_individual(individual, RIGHT_ANSWER), individual))

    return evaluated_individuals


def grade(population, target):
    summed = reduce(add, (fitness_evaluation_individual(x, target) for x in population), 0)
    return summed / (len(population) * 1.0)


def evolve(population, percent_winners=0.2, random_select=0.05, mutate=0.01):
    evaluated_individuals = sorted(fitness_evaluation_population(population), key=itemgetter(0))
    selecteds = [x[1] for x in evaluated_individuals]  # remove fitness value, leaving just the individual
    qtd_winners = int((len(selecteds) * percent_winners))
    selecteds_genes = selecteds[:qtd_winners]

    for individual in selecteds[qtd_winners:]:
        if random_select > random():
            selecteds_genes.append(individual)

    for individual in selecteds_genes:
        if mutate > random():
            position_to_mutate = randint(0, len(individual)-1)
            individual[position_to_mutate] = uniform(min(individual), max(individual))

    parents_length = len(selecteds_genes)
    children = []

    while len(children) < (len(population) - parents_length):
        father = randint(0, parents_length - 1)
        mother = randint(0, parents_length - 1)

        if father != mother:
            father = selecteds_genes[father]
            mother = selecteds_genes[mother]

            half = len(father)/2
            child = np.concatenate((father[:half], mother[half:]))
            children.append(child)

    selecteds_genes.extend(children)

    return selecteds_genes


if __name__ == "__main__":
    population = generate_float_population(100, 2)

    fitness_history = [grade(population, RIGHT_ANSWER)]
    number_generation = 1
    for item in range(NUMBER_GENERATIONS):

        population = evolve(population)
        fitness_history.append(grade(population, RIGHT_ANSWER))

    for datum in fitness_history:
        print ("Geracao -", number_generation)
        number_generation += 1
        print datum

    print population

