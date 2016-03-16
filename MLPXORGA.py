import numpy as np
from operator import itemgetter, add
from random import random, randint, uniform

inputs = np.array([[-1, 0, 0], [-1, 0, 1], [-1, 1, 0], [-1, 1, 1]])
RIGHT_ANSWER = [0, 1, 1, 1]


# w = np.matrix(np.zeros((3, 3)))
# z = np.matrix(np.zeros((1, 3)))

W = np.matrix([[-9.8050,   -6.0907,   -7.0623], [-2.4839,   -5.3249,   -6.9537]])
Z = np.matrix([[5.7278,   12.1571,  -12.8941]])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def feed_forward(input, w, z):  # weights should be transpose
    layer1 = [-1]
    for index in xrange(len(w)-1):
        summary_l1 = input * w[:, index]
        layer1.append(sigmoid(summary_l1))

    summary_l2 = (layer1 * z)

    return sigmoid(summary_l2)


# for input in inputs:
#     print feed_foward(input, w.T, z.T)


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

for input in inputs:
    print feed_forward(input, W.T, Z.T)