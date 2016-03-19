import numpy as np
from operator import itemgetter, add
from random import random, randint, uniform

inputs = np.matrix([[-1, 0, 0], [-1, 0, 1], [-1, 1, 0], [-1, 1, 1]])
RIGHT_ANSWER = [0, 1, 1, 0]
NUMBER_GENERATIONS = 300

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

    summary_l2 = (np.matrix(layer1) * z)

    return sigmoid(summary_l2)


# for input in inputs:
#     print feed_foward(input, w.T, z.T)


def fitness_evaluation_individual(chromosome, right_answer):
    matrix = chromosome.reshape(3, 3)
    w = np.matrix(matrix[:2, :])
    z = np.matrix(matrix[2, :])
    outs = []
    for item in inputs:
        outs.append(feed_forward(item, w.T, z.T))

    out_vec = [np.abs(x - y) for x, y in zip(right_answer, outs)]

    error = reduce(add, out_vec)
    return error


def generate_float_individual(size):
    individual = np.array(np.random.uniform(-20, 20, size))
    return individual


def generate_float_population(count, size):
    return [generate_float_individual(size) for item in range(count)]


def fitness_evaluation_population(population):
    evaluated_individuals = []
    for individual in population:
        evaluated_individuals.append(
            (fitness_evaluation_individual(individual, RIGHT_ANSWER),
             individual))

    return evaluated_individuals


def evolve(population, percent_winners=0.2, random_select=0.05, mutate=0.01):
    evaluated_individuals = sorted(fitness_evaluation_population(population), key=itemgetter(0))

    selecteds = [x[1] for x in evaluated_individuals]  # remove fitness value, leaving just the individual

    qtd_winners = int((len(selecteds) * percent_winners))

    selecteds_genes = selecteds[:qtd_winners]

    size_individual = len(selecteds_genes[0]) - 1

    for individual in selecteds[qtd_winners:]:
        if random_select > random():
            selecteds_genes.append(individual)

    # implementar Roleta

    for individual in selecteds_genes:
        if mutate > random():
            pst_to_mutate = randint(0, size_individual)
            individual[pst_to_mutate] = uniform(min(individual), max(individual))

    parents_length = len(selecteds_genes)
    children = []

    while len(children) < (len(population) - parents_length):
        father = randint(0, parents_length - 1)
        mother = randint(0, parents_length - 1)

        if father != mother:
            father = selecteds_genes[father]
            mother = selecteds_genes[mother]

            fst_part_cross = randint(0, size_individual)
            # scd_part_cross = randint(fst_part_cross, size_individual)

            child_1 = np.hstack((father[:fst_part_cross], mother[fst_part_cross:]))
            child_2 = np.hstack((mother[:fst_part_cross], father[fst_part_cross:]))

            children.append(child_1)
            children.append(child_2)

    selecteds_genes.extend(children)

    return selecteds_genes


# chromosome = np.matrix([-3.20601877, -16.79708469, -16.56907027, -15.70006945, -11.85963456, -10.14149513,   4.89624683, -15.4298402 ,  11.32024019])
# matrix = chromosome.reshape(3, 3)
# w = np.matrix(matrix[:2, :])
# z = np.matrix(matrix[2, :])
# for input in inputs:
#     print feed_forward(input, W.T, Z.T)

# print fitness_evaluation_individual(W.T, Z.T, RIGHT_ANSWER)

# if __name__ == "__main__":
#     population = generate_float_population(100, 9)
#
#     # print population
#     fitness_history = []
#     number_generation = 1
#     for item in range(NUMBER_GENERATIONS):
#
#         population = evolve(population)
#         fitness_history.append(population[0])  # best solution at the moment
#
#     for datum in fitness_history:
#         print ("Geracao -", number_generation)
#         number_generation += 1
#         print datum
#
#     print population