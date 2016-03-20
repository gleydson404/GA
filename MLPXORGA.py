import numpy as np
from operator import itemgetter, add
from random import random, randint, uniform
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

inputs = np.matrix([[-1, 0, 0], [-1, 0, 1], [-1, 1, 0], [-1, 1, 1]])
RIGHT_ANSWER = [0, 1, 1, 0]
NUMBER_GENERATIONS = 100


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def feed_forward(input, w, z):  # weights should be transpose
    layer1 = [-1]
    for index in xrange(len(w)-1):
        summary_l1 = input * w[:, index]
        layer1.append(sigmoid(summary_l1))

    summary_l2 = (np.matrix(layer1) * z)

    return sigmoid(summary_l2)


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
    return np.array(np.random.uniform(-20, 20, size))


def generate_float_population(count, size):
    # return [generate_float_individual(size) for item in range(count)]
    return map(generate_float_individual, [size] * count)


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

            child_1 = np.hstack((father[:fst_part_cross], mother[fst_part_cross:]))
            child_2 = np.hstack((mother[:fst_part_cross], father[fst_part_cross:]))

            children.append(child_1)
            children.append(child_2)

    selecteds_genes.extend(children)

    return selecteds_genes


if __name__ == "__main__":
    population = generate_float_population(100, 9)

    fitness_history = []
    number_generation = 1
    number_generation_vec = []
    mean_fitness = []
    for item in range(NUMBER_GENERATIONS):

        population = evolve(population)
        fitness_history.append(fitness_evaluation_individual(population[0], RIGHT_ANSWER).flat[0])  # best solution at the moment
        number_generation_vec.append(number_generation)
        mean_fitness.append(np.mean([fitness_evaluation_individual(individual, RIGHT_ANSWER) for individual in population]))
        number_generation += 1

    plot_lines = []
    plt.plot(number_generation_vec, fitness_history)
    plt.plot(number_generation_vec, mean_fitness)
    blue_line = mlines.Line2D([], [], color='blue')
    green_line = mlines.Line2D([], [], color='green')
    plot_lines.append([blue_line, green_line])
    legend1 = plt.legend(plot_lines[0], ["Best Fitness", "Mean Fitness"], loc=1)
    plt.gca().add_artist(legend1)
    plt.xlabel("Generations")
    plt.title("Genetic Algorithm Gleydson")
    plt.show()

    print ("Best Solution", population[0])




