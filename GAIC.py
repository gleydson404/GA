import numpy as np
from operator import itemgetter
from random import random, randint, uniform
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import BinarytoReal

NUMBER_GENERATIONS = 1000
NUMBER_BITS = 12
NUMBER_DIMENSIONS = 30
SIZE_POPULATION = 50
HIGH_LIMIT = 100
LOW_LIMIT = -100


def x_square_30(individual):
    return np.sum([x ** 2 for x in individual])

def x_square_30x(individual):
    return np.sum([np.abs((x + 0.5) ** 2) for x in individual])


def convert_individual_to_real(individual):
    converted = []
    for item in individual:
        try:
            converted.append(BinarytoReal.convert(HIGH_LIMIT, LOW_LIMIT, NUMBER_BITS, ''.join(map(str, item))))
        except TypeError:
            print ("erro", item)
            print ("erro", individual)

    return converted

def fitness_evaluation_individual(fitness_function, individual):
    return fitness_function(convert_individual_to_real(individual))


def generate_binary_individual(size):
    return np.array([np.random.randint(2, size=NUMBER_BITS) for _ in range(size)])


def generate_float_population(count, size):
    # return [generate_float_individual(size) for item in range(count)]
    return map(generate_binary_individual, [size] * count)


def fitness_evaluation_population(population):
    evaluated_individuals = []
    for individual in population:
        evaluated_individuals.append((fitness_evaluation_individual(x_square_30, individual), individual))

    return evaluated_individuals


def evolve(population, percent_winners=0.3, random_select=0.1, mutate=0.3):
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
            pst_to_mutate_crm = randint(0, size_individual)
            pst_to_mutate_alelo = randint(0, NUMBER_BITS-1)
            individual[pst_to_mutate_crm][pst_to_mutate_alelo] = np.random.randint(2, size=1)[0]

    parents_length = len(selecteds_genes)
    children = []

    while len(children) < (len(population) - parents_length):
        father = randint(0, parents_length - 1)
        mother = randint(0, parents_length - 1)

        if father != mother:
            father = selecteds_genes[father]
            mother = selecteds_genes[mother]

            fst_part_cross = randint(0, size_individual)

            child_1 = np.append(father[:fst_part_cross], mother[fst_part_cross:], axis=0)
            child_2 = np.append(mother[:fst_part_cross], father[fst_part_cross:], axis=0)

            children.append(child_1)
            children.append(child_2)

    selecteds_genes.extend(children)

    return selecteds_genes


if __name__ == "__main__":
    population = generate_float_population(SIZE_POPULATION, NUMBER_DIMENSIONS)

    fitness_history = []
    number_generation = 1
    number_generation_vec = []
    mean_fitness = []
    for item in xrange(NUMBER_GENERATIONS):

        population = evolve(population)
        fitness_history.append(fitness_evaluation_individual(x_square_30, population[0]))  # best solution at the moment
        number_generation_vec.append(number_generation)
        mean_fitness.append(np.mean([fitness_evaluation_individual(x_square_30, population[0]) for individual in population]))
        number_generation += 1

    plot_lines = []
    plt.title("Genetic Algorithm Gleydson")
    plt.plot(number_generation_vec, fitness_history, marker="*")
    plt.plot(number_generation_vec, mean_fitness)
    blue_line = mlines.Line2D([], [], color='blue')
    green_line = mlines.Line2D([], [], color='green')
    plot_lines.append([blue_line, green_line])
    legend1 = plt.legend(plot_lines[0], ["Best Fitness", "Mean Fitness"], loc=1)
    plt.gca().add_artist(legend1)
    plt.xlabel("Generations")
    plt.show()

    print ("Best Solution", population[0])