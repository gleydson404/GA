from operator import itemgetter
from random import random, randint

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import BinarytoReal

NUMBER_GENERATIONS = 400
NUMBER_BITS = 16
NUMBER_DIMENSIONS = 30
SIZE_POPULATION = 100
HIGH_LIMIT = 100
LOW_LIMIT = -100
MAXIMIZATION = False


def x_square_30x(individual):
    return np.sum([x ** 2 for x in individual])

def x_square_30(individual):
    return np.sum([np.abs((x + 0.5) ** 2) for x in individual])


def roullete(population):

    fitness_list = [individual[0] for individual in population] #population is a list of tuples (fitness, individual), individual[0] brings me just the fitness

    total_fitness = np.sum(fitness_list)
    max_fitness = max(fitness_list)
    min_fitness = min(fitness_list)
    p = random() * total_fitness
    t = max_fitness + min_fitness
    choosen = 0
    for index in xrange(len(population) - 1):
        if MAXIMIZATION:
            p -= population[index][0]
        else:
            p -= (t - population[index][0])
        if p < 0:
            choosen = index
            break

    return choosen

    # random_number = randint(0, int(total_fitness))
    #
    # partial_sum = 0
    #
    # for index in xrange(len(fitness_list)):
    #
    #     partial_sum += fitness_list[index]
    #
    #     if partial_sum >= random_number:
    #         return index


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


def evolve2(population, tx_crossover=1, tx_mutation=0.3, tx_reproduction=0.3):
    evaluated_individuals = sorted(fitness_evaluation_population(population), key=itemgetter(0))

    new_population = []

    while len(new_population) <= SIZE_POPULATION:

        if tx_mutation > random(): #mutation
            individual = evaluated_individuals[roullete(evaluated_individuals)] # eu seleciono o inviduo antes de verificar a probabilidade, ou depois ?
            simple_mutation(individual[1])
            new_population.append(individual[1])

        if tx_crossover > random():

            father_index = roullete(evaluated_individuals)
            mother_index = roullete(evaluated_individuals)

            if father_index != mother_index:

                father = evaluated_individuals[father_index]
                mother = evaluated_individuals[mother_index]

                new_population.extend(two_points_crossover(mother[1], father[1]))

        if tx_reproduction > random():
            new_population.append(evaluated_individuals[roullete(evaluated_individuals)][1])

    new_population = sorted(fitness_evaluation_population(new_population), key=itemgetter(0))

    return [individual[1] for individual in new_population]  # remove fitness value, leaving just the individual


def one_point_crossover(mother, father):
    fst_part_cross = randint(0, len(father)-1)

    child_1 = np.append(father[:fst_part_cross], mother[fst_part_cross:], axis=0)
    child_2 = np.append(mother[:fst_part_cross], father[fst_part_cross:], axis=0)

    return [child_1, child_2]


def two_points_crossover(mother, father):
    fst_part_cross = randint(0, len(father) - 1)
    scd_part_cross = randint(fst_part_cross, len(father) - 1)

    # partes do primeiro pai fora dos pontos de corte mais a parte do segundo pai dentro dos pontos de corte
    part_1_father1 = mother[:fst_part_cross]
    part_2_father1 = father[fst_part_cross:scd_part_cross]
    part_3_father1 = mother[scd_part_cross:]

    part_1_father2 = father[:fst_part_cross]
    part_2_father2 = mother[fst_part_cross:scd_part_cross]
    part_3_father2 = father[scd_part_cross:]

    child_1 = np.append(part_1_father1, part_3_father1,  axis=0)
    child_1 = np.append(child_1, part_2_father1, axis=0)
    child_2 = np.append(part_1_father2, part_3_father2, axis=0)
    child_2 = np.append(child_2, part_2_father2, axis=0)

    return [child_1, child_2]


def uniform_crossover(mother, father):
    child_1 = []
    child_2 = []
    for index in range(len(mother)):
        if np.random.randint(2, size=1)[0] == 0:
            child_1.append(mother[index])
            child_2.append(father[index])
        else:
            child_1.append(father[index])
            child_2.append(mother[index])

    return [child_1, child_2]


def simple_mutation(individual):
    pst_to_mutate_crm = randint(0, len(individual)-1)
    pst_to_mutate_alelo = randint(0, NUMBER_BITS - 1)
    individual[pst_to_mutate_crm][pst_to_mutate_alelo] = np.random.randint(2, size=1)[0]
    return individual


if __name__ == "__main__":
    population = generate_float_population(SIZE_POPULATION, NUMBER_DIMENSIONS)

    fitness_history = []
    number_generation = 1
    number_generation_vec = []
    mean_fitness = []
    st_deviation = []
    for item in xrange(NUMBER_GENERATIONS):

        population = evolve2(population)
        fitness_history.append(fitness_evaluation_individual(x_square_30, population[0]))  # best solution at the moment
        number_generation_vec.append(number_generation)
        mean_fitness.append(np.mean([fitness_evaluation_individual(x_square_30, individual) for individual in population]))
        st_deviation.append(np.std([fitness_evaluation_individual(x_square_30, individual) for individual in population]))
        number_generation += 1

    plot_lines = []
    plt.title("Genetic Algorithm Gleydson")
    plt.plot(number_generation_vec, fitness_history)
    plt.plot(number_generation_vec, mean_fitness)
    plt.plot(number_generation_vec, st_deviation, color="yellow")
    blue_line = mlines.Line2D([], [], color='blue')
    green_line = mlines.Line2D([], [], color='green')
    yellow_line = mlines.Line2D([], [], color='yellow')
    plot_lines.append([blue_line, green_line, yellow_line])
    legend1 = plt.legend(plot_lines[0], ["Best Fitness", "Mean Fitness", "Standart Deviation"], loc=1)
    plt.gca().add_artist(legend1)
    plt.xlabel("Generations")
    plt.show()

    print ("Best Solution", population[0])

