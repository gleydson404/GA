from operator import itemgetter
from random import random, randint

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

import BinarytoReal

# NUMBER_GENERATIONS = 100
# NUMBER_BITS = 16
# NUMBER_DIMENSIONS = 30
# SIZE_POPULATION = 30
# HIGH_LIMIT = 100
# LOW_LIMIT = -100
# MAXIMIZATION = False
# TX_CROSSOVER = 1
# TX_MUTATION = 0.3
# TX_REPRODUCTION = 0.3
FUNCTION = ''

OPERATORS = {
    'crossover': 'one_points_crossover',
    'mutation': 'simple_mutation',
    'elitism': True,
    'selection': 'roullete',
    'number_generations': 400,
    'number_bits': 10,
    'number_dimensions': 2,
    'size_population': 100,
    'high_limit': 5,
    'low_limit': -5,
    'is_maximization': True,
    'tx_crossover': 1,
    'tx_mutation': 0.3,
    'tx_reproduction': 0.3,
    'function': 'rastrigin'
}


def x_square(individual):
    return np.sum([x ** 2 for x in individual])


def x_square_plus_0dot5(individual):
    return np.sum([np.abs((x + 0.5) ** 2) for x in individual])


def multimodal_min_locals(individual):
    return np.sum([-x * np.sin(np.sqrt(np.abs(x))) for x in individual ])


def rastrigin(individual):
    x = individual[0]
    y = individual[1]

    zx = (x ** 2) - 10 * np.cos(2 * np.pi * x) + 10
    zy = (y ** 2) - 10 * np.cos(2 * np.pi * y) + 10
    z = zx + zy

    return -z


def setup_funcion(name_function, FUNCTION):
    if name_function == 'x_square':
        FUNCTION = x_square
    elif name_function == 'x_square_plus_0dot5':
        FUNCTION = x_square_plus_0dot5
    elif name_function == 'multimodal_min_locals':
        FUNCTION = multimodal_min_locals
    elif name_function == 'rastrigin':
        FUNCTION = rastrigin

    return FUNCTION

def tournament(population):
    pass


def roullete(population):
    # population is a list of tuples (fitness, individual), individual[0] brings me just the fitness
    fitness_list = [individual[0] for individual in population]
    fitness_list = [normalization(fitness_element, min(fitness_list), max(fitness_list)) for fitness_element in fitness_list]
    total_fitness = np.sum(fitness_list)
    max_fitness = max(fitness_list)
    min_fitness = min(fitness_list)

    p = random() * total_fitness
    t = max_fitness + min_fitness
    choosen = 0
    for index in xrange(len(fitness_list) - 1):
        if OPERATORS['is_maximization']:
            p -= fitness_list[index]
        else:
            p -= (t - fitness_list[index])
        if p < 0:
            choosen = index
            break

    return choosen


def normalization(value, min, max):
    old_range = max - min
    new_min = 1
    new_range = 10 + 0.9999999999 - new_min
    return float((value - min) / float(old_range * new_range + new_min))


def convert_individual_to_real(individual):
    converted = []

    for item in individual:
        converted.append(BinarytoReal.convert(OPERATORS['high_limit'], OPERATORS['low_limit'], OPERATORS['number_bits'], ''.join(map(str, item))))

    return converted


def fitness_evaluation_individual(fitness_function, individual):
    return fitness_function(convert_individual_to_real(individual))


def generate_binary_individual(size):
    return np.array([np.random.randint(2, size=OPERATORS['number_bits']) for _ in range(size)])


def generate_float_population(count, size):
    # return [generate_float_individual(size) for item in range(count)]
    return map(generate_binary_individual, [size] * count)


def fitness_evaluation_population(population):
    evaluated_individuals = []
    for individual in population:
        evaluated_individuals.append((fitness_evaluation_individual(FUNCTION, individual), individual))

    return evaluated_individuals


def evolve(population, percent_winners=0.3, random_select=0.1, mutate=0.3):
    if OPERATORS['is_maximization']:
        evaluated_individuals = sorted(fitness_evaluation_population(population), key=itemgetter(0), reverse=True)
    else:
        evaluated_individuals = sorted(fitness_evaluation_population(population), key=itemgetter(0), reverse=False)

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
            pst_to_mutate_alelo = randint(0, OPERATORS['number_bits']-1)
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


def evolve2(population, tx_crossover=OPERATORS['tx_crossover'], tx_mutation=OPERATORS['tx_mutation'],
            tx_reproduction=OPERATORS['tx_reproduction'], operators=OPERATORS, function=FUNCTION):
    new_population = []

    if len(operators) is not 0:

        if OPERATORS['is_maximization']:
            evaluated_individuals = sorted(fitness_evaluation_population(population), key=itemgetter(0), reverse=True)
        else:
            evaluated_individuals = sorted(fitness_evaluation_population(population), key=itemgetter(0), reverse=False)

        while len(new_population) <= OPERATORS['size_population']:

            if tx_mutation > random(): #mutation
                individual = evaluated_individuals[roullete(evaluated_individuals)]
                if operators['mutation'] == 'twors_mutation':
                    new_population.append(twors_mutation(individual[1]))
                elif operators['mutation'] == 'thrors_mutation':
                    new_population.append(thrors_mutation(individual[1]))
                elif operators['mutation'] == 'simple_mutation':
                    new_population.append(simple_mutation(individual[1]))

            if tx_crossover > random():

                father_index = roullete(evaluated_individuals)
                mother_index = roullete(evaluated_individuals)

                if father_index != mother_index:

                    father = evaluated_individuals[father_index]
                    mother = evaluated_individuals[mother_index]

                    if operators['crossover'] == 'one_point_crossover':
                        new_population.extend(one_point_crossover(mother[1], father[1]))
                    elif operators['crossover'] == 'two_points_crossover':
                        new_population.extend(two_points_crossover(mother[1], father[1]))
                    elif operators['crossover'] == 'uniform_crossover':
                        new_population.extend(uniform_crossover(mother[1], father[1]))

            if operators['elitism']:
                if tx_reproduction > random():
                    new_population.append(evaluated_individuals[roullete(evaluated_individuals)][1])

        if OPERATORS['is_maximization']:
            new_population = sorted(fitness_evaluation_population(new_population), key=itemgetter(0), reverse=True)
        else:
            new_population = sorted(fitness_evaluation_population(new_population), key=itemgetter(0), reverse=False)

        return [individual[1] for individual in new_population]  # remove fitness value, leaving just the individual
    else:
        return "Sem parametros"


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
    pst_to_mutate_alelo = randint(0, OPERATORS['number_bits'] - 1)
    individual[pst_to_mutate_crm][pst_to_mutate_alelo] = np.random.randint(2, size=1)[0]
    return individual


def twors_mutation(individual):
    random_position_1 = randint(0, len(individual) - 1)
    random_position_2 = randint(0, len(individual) - 1)

    individual[random_position_1], individual[random_position_2] = \
        individual[random_position_2], individual[random_position_1]
    return individual


def thrors_mutation(individual):

    random_position_i = randint(0, len(individual) - 1)
    random_position_j = randint(0, len(individual) - 1)
    random_position_k = randint(0, len(individual) - 1)

    individual_i_old = individual[random_position_i]
    individual[random_position_i] = individual[random_position_j]
    individual[random_position_j] = individual[random_position_k]
    individual[random_position_k] = individual_i_old

    return individual


if __name__ == "__main__":
    FUNCTION = setup_funcion(OPERATORS['function'], FUNCTION)
    population = generate_float_population(OPERATORS['size_population'], OPERATORS['number_dimensions'])

    fitness_history = []
    number_generation = 1
    number_generation_vec = []
    mean_fitness = []
    st_deviation = []
    for index in xrange(OPERATORS['number_generations']):

        population = evolve2(population)

        fitness_history.append(fitness_evaluation_individual(FUNCTION, population[0]))  # best solution at the moment
        number_generation_vec.append(index + 1)
        mean_fitness.append(np.mean([fitness_evaluation_individual(FUNCTION, individual) for individual in population]))
        st_deviation.append(np.std([fitness_evaluation_individual(FUNCTION, individual) for individual in population]))

        if index % 10 == 0:
            cumulator = 0
            for inner_index in reversed(xrange(index)):
                if mean_fitness[index] == mean_fitness[inner_index]:
                    cumulator += 1
            if cumulator >= 5:
                print 'break'
                break

    plot_lines = []
    plt.title("Genetic Algorithm Gleydson")
    plt.plot(number_generation_vec, fitness_history, color='blue', linewidth=4, linestyle='-')
    plt.plot(number_generation_vec, mean_fitness, color='green', linewidth=4, linestyle='-.')
    plt.plot(number_generation_vec, st_deviation, color="red", linewidth=4, linestyle='--')

    blue_line = mlines.Line2D([], [], color='blue')
    green_line = mlines.Line2D([], [], color='green')
    red_line = mlines.Line2D([], [], color='red')
    plot_lines.append([blue_line, green_line, red_line])

    legend1 = plt.legend(plot_lines[0], ["Best Fitness", "Mean Fitness", "Standart Deviation"], loc=1)
    plt.grid(True)
    plt.autoscale()
    plt.gca().add_artist(legend1)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.show()

    print ("Best Solution", convert_individual_to_real(population[0]))
    print ("Best Solution in the function", (fitness_evaluation_individual(FUNCTION, population[0])))
