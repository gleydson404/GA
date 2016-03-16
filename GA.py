import numpy as np
from operator import itemgetter, add
from random import random, randint

RIGHT_ANSWER = 371
POP_LENGHT = 100
NUMBER_GENERATIONS = 100
MIN = 0
MAX = 200
QTD_ELEMENTS_INDIVUAL = 5


def generate_int_individual(length, min=0, max=1):
    return np.random.randint(min, max, length)


def generate_float_individual(length):
    return np.random.random_sample(length)


def generate_int_population(count, length, min, max):
    return [generate_int_individual(length, min, max) for item in range(count)]

def generate_float_population(count, length, min, max):
    return [generate_float_individual(length) for item in range(count)]


def fitness_evaluation_individual(individual, righ_answer):
    sum = 0
    for element in individual:
        sum += element

    return np.abs(righ_answer - sum)


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
    selecteds = [x[1] for x in evaluated_individuals] #remove fitness value, lefting just the individual
    qtd_winners = int((len(selecteds) * percent_winners))
    selecteds_genes = selecteds[:qtd_winners]

    for individual in selecteds[qtd_winners:]:
        if random_select > random():
            selecteds_genes.append(individual)


    for individual in selecteds_genes:
        if mutate > random():
            position_to_mutate = randint(0, len(individual)-1)
            individual[position_to_mutate] = randint(min(individual), max(individual))

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


population = generate_int_population(POP_LENGHT, QTD_ELEMENTS_INDIVUAL, MIN, MAX)

fitness_history = [grade(population, RIGHT_ANSWER)]
for item in range(NUMBER_GENERATIONS):
    population = evolve(population)
    fitness_history.append(grade(population, RIGHT_ANSWER))

for datum in fitness_history:
    print datum


print population