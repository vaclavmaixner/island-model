#!/usr/bin/env python3

import sys
import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt


''' 
The problem is: distribute pizza across a town, using N vehicles
to deliver to M customers. This algorithm uses the solution to
Travelling Salesman Problem, only with multiple salesmen. These
are created by having the one salesman from TSP come back periodically
to the pizzeria, thus simulating multiple salesmen travelling at the 
same time.
'''


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        dx = abs(self.x - city.x)
        dy = abs(self.y - city.y)
        return np.sqrt((dx ** 2) + (dy ** 2))


class Fitness():
    def __init__(self, route):
        self.route = route


def setupCityList(noCities, gridSize):
    cityList = []
    testList = [(17, 1), (16, 12), (19, 8), (0, 12), (2, 3), (11, 15), (7, 19), (16, 2), (11, 6), (0, 19), (9, 3), (10, 5),
                (8, 12), (4, 17), (1, 8), (9, 8), (13, 14), (4, 2), (4, 10), (8, 0), (16, 9), (2, 0), (4, 7), (7, 19), (9, 8)]
    for i in range(0, noCities):
        city = City(int(random.random() * gridSize),
                    int(random.random() * gridSize))
        cityList.append(city)

    return test


# count distance between two cities
def countDistance(city1, city2):
    distance = ((city2.x - city1.x) ** 2 + (city2.y - city1.y) ** 2)
    return distance


# return inverse of route length
def routeFitness(route):
    totalDistance = 0
    i = 0

    for i in range(0, len(route)):
        from_city = route[i]
        to_city = None
        if i+1 < len(route):
            to_city = route[i+1]
        else:
            to_city = route[0]
        totalDistance += countDistance(from_city, to_city)

    routeFitness = 1 / float(totalDistance)
    return routeFitness


# call route ranking
def rankRoutes(population):
    rankedRoutes = {}

    for i in range(0, len(population)):
        rankedRoutes[i] = routeFitness(population[i])
    result = sorted(rankedRoutes.items(),
                    key=operator.itemgetter(1), reverse=True)
    return result


# return a random route out of the city list
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


# initialize population
def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))

    return population


def naturalSelection(rankedPop, eliteSize):
    selectionResult = []

    df = pd.DataFrame(np.array(rankedPop), columns=["Index", "Fitness"])
    df["cum_sum"] = df.Fitness.cumsum()
    df["cum_perc"] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResult.append(rankedPop[i][0])

    # adds more fit people to the elite
    for i in range(0, len(rankedPop) - eliteSize):
        pick = 100 * random.random()

        for i in range(0, len(rankedPop)):
            if pick <= df.iat[i, 3]:
                selectionResult.append(rankedPop[i][0])
                break

    return selectionResult


# create easier data structure for mating
def matingPool(population, selectionResult):
    matingPool = []

    for i in range(0, len(selectionResult)):
        index = selectionResult[i]
        matingPool.append(population[index])

    return matingPool


# breeds two parents to create one child
def breed(parent1, parent2):
    child = []

    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


# breed the whole population
def breedPopulation(matingPool, eliteSize):
    children = []

    pool = random.sample(matingPool, len(matingPool))

    for i in range(0, eliteSize):
        children.append(matingPool[i])

    for i in range(eliteSize, len(matingPool)):
        child = breed(pool[i], pool[len(matingPool) - i - 1])
        children.append(child)

    return children


# causes a single mutation based on the rate, mutation is a swap of 2 citites
def mutate(mutant, mutationRate):
    for swapped in range(len(mutant)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(mutant))

            city1 = mutant[swapped]
            city2 = mutant[swapWith]

            mutant[swapped] = city2
            mutant[swapWith] = city1
    return mutant


def mutatePopulation(population, mutationRate, eliteSize):
    mutants = []

    for i in range(0, eliteSize):
        mutants.append(population[i])
    for i in range(eliteSize, len(population)):
        mutant = mutate(population[i], mutationRate)
        mutants.append(mutant)

    if (population[0][0].x != mutants[0][0].x):
        print('uh oh jelly beano')
    return mutants


def nextGeneration(currentGeneration, eliteSize, mutationRate):
    rankedPop = rankRoutes(currentGeneration)
    selectionResult = naturalSelection(rankedPop, eliteSize)
    matingpool = matingPool(currentGeneration, selectionResult)
    children = breedPopulation(matingpool, eliteSize)
    mutants = mutatePopulation(children, mutationRate, eliteSize)

    return mutants


def Main():
    noGenerations = 900
    noCities = 5
    gridSize = 20
    popSize = noCities * 10
    eliteSize = int(popSize / 10)
    mutationRate = 0.1

    cityList = setupCityList(noCities, gridSize)

    progress = []

    # population is a list of routes
    pop = initialPopulation(popSize, cityList)

    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0, noGenerations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))

    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]

    plt.subplot(121)
    plt.plot(progress)

    plt.subplot(122)
    print(bestRoute)

    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()


Main()
