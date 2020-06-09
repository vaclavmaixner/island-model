import sys
import random
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time

parser = argparse.ArgumentParser()
parser.add_argument("--no_cycles", default=2000,
                    type=int, help="Number of cycles of evolution.")
parser.add_argument("--no_cities", default=40,
                    type=int, help="Number of cities to travel through.")
parser.add_argument("--no_islands", default=4,
                    type=int, help="Number of islands for the island model.")
parser.add_argument("--population_size", default=300,
                    type=int, help="Size of population of individual islands.")
parser.add_argument("--map_size", default=(15, 15),
                    type=tuple, help="Int dimensions of the map.")
parser.add_argument("--mutation_rate", default=(0.01, 0.02, 0.04, 0.005),
                    type=tuple, help="Mutation rates for the individual islands")
parser.add_argument("--catastrophy_lethality", default=0.2,
                    type=float, help="What portion of island perishes in catastrophy")
parser.add_argument("--migration_rate", default=0.01,
                    type=float, help="How often does a solution migrate between islands")
parser.add_argument("--plot", default=False,
                    action="store_true", help="Plot progress.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
args = parser.parse_args()


class City:
    def __init__(self, index, x, y):
        self.index = index
        self.x = x
        self.y = y


class Solution():
    def __init__(self, route, fitness):
        self.route = route
        self.fitness = fitness

    def distance(self, city1, city2):
        dx = abs(city2.x - city1.x)
        dy = abs(city2.y - city1.y)
        return np.sqrt((dx ** 2) + (dy ** 2))

    def count_fitness(self):
        total_distance = 0.0

        for i in range(len(self.route)):
            from_city = self.route[i]
            to_city = None

            if i+1 < len(self.route):
                to_city = self.route[i+1]
            else:
                to_city = self.route[0]

            total_distance += self.distance(from_city, to_city)

        self.fitness = 1 / float(total_distance)

        return self.fitness

    def print_route(self):
        # print('Route: ', self.fitness)
        for city in self.route:
            print(city.index, end=' ')
        print(end='     ')
        print(self.fitness)

    def plot_solution(self, save, i):
        x = []
        y = []

        for city in self.route:
            x.append(city.x)
            y.append(city.y)

        x.append(self.route[0].x)
        y.append(self.route[0].y)

        plt.clf()
        plt.plot(x, y, color='red', marker='o', linestyle='dashed',
                 linewidth=1, markersize=4)
        # plt.show()
        if save:
            plt.savefig('out/route' + str(i) + '.png')


class Island():
    def __init__(self, solutions, mutation_rate, elite_size, number):
        self.solutions = solutions
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.best_solution = self.evaluate_generation()[0]
        self.number = number
        self.progress = []

    def evaluate_generation(self):
        ordered_solutions = []
        for solution in self.solutions:
            solution.count_fitness()
            ordered_solutions.append(solution)

        ordered_solutions = sorted(
            ordered_solutions, key=lambda x: x.fitness, reverse=True)

        self.best_solution = ordered_solutions[0]

        # print('Solution fitness:')
        # print([x.fitness for x in ordered_solutions])
        # print()

        return ordered_solutions

    def overview(self, extended):
        if extended:
            # print('Island overview', self.number)
            # print('Best result: ', self.best_solution.fitness)
            print('_'*40)
            # print([(solution.print_route(), solution.fitness) for solution in self.solutions])
            for solution in self.solutions:
                solution.print_route()
                # print(solution.fitness, '-fitness')
            print()

    def show_diversity(self):
        unique_routes = []

        routes = [x.route for x in self.solutions]

        for route in routes:
            if route not in unique_routes:
                unique_routes.append(route)
        print(len(unique_routes), end=' ')
        return unique_routes


def setup_test_data():
    random.seed(a=args.seed)

    cities = []
    for i in range(args.no_cities):
        city = City(index=i,
                    x=int(random.random()*args.map_size[0]),
                    y=int(random.random()*args.map_size[1]))
        cities.append(city)

    islands = []
    for j in range(args.no_islands):
        solutions = []
        for k in range(args.population_size):
            route = random.sample(cities, len(cities))
            solution = Solution(route=route, fitness=0)
            solution.fitness = solution.count_fitness()
            solutions.append(solution)

        island = Island(solutions=solutions,
                        mutation_rate=args.mutation_rate[j], elite_size=0.01, number=j)
        islands.append(island)

    return islands


def breed(parent1, parent2):
    child1 = Solution(route=[], fitness=None)
    child2 = Solution(route=[], fitness=None)

    gene1 = int(random.random() * len(parent1.route))
    gene2 = int(random.random() * len(parent1.route))

    start_gene = min(gene1, gene2)
    end_gene = max(gene1, gene2)

    for i in range(start_gene, end_gene):
        child1.route.append(parent1.route[i])
        child2.route.append(parent2.route[i])

    finish1 = [item for item in parent2.route if item not in child1.route]
    finish2 = [item for item in parent1.route if item not in child2.route]

    child1.route = child1.route + finish1
    child2.route = child2.route + finish2

    child1.fitness = child1.count_fitness()
    child2.fitness = child2.count_fitness()

    return child1, child2


def breed_population(selected_for_breeding):
    bred_population = []

    for pair in selected_for_breeding:
        parent1 = pair[0]
        parent2 = pair[1]

        child1, child2 = breed(parent1, parent2)

        family = [parent1, parent2, child1, child2]

        unique_family = []
        routes = []
        for solution in family:
            if solution.route not in routes:
                unique_family.append(solution)
                routes.append(solution.route)

        family = sorted(unique_family, key=lambda x: x.fitness, reverse=True)
        family = family[:2]

        bred_population.extend(family)

    return bred_population


def select_mating_pool(solutions):
    selected_for_breeding = []

    no_parent_pairs = len(solutions) // 4

    for i in range(no_parent_pairs):
        parent1 = random.choice(solutions)
        while True:
            parent2 = random.choice(solutions)
            if parent1.route != parent2.route:
                break
        selected_for_breeding.append((parent1, parent2))
        solutions.remove(parent1)
        solutions.remove(parent2)

    return selected_for_breeding, solutions


def mutate(solution, mutation_rate):
    for i in range(len(solution.route)):
        if(random.random() < mutation_rate):
            city1_index = int(random.random() * len(solution.route))
            city1 = solution.route[city1_index]

            swap_direction = random.choice([-1, 1])
            city2_index = (city1_index + swap_direction) % len(solution.route)
            city2 = solution.route[city2_index]

            city1, city2 = city2, city1

            solution.route[city1_index] = city2
            solution.route[city2_index] = city1

    return solution


def mutate_population(population, mutation_rate):
    mutants = []

    for solution in population:
        mutants.append(mutate(solution, mutation_rate))

    return mutants


def cause_catastrophy(island):
    no_perished = int(len(island.solutions) * args.catastrophy_lethality)

    route = island.solutions[0].route
    new_route = route[:]

    for i in range(len(island.solutions)-no_perished, len(island.solutions)):
        random.shuffle(new_route)
        island.solutions[i].route = new_route
        island.solutions[i].fitness = island.solutions[i].count_fitness()

    return island


def create_next_gen(island, catastrophy):
    ranked_population = island.solutions

    selected_to_breed, non_breeding = select_mating_pool(ranked_population)

    bred_population = breed_population(selected_to_breed)

    population = non_breeding + bred_population

    mutated_population = mutate_population(population, island.mutation_rate)

    island.evaluate_generation()

    if catastrophy:
        island = cause_catastrophy(island)

    island.solutions = mutated_population
    island.best_solution = island.evaluate_generation()[0]

    return island


def migrate(island1, island2):
    island1_index = random.randint(0, len(island1.solutions))
    island2_index = random.randint(0, len(island2.solutions))

    island1.solutions[island1_index], island2.solutions[island2_index] = island2.solutions[island2_index], island1.solutions[island1_index]

    return island1, island2


def Main():
    islands = setup_test_data()

    for i in range(args.no_cycles):
        for island in islands:
            if i == 10:
                island.best_solution.plot_solution(save=True, i=i)

            if i >= 400 and i % 50 == 0:
                island = create_next_gen(island, catastrophy=True)
            else:
                island = create_next_gen(island, catastrophy=False)
            island.progress.append(island.best_solution.fitness)

        if random.random() < args.migration_rate:
            index = int(random.random() * len(islands))
            island1 = islands[index]
            island2 = islands[(index + 1) % len(islands)]
            island1, island2 = migrate(island1, island2)

    for i in range(len(islands)):
        plt.clf()
        plt.plot(islands[i].progress)
        name = 'out/progress_' + str(i) + '.png'
        plt.savefig(name)

        islands[i].best_solution.plot_solution(save=True, i=i+100)

    print('Done')


Main()
