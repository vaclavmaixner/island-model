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
parser.add_argument("--no_cycles", default=3,
                    type=int, help="Number of cycles of evolution.")
parser.add_argument("--no_cities", default=5,
                    type=int, help="Number of cities to travel through.")
parser.add_argument("--no_islands", default=1,
                    type=int, help="Number of islands for the island model.")
parser.add_argument("--population_size", default=4,
                    type=int, help="Size of population of individual islands.")
parser.add_argument("--map_size", default=(10, 10),
                    type=tuple, help="Int dimensions of the map.")
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

    def plot_solution(self):
        x = []
        y = []

        for city in self.route:
            x.append(city.x)
            y.append(city.y)

        x.append(self.route[0].x)
        y.append(self.route[0].y)

        plt.plot(x, y, color='red', marker='o', linestyle='dashed',
                 linewidth=1, markersize=4)
        plt.show()


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
                        mutation_rate=0.01, elite_size=0.01, number=j)
        islands.append(island)

    return islands


def breed(parent1, parent2):
    child = Solution(route=[], fitness=None)

    gene1 = int(random.random() * len(parent1.route))
    gene2 = int(random.random() * len(parent1.route))

    start_gene = min(gene1, gene2)
    end_gene = max(gene1, gene2)

    for i in range(start_gene, end_gene):
        child.route.append(parent1.route[i])

    print('child')
    child.print_route()

    finish = [item for item in parent2.route if item not in child.route]

    child.route = child.route + finish

    child.fitness = child.count_fitness()

    print('breed')
    parent1.print_route()
    parent2.print_route()
    child.print_route()
    print('/breed')

    return child


def breed_population(selected_for_breeding):
    bred_population = []

    for pair in selected_for_breeding:
        parent1 = pair[0]
        parent2 = pair[1]

        child = breed(parent1, parent2)

        print('breeding')
        parent1.print_route()
        parent2.print_route()
        child.print_route()
        print('/breeding')

        family = [parent1, parent2, child]
        family = sorted(family, key=lambda x: x.fitness, reverse=True)
        # print([x.fitness for x in family])
        family.pop()

        # parent1.plot_solution()
        # parent2.plot_solution()
        # child.plot_solution()

        # print(child.fitness, 'child')

        bred_population.extend(family)

    return bred_population


def select_mating_pool(solutions):
    print('mating')
    for solution in solutions:
        solution.print_route()
    print('/mating')
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

    # print(selected_for_breeding[0][0].fitness, 'parent1')
    # print(selected_for_breeding[0][1].fitness, 'parent2')
    # print(len(solutions))

    print('parent1')
    parent1.print_route()
    print('parent2')
    parent2.print_route()
    print('/parents')

    print('rest')
    for solution in solutions:
        solution.print_route()
    print('/rest')

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


def create_next_gen(island):
    # ranked_population = island.evaluate_generation()

    ranked_population = island.solutions

    selected_to_breed, non_breeding = select_mating_pool(ranked_population)

    bred_population = breed_population(selected_to_breed)

    population = non_breeding + bred_population

    mutated_population = mutate_population(population, island.mutation_rate)

    island.solutions = mutated_population
    island.best_solution = island.evaluate_generation()[0]

    print()
    # sys.stdout.write('.')
    # sys.stdout.flush()

    return island


def Main():
    islands = setup_test_data()

    for i in range(args.no_cycles):
        # print('New Generation')
        for island in islands:
            island = create_next_gen(island)
            # print(island.best_solution.fitness)

            island.progress.append(island.best_solution.fitness)

            if island.number == 0:
                island.overview(extended=True)
                time.sleep(0.5)

    print(islands[0].progress)
    plt.plot(islands[0].progress)
    plt.show()

    islands[0].best_solution.plot_solution()

    print(islands[1].progress)
    plt.plot(islands[1].progress)
    plt.show()

    islands[1].best_solution.plot_solution()


Main()
