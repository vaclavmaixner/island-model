import sys
import random
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--no_cycles", default=3,
                    type=int, help="Number of cycles of evolution.")
parser.add_argument("--no_cities", default=8,
                    type=int, help="Number of cities to travel through.")
parser.add_argument("--no_islands", default=2,
                    type=int, help="Number of islands for the island model.")
parser.add_argument("--population_size", default=20,
                    type=int, help="Size of population of individual islands.")
parser.add_argument("--map_size", default=(10, 10),
                    type=tuple, help="Int dimensions of the map.")
parser.add_argument("--plot", default=False,
                    action="store_true", help="Plot progress.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
args = parser.parse_args()


class City:
    def __init__(self, x, y):
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
        print('Route: ', self.fitness)
        for city in self.route:
            print(city.x, city.y)
        print()

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
    def __init__(self, solutions, mutation_rate, elite_size):
        self.solutions = solutions
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

    def evaluate_generation(self):
        ordered_solutions = []
        for solution in self.solutions:
            solution.count_fitness()
            ordered_solutions.append(solution)

        ordered_solutions = sorted(ordered_solutions, key=lambda x: x.fitness, reverse=True)
        print('Solution fitness:')
        print([x.fitness for x in ordered_solutions])
        print()
        return ordered_solutions

    def breed(self, parent1, parent2):
        child = Solution(route=[], fitness=None)
        
        gene1 = int(random.random() * len(parent1.route))
        gene2 = int(random.random() * len(parent1.route))

        startGene = min(gene1, gene2)
        endGene = max(gene1, gene2)

        for i in range(startGene, endGene):
            child.route.append(parent1.route[i])

        finish = [item for item in parent2.route if item not in child.route]

        child.route = child.route + finish

        child.fitness = child.count_fitness()

        return child


    def select_mating_pool(self, solutions):
        selected_for_breeding = []
        no_parents = len(self.solutions) // 4
        print(no_parents)

        for i in range(no_parents):
            parent1 = random.choice(self.solutions)
            while True:
                parent2 = random.choice(self.solutions)
                if parent1 != parent2:
                    break
            selected_for_breeding.append((parent1, parent2))
            self.solutions.remove(parent1)
            self.solutions.remove(parent2)

        print(selected_for_breeding[0][0].fitness, 'parent1')
        print(selected_for_breeding[0][1].fitness, 'parent2')
        print(len(self.solutions))
        return selected_for_breeding

    def breed_population(self, selected_for_breeding):
        for pair in selected_for_breeding:
            parent1 = pair[0]
            parent2 = pair[1]

            child = self.breed(parent1, parent2)

            family = [parent1, parent2, child]
            family = sorted(family, key=lambda x: x.fitness, reverse=True)
            family.pop()

            print(child.fitness, 'child')
        
            self.solutions.extend(family)
            print(len(self.solutions))
        
    def create_next_gen(self):
        # for solution in self.solutions:
        #     pass
        self.breed_population(self.select_mating_pool())


def setup_test_data():
    random.seed(a=args.seed)

    cities = []
    for i in range(args.no_cities):
        city = City(x=int(random.random()*args.map_size[0]),
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
                        mutation_rate=0.1, elite_size=0.01)
        islands.append(island)

    for island in islands:
        island.evaluate_generation()

    islands[0].create_next_gen()

    for island in islands:
        island.evaluate_generation()


# def run_tsp():
#     n = 0
#     while n < args.no_cycles:
#         for island in islands:
#             new_population = create_next_gen(old_population)
#             n += 1


setup_test_data()
