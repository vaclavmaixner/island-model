import sys
import random
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    def count_fitness(self, route):
        total_distance = 0.0

        for i in range(len(route)):
            from_city = route[i]
            to_city = None

            if i+1 < len(route):
                to_city = route[i+1]
            else:
                to_city = route[0]

            total_distance += self.distance(from_city, to_city)

        return 1 / float(total_distance)

    def print_route(self):
        print('Route: ', self.fitness)
        for city in self.route:
            print(city.x, city.y)
        print()


class Island():
    def __init__(self, solutions, mutation_rate, elite_size):
        self.solutions = solutions
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

    def create_next_gen(self):
        for solution in self.solutions:
            # print(solution)
            pass


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
            solution.fitness = solution.count_fitness(solution.route)
            solutions.append(solution)

        island = Island(solutions=solutions,
                        mutation_rate=0.1, elite_size=0.01)
        islands.append(island)
    

# def run_tsp():
#     n = 0
#     while n < args.no_cycles:
#         for island in islands:
#             new_population = create_next_gen(old_population)
#             n += 1


setup_test_data()
