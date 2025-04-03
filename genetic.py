import AntColony as ant
import data as dt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import greedy as gd
import random

class Genetic():
    def __init__(self, greedy, initial_population=None, initial_size=1, num_iterations=1000, verbose=True):
        # Store the greedy solution without deep copying
        self.initial = greedy
        self.population = initial_population
        self.initial_size = initial_size
        self.num_iterations = num_iterations
        self.zone_times = []
        self.verbose = verbose

    def create_initial_population(self):
        """Generate initial population using ant colony optimization with random parameters"""
        if self.population is None:
            if self.verbose:
                print("Creating initial population...")
            self.population = []
            for i in range(self.initial_size):
                # Create a new PTL instance instead of deep copying
                ptl_copy = dt.PTL(option=self.initial._option)
                ptl_copy.load_data()
                
                # Copy the solution from the initial greedy solution
                ptl_copy.set_parameter("solution", self.initial._solution)
                ptl_copy.compute_order_times()
                ptl_copy.compute_total_zone_time()
                
                # Apply ant colony optimization with random parameters
                ant_colony = ant.AntColony(
                    ptl_copy, 
                    n_ants=random.randint(5, 20), 
                    n_iterations=random.randint(50, 200), 
                    alpha=random.uniform(0.5, 2.0), 
                    beta=random.uniform(1.0, 3.0), 
                    rho=random.uniform(0.05, 0.3), 
                    q=random.uniform(0.5, 2.0)
                )
                ant_colony.run_ant_colony(verbose=False)
                self.population.append(ant_colony.ptl)
        if self.verbose:
            print("Initial population created")
        
        # Sort the population by fitness (zone times)
        self.population.sort(key=lambda x: x.get_parameter("zone_times"))
        
        # Update zone times list
        self.zone_times = [individual.get_parameter("zone_times") for individual in self.population]
        
        return self.population

    def swap_method(self, individual_solution):
        """Swap two random keys in the solution dictionary"""
        # Create a copy to avoid modifying the original
        new_individual = {k: v for k, v in individual_solution.items()}
        
        # Swap two random keys
        keys = list(new_individual.keys())
        if len(keys) < 2:
            return new_individual  # Can't swap if there's only one key
            
        a = random.choice(keys)
        b = random.choice(keys)
        while a == b:
            b = random.choice(keys)
            
        temp = new_individual[a]
        new_individual[a] = new_individual[b]
        new_individual[b] = temp
        
        return new_individual
    
    def create_new_individual(self):
        """Create a new individual by mutating an existing one"""
        # Select random individual from the population
        individual = random.choice(self.population)
        
        # Create a new PTL instance
        new_individual = dt.PTL(option=individual._option)
        new_individual.load_data()
        
        # Apply swap mutation to the solution and assign it to the new individual
        new_solution = self.swap_method(individual._solution)
        new_individual._solution = new_solution
        
        # Recompute metrics for the new individual
        new_individual.compute_order_times()
        new_individual.compute_total_zone_time()
        
        return new_individual

    def evaluate_and_insert(self, new_individual, iteration_num):
        """Evaluate new individual and insert into population if good enough"""
        new_zone_time = new_individual.get_parameter("zone_times")
        max_zone_time = max(new_zone_time)
        
        # Check if new individual is better than any in the current population
        insert_position = None
        for i in range(len(self.zone_times)):
            max_zone_time_i = max(self.zone_times[i])
            if max_zone_time < max_zone_time_i:
                insert_position = i
                break
        
        # If the new individual is better than at least one current individual
        if insert_position is not None:
            self.population.insert(insert_position, new_individual)
            self.zone_times.insert(insert_position, new_zone_time)
            if self.verbose:
                print(f"Iteration {iteration_num + 1}/{self.num_iterations}: New individual added with max zone time {max(new_zone_time):.2f}. Worst individual removed.")
            
            # Remove the worst individual if population exceeds the limit
            if len(self.population) > self.initial_size:
                self.population.pop()
                self.zone_times.pop()
            
            return True
        
        return False
    
    def genetic_algorithm_iteration(self, iteration_num):
        """Perform one iteration of the genetic algorithm"""
        # Create a new individual through mutation
        new_individual = self.create_new_individual()
        
        # Evaluate and potentially insert the new individual
        success = self.evaluate_and_insert(new_individual, iteration_num)
        
        return success
    
    def genetic_algorithm(self):
        """Run the complete genetic algorithm"""
        # Create initial population if not provided
        if not self.population:
            self.create_initial_population()
            
        # Main loop
        improvements = 0
        for i in range(self.num_iterations):
            if self.genetic_algorithm_iteration(iteration_num=i):
                improvements += 1
        
        best_individual = self.population[0]
        # convert the best individual to a PTL instance
        best_individual = dt.PTL(option=best_individual._option)
        best_individual.load_data()
        # Assign the best solution found to the PTL instance
        best_individual._solution = self.population[0]._solution
        best_individual.compute_order_times()
        best_individual.compute_total_zone_time()
        return best_individual  # Return the best individual
