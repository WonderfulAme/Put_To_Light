import numpy as np
import random
import greedy as gd

class AntColony:
    def __init__(self, ptl, n_ants, n_iterations, alpha, beta, rho, q):
        self.ptl = ptl
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.pheromone = None

    def ant_colony_optimization(self):
        """
        Solve the order-to-departure assignment problem using Ant Colony Optimization.
        Each order must be assigned to exactly one departure, and each departure can handle at most one order.
        Departures are assigned to zones, and we want to minimize the maximum zone time.
        
        Parameters:
        -----------
        ptl : object
            Problem instance object that contains:
            - Methods: get_parameter(), compute_order_times(), compute_total_zone_time()
            - Attributes: _solution (dictionary to store order-to-departure assignments)
        
        Returns:
        --------
        best_solution : dict
            The best order-to-departure assignment found
        best_makespan : float
            The makespan (maximum zone time) of the best solution
        """
        # Get problem data
        orders = self.ptl.get_set("orders")
        departures = self.ptl.get_set("departures")
        n_orders = self.ptl.get_parameter("n_orders")
        n_departures = self.ptl.get_parameter("n_departures")
    
        # Initialize pheromone matrix (orders x departures)
        self.pheromone = np.random.rand(n_orders, n_departures) * 0.1  # Small initial pheromone values

        # Track best solution
        best_solution = None
        best_makespan = float('inf')
        
        # Start iterations
        for iteration in range(self.n_iterations):
            # Solutions for this iteration
            iteration_solutions = []
            iteration_makespans = []
            
            # Each ant builds a solution
            for ant in range(self.n_ants):      
                # Create a solution trail (order -> departure assignment)
                solution = self.generate_solution_trails()
                
                # Apply solution to the problem instance
                self.ptl.set_parameter("solution", solution)
                self.ptl.compute_order_times()
                
                # Compute zone times based on the assignment
                zone_times = self.ptl.compute_total_zone_time()
                
                # Calculate makespan (maximum time across all zones)
                makespan = max(zone_times)
                
                # Store solution
                iteration_solutions.append(solution)
                iteration_makespans.append(makespan)
                
                # Update best solution if better
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_solution = solution.copy()

            # Evaporate pheromones and deposit new pheromones based on solution quality
            self.evaporate_and_deposit_pheromones(iteration_solutions, iteration_makespans, orders, departures)
        
        # Apply the best solution to the problem
        if best_solution:
            self.ptl.set_parameter("solution", best_solution)
            self.ptl.compute_order_times()
            self.ptl.compute_total_zone_time()
        
        return best_solution, best_makespan
    
    def generate_solution_trails(self):
        """
        Generate a valid solution where each order is assigned to exactly one departure,
        and each departure handles at most one order.
        
        Returns:
        --------
        solution : dict
            Dictionary mapping orders to departures
        """
        orders = self.ptl.get_set("orders")
        departures = self.ptl.get_set("departures")
        
        # Get processing times for orders
        order_processing_times = self.ptl.get_parameter("total_departure_time")
        
        solution = {}
        available_orders = orders.copy()
        available_departures = departures.copy()
        
        # Process orders randomly
        random.shuffle(available_orders)
        
        for order in available_orders:
            order_index = orders.index(order)
            
            # Calculate probabilities for assigning this order to available departures
            probabilities = np.zeros(len(available_departures))
            
            for i, departure in enumerate(available_departures):
                departure_index = departures.index(departure)
                
                # Use inverse of processing time as heuristic (faster processing is better)
                processing_time = order_processing_times.get((order, departure), 1.0)
                heuristic_value = 1.0 / (processing_time + 0.1)  # Add 0.1 to avoid division by zero
                
                # Calculate probability based on pheromone and heuristic
                probabilities[i] = (self.pheromone[order_index, departure_index] ** self.alpha) * \
                                   (heuristic_value ** self.beta)
            
            # Normalize probabilities
            if np.sum(probabilities) > 0:
                probabilities = probabilities / np.sum(probabilities)
            else:
                probabilities = np.ones(len(available_departures)) / len(available_departures)
            
            # Select departure based on probabilities
            departure_idx = np.random.choice(len(available_departures), p=probabilities)
            selected_departure = available_departures.pop(departure_idx)
            
            # Assign order to departure
            solution[order] = selected_departure
            
            # Stop if we've used all available departures
            if not available_departures:
                break
        
        return solution
    
    def evaporate_and_deposit_pheromones(self, iteration_solutions, iteration_makespans, orders, departures):
        """Evaporate pheromones and deposit new pheromones based on the solutions of the current iteration"""
        # Evaporate pheromones
        self.evaporate_pheromones()
        
        # Deposit pheromones based on solution quality
        for ant in range(self.n_ants):
            self.deposit_pheromones(iteration_solutions[ant], iteration_makespans[ant], orders, departures)
    
    def evaporate_pheromones(self):
        """Reduce all pheromone levels by the evaporation rate"""
        self.pheromone *= (1 - self.rho)

    def deposit_pheromones(self, solution, makespan, orders, departures):
        """
        Deposit pheromones on the paths used in the solution.
        Better solutions (lower makespan) receive more pheromone.
        
        Parameters:
        -----------
        solution : dict
            Mapping of orders to departures
        makespan : float
            Maximum zone time for this solution
        orders : list
            List of all orders
        departures : list
            List of all departures
        """
        # Calculate pheromone deposit (better solutions get more pheromone)
        deposit = self.q / makespan if makespan > 0 else self.q
        
        # Deposit pheromones on paths used in this solution
        for order, departure in solution.items():
            order_idx = orders.index(order)
            departure_idx = departures.index(departure)
            self.pheromone[order_idx, departure_idx] += deposit

    def run_ant_colony(self, verbose=True):
        """
        Run the ACO algorithm with additional features like reporting.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print progress information
            
        Returns:
        --------
        best_solution : dict
            The best solution found
        """
        # Get baseline makespan
        baseline_zone_times = self.ptl.get_parameter("zone_times")
        baseline_makespan = max(baseline_zone_times)

        if verbose:
            print("----------------------------------")
            print("Running Ant Colony Optimization...")
            print(f"Baseline solution maximum of zone time: {baseline_makespan:.2f}")
        
        # Run ACO
        best_solution, best_makespan = self.ant_colony_optimization()
        
        if verbose:
            print(f"ACO solution maximum of zone time: {best_makespan:.2f}")
            improvement = (baseline_makespan - best_makespan) / baseline_makespan * 100
            print(f"Improvement: {improvement:.2f}%")

        return best_solution
