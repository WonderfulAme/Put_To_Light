import numpy as np
class Greedy:
    def __init__(self, ptl):
        self.ptl = ptl

    """
    Constructs an initial solution for the order-to-zone assignment problem using a greedy heuristic.
    
    This function assigns orders to departure zones by:
    1. Sorting orders by the number of SKUs (highest to lowest)
    2. Sorting zones by their current total processing time (lowest to highest)
    3. Assigning the largest orders to the least busy zones
    
    This greedy approach aims to balance the workload across zones while prioritizing 
    large orders that would have the most impact on zone processing times.
    
    Parameters:
    -----------
    ptl : object
        Problem instance object that contains:
        - Methods: get_parameter(), compute_order_times(), compute_total_departure_time()
        - Attributes: _solution (dictionary to store order-to-zone assignments)
        
    Returns:
    --------
    None
        The solution is stored directly in the ptl._solution attribute
    """
    def greedy_heuristic(self):
        # Retrieve the dictionary mapping order IDs to the number of SKUs they contain
        n_skus_per_order = self.ptl.get_parameter("n_skus_per_order")

        # Sort orders in descending order by their SKU count (largest orders first)
        list_of_orders = list(n_skus_per_order.keys())
        list_of_orders.sort(key=lambda x: n_skus_per_order[x], reverse=True)
        
        # Calculate the total processing time for each zone by summing across all orders
        total_departure_time = self.ptl.get_parameter("total_departure_time")
        
        # Sort zones in ascending order by their current total processing time (least busy zones first)
        total_departure_time_sorted = list(total_departure_time.keys())
        total_departure_time_sorted.sort(key=lambda x: total_departure_time[x])

        # Assign each order to a zone, matching the largest orders with the least busy zones
        for i in range(len(list_of_orders)):
                self.ptl._solution[list_of_orders[i]] = total_departure_time_sorted[i]
        
        # Update solution metrics after assignment
        self.ptl.compute_order_times()
        self.ptl.compute_total_zone_time()
        return self.ptl._solution

    def run_greedy_heuristic(self, verbose = True):
        if verbose:
            print("----------------------------")
            print("Running greedy heuristic...")
        self.greedy_heuristic()
        if verbose:
            print("Total zone time after greedy heuristic:")
            total_zone_time = self.ptl.get_parameter("zone_times")
            for i in range(len(total_zone_time)):
                print(f"Zone {i + 1}: {total_zone_time[i]:.2f}")
        
         


