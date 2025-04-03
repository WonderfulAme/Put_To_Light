import data as dt
import greedy as gd
import matplotlib.pyplot as plt
import genetic as gen
import time
import pandas as pd
import scipy.stats as stats

if __name__ == '__main__':
    
    for i in range(5, 6):
        ptl = dt.PTL(option = i + 1)
        file_name = ptl.get_parameter("file_name")[:-4]
        ptl.load_data()
        ptl.compute_order_times()
        ptl.compute_total_zone_time()

        greedy_start_time = time.time()
        greedy_heuristic = gd.Greedy(ptl)
        greedy_heuristic.run_greedy_heuristic(verbose=True)
        greedy_time = time.time() - greedy_start_time
        greedy_zone_times = ptl.get_parameter("zone_times")
        if not ptl.check_solution():
            print("Invalid solution found, stopping iterations.")
            break
        
        instance_results = []
        
        for j in range(0, 5):  # 30 iterations for statistical significance
            iteration_start_time = time.time()
            genetic = gen.Genetic(greedy_heuristic.ptl, initial_size=1)
            solution = genetic.genetic_algorithm()
            if not ptl.check_solution():
                print("Invalid solution found, stopping iterations.")
                break
            iteration_time = time.time() - iteration_start_time
            
            zone_times = solution.get_parameter("zone_times")
            improvement = max(greedy_zone_times) - max(zone_times)
            
            if j == 0:
                # Save the best solution found in the first iteration
                solution.save_solution(f"sol/Solution_Genetic_{file_name}.xlsx")

            instance_results.append({
                'problem_instance': i + 1,
                'file_name': file_name,
                'iteration': j+1,
                'greedy_time': greedy_time,
                'genetic_time': iteration_time,
                'genetic_max_zone_time': max(zone_times),
                'greedy_max_zone_times': max(greedy_zone_times),
                'greedy_difference': max(greedy_zone_times) - min(greedy_zone_times),
                'genetic_difference': max(zone_times) - min(zone_times), 
                'improvement': improvement,
            })

        instance_results_df = pd.DataFrame(instance_results)
        
        # Compute confidence intervals for max_zone_time
        confidence = 0.95
        mean_max_zone_time = instance_results_df['genetic_max_zone_time'].mean()
        std_max_zone_time = instance_results_df['genetic_max_zone_time'].std()
        n = len(instance_results_df['genetic_max_zone_time'])
        conf_interval_max_zone_time = stats.t.interval(confidence, n-1, loc=mean_max_zone_time, scale=std_max_zone_time / (n ** 0.5))

        # Compute confidence intervals for improvement
        mean_improvement = instance_results_df['improvement'].mean()
        std_improvement = instance_results_df['improvement'].std()
        conf_interval_improvement = stats.t.interval(confidence, n-1, loc=mean_improvement, scale=std_improvement / (n ** 0.5))

        # Compute confidence intervals for difference
        mean_difference = instance_results_df['genetic_difference'].mean()
        std_difference = instance_results_df['genetic_difference'].std()
        conf_interval_difference = stats.t.interval(confidence, n-1, loc=mean_difference, scale=std_difference / (n ** 0.5))
        
        instance_results_df.to_excel(f"stats/Instance_Results_{file_name}.xlsx", index=False)
        
