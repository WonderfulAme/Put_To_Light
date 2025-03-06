import data as dt
import greedy as gd
import AntColony as ant
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    for i in range(6):
        ptl = dt.PTL(option = i + 1)
        file_name = ptl.get_parameter("file_name")[:-4]
        ptl.load_data()
        ptl.compute_order_times()
        ptl.compute_total_zone_time()

        greedy_heuristic = gd.Greedy(ptl)
        greedy_heuristic.run_greedy_heuristic(verbose=True)
        greedy_zone_times = ptl.get_parameter("zone_times")
        ptl.save_solution(f'sol/Solucion_Greedy_{file_name}.xlsx')


        ant_colony = ant.AntColony(ptl, n_ants=10, n_iterations=100, alpha=1.0, beta=2.0, rho=0.1, q=1.0)
        ant_colony.run_ant_colony(verbose=True)
        ant_zone_times = ptl.get_parameter("zone_times")
        ptl.save_solution(f'sol//Solucion_AntColony_{file_name}.xlsx')

        # Make and save a multi bar graph comparing the zone times, two bars per zone
        n_zones = len(greedy_zone_times)
        x = range(n_zones)
        width = 0.35
        plt.bar(x, greedy_zone_times, width, label='Greedy')
        plt.bar([i + width for i in x], ant_zone_times, width, label='Ant Colony')
        plt.xlabel('Zone')
        plt.ylabel('Total Time')
        plt.title(f'Greedy vs Ant Colony: {i}')
        plt.xticks([i + width/2 for i in x], [f'Zone {i + 1}' for i in x])
        plt.legend()
        plt.savefig(f'figs//Grafico_{file_name}.png')
        plt.close()
        


