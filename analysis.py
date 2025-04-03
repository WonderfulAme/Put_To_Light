import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
# Since the exact Excel file isn't accessible, I'll create example code assuming the data structure
# You'll need to adjust the file path to your actual Excel file
def load_data(file_path):
    # Try reading the data
    df = pd.read_excel(file_path)


    return df

def analyze_data(df):
    # Group by problem instance if there are multiple instances
    if 'problem_instance' in df.columns and len(df['problem_instance'].unique()) > 1:
        grouped = df.groupby('problem_instance')
    else:
        # If only one instance or no instance column, analyze all data together
        grouped = {0: df}
    
    results = []
    
    for group_name, group_data in grouped.items():
        # Calculate required metrics
        greedy_time = group_data['greedy_time'].mean()
        genetic_time = group_data['genetic_time'].mean()
        gmzt_genetic = group_data['genetic_max_zone_time'].mean()
        gmzt_greedy = group_data['greedy_max_zone_times'].mean()
        gd_genetic = group_data['genetic_difference'].mean()
        gd_greedy = group_data['greedy_difference'].mean()
        improvement = group_data['improvement'].mean() if 'improvement' in group_data.columns else (gmzt_greedy - gmzt_genetic)
        
        print(f"Group: {group_name}, Greedy Time: {greedy_time}, Genetic Time: {genetic_time}, GMZT Genetic: {gmzt_genetic}, GMZT Greedy: {gmzt_greedy}, GD Genetic: {gd_genetic}, GD Greedy: {gd_greedy}, Improvement: {improvement}")
        # Calculate confidence intervals (95%)
        n = len(group_data)
        t_value = stats.t.ppf(0.975, n-1)  # Two-tailed 95% confidence interval
        
        time_genetic_ci = t_value * (group_data['genetic_time'].std() / np.sqrt(n))
        time_greedy_ci = t_value * (group_data['greedy_time'].std() / np.sqrt(n))
        gmzt_genetic_ci = t_value * (group_data['genetic_max_zone_time'].std() / np.sqrt(n))
        gmzt_greedy_ci = t_value * (group_data['greedy_max_zone_times'].std() / np.sqrt(n))
        gd_genetic_ci = t_value * (group_data['genetic_difference'].std() / np.sqrt(n))
        gd_greedy_ci = t_value * (group_data['greedy_difference'].std() / np.sqrt(n))
        gd_genetic_ci = t_value * (group_data['genetic_difference'].std() / np.sqrt(n))
        improvement_ci = t_value * (group_data['improvement'].std() / np.sqrt(n)) if 'improvement' in group_data.columns else None


        # Hypothesis testing
        # H0: GMZT_genetic <= GMZT_greedy, H1: GMZT_genetic > GMZT_greedy
        t_stat_gmzt, p_value_gmzt = stats.ttest_rel(
            group_data['genetic_max_zone_time'], 
            group_data['greedy_max_zone_times'], 
            alternative='greater'
        )
        reject_h0_gmzt = p_value_gmzt < 0.05  # one-tailed test

        print(f"t_stat_gmzt: {t_stat_gmzt}, p_value_gmzt: {p_value_gmzt}")

        # H0: GD_genetic <= GD_greedy, H1: GD_genetic > GD_greedy
        t_stat_gd, p_value_gd = stats.ttest_rel(
            group_data['genetic_difference'], 
            group_data['greedy_difference'], 
            alternative='greater'
        )
        reject_h0_gd = p_value_gd < 0.05  # one-tailed test

        print(f"t_stat_gd: {t_stat_gd}, p_value_gd: {p_value_gd}")

        # H0: Imp >= 0, H1: Imp < 0
        t_stat_imp, p_value_imp = stats.ttest_1samp(
            group_data['improvement'], 
            0, 
            alternative='less'
        )
        reject_h0_imp = p_value_imp < 0.05  # one-tailed test

        print(f"t_stat_imp: {t_stat_imp}, p_value_imp: {p_value_imp}")

        
        # Compile results
        instance_results = {
            'Instance': group_name if isinstance(group_name, (int, str)) else 'All',
            'Greedy_Time': greedy_time,
            'Greedy_Time_CI': time_greedy_ci,
            'Genetic_Time': genetic_time,
            'Genetic_Time_CI': time_genetic_ci,
            'GMZT_Genetic': gmzt_genetic,
            'GMZT_Genetic_CI': gmzt_genetic_ci,
            'GMZT_Greedy': gmzt_greedy,
            'GMZT_Greedy_CI': gmzt_greedy_ci,
            'GD_Genetic': gd_genetic,
            'GD_Genetic_CI': gd_genetic_ci,
            'GD_Greedy': gd_greedy,
            'GD_Greedy_CI': gd_greedy_ci,
            'Improvement': improvement,
            'Improvement_CI': improvement_ci,
            'Reject_H0_GMZT': reject_h0_gmzt,
            'Reject_H0_GD': reject_h0_gd,
            'Reject_H0_Imp': reject_h0_imp,
            'p_value_GMZT': p_value_gmzt/2 if t_stat_gmzt > 0 else p_value_gmzt,
            'p_value_GD': p_value_gd/2 if t_stat_gd > 0 else p_value_gd,
            'p_value_Imp': p_value_imp/2 if t_stat_imp > 0 else p_value_imp
        }
        
        results.append(instance_results)
    
    return pd.DataFrame(results)

def create_summary_table(results_df):
    # Create a formatted summary table
    summary_table = results_df.copy()
    
    # Format the numeric columns
    for col in ['GMZT_Genetic', 'GMZT_Greedy', 'GD_Genetic', 'GD_Greedy', 'Improvement']:
        if col in summary_table.columns:
            summary_table[col] = summary_table[col].round(2)
    
    # Add confidence intervals
    for col in ['GMZT_Genetic', 'GMZT_Greedy', 'GD_Genetic', 'GD_Greedy', 'Improvement']:
        if col in summary_table.columns and f'{col}_CI' in summary_table.columns:
            ci_col = f'{col}_CI'
            summary_table[col] = summary_table.apply(
                lambda row: f"{row[col]} ± {row[ci_col]:.2f}", axis=1
            )
    
    # Add hypothesis test results
    for hypothesis in ['GMZT', 'GD', 'Imp']:
        col_name = f'Reject_H0_{hypothesis}'
        if col_name in summary_table.columns:
            p_value_col = f'p_value_{hypothesis}'
            if p_value_col in summary_table.columns:
                summary_table[col_name] = summary_table.apply(
                    lambda row: f"Yes (p={row[p_value_col]:.4f})" if row[col_name] else "No", axis=1
                )
    
    # Select only the relevant columns for display
    display_cols = ['Instance', 'GMZT_Genetic', 'GMZT_Greedy', 'GD_Genetic', 'GD_Greedy', 
                   'Improvement', 'Reject_H0_GMZT', 'Reject_H0_GD', 'Reject_H0_Imp']
    
    return summary_table[display_cols]

def main():
    # Replace with your actual file path
    file_path = "stats/Instance_Results_Data_80_Salidas_composicion_zonas_heterogeneas.xlsx"
    
    # Load data
    print("Loading data...")
    df = load_data(file_path)
    
    # Analyze data
    print("Analyzing data...")
    results_df = analyze_data(df)
    
    # Create summary table
    print("Creating summary table...")
    summary_table = create_summary_table(results_df)

    # Display summary table
    print("\nSummary of Results:")
    print(summary_table.to_string())

if __name__ == "__main__":
    main()