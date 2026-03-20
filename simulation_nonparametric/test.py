import pandas as pd
import glob
import os

# 1. Set your base path and use wildcards to match all result files
# This will match n200_results.csv, n500_results.csv, etc. in folders such as group111111_nested_dims, group132211_nested_dims, etc.base_path = '/home/wsh/simulationby/result/*/*.csv'
file_paths = glob.glob(base_path)

if not file_paths:
    print("No CSV file found, please check whether the path is correct!")
else:
    print(f"A total of {len(file_paths)} CSV files found, reading and merging...")

    # 2. Read and merge all data
    # Use list comprehensions to quickly read and merge into a total DataFrame
    df_list = [pd.read_csv(file) for file in file_paths]
    full_df = pd.concat(df_list, ignore_index=True)

    # 3. Process pmr_ci_cover and convert it to a Boolean value to calculate the proportion of true
    # astype(str).str.lower() is compatible with various situations such as 'True', 'true', True, 1, etc. in the file
    # Note: If the original value is 1/0, it can be changed to == '1'
    full_df['pmr_ci_cover_bool'] = full_df['pmr_ci_cover'].astype(str).str.lower().isin(['true', '1'])

    # 4. Define the columns and calculation methods that require statistics
    agg_funcs = {
        'por_mse': ['mean', 'std'],
        'pipw_mse': ['mean', 'std'],
        'phe1_mse': ['mean', 'std'],
        'phe2_mse': ['mean', 'std'],
        'pmr_mse': ['mean', 'std'],
        'pmr_ci_cover_bool': ['mean'] # The mean of True(1) is the coverage (proportion)
    }

    # 5. Group by group and sample_size and perform aggregation calculations
    summary_table = full_df.groupby(['group', 'sample_size']).agg(agg_funcs).reset_index()

    # 6. Organize the table header to make it flatter and easier to read.
    # The original header will be multi-level, such as ('por_mse', 'mean'), we will combine it into 'por_mse_mean'
    new_columns = []
    for col in summary_table.columns:
        if col[1] == '':
            new_columns.append(col[0])
        else:
            new_columns.append(f"{col[0]}_{col[1]}")
    summary_table.columns = new_columns

    # 7. Rename the coverage column name
    summary_table.rename(columns={'pmr_ci_cover_bool_mean': 'pmr_ci_cover_prop'}, inplace=True)

    # 8. Sorting (optional): To make the table look better, sort it in ascending order according to group and sample_size
    summary_table = summary_table.sort_values(by=['group', 'sample_size']).reset_index(drop=True)

    # 9. Print the result and save it
    print("\n======== Statistical result preview ========")
    print(summary_table.head(10)) # Preview 10 lines before printing
    
    # Save to the same directory as the result folder
    output_file = '/home/wsh/simulationby/simulation_summary_table.csv'
    summary_table.to_csv(output_file, index=False)
    print(f"\nAll statistics completed! The final table has been saved to: {output_file}")