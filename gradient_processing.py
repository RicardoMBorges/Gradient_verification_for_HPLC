# gradient_processing.py

import os
import pandas as pd
import glob
import numpy as np
import plotly.graph_objs as go

### IMPORT DATA 
def extract_data(file_path):
    data = []
    start_extraction = False

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("R.Time (min)"):
                start_extraction = True
                continue
            if start_extraction:
                columns = line.strip().split()
                if len(columns) == 2:
                    # Replace commas with dots in each column
                    columns = [col.replace(',', '.') for col in columns]
                    data.append(columns)

    return data

def combine_and_trim_data(input_folder, output_folder, retention_time_start, retention_time_end):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_folder, file_name)
            data = extract_data(file_path)

            # Save the data into a new file
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_table.csv")
            with open(output_file_path, 'w') as output_file:
                for row in data:
                    output_file.write('\t'.join(row) + '\n')

    # Get a list of all files matching the pattern *_table.csv
    file_list = glob.glob(os.path.join(output_folder, '*_table.csv'))

    # Initialize an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()

    # Loop through each file and read its data into a DataFrame
    for file in file_list:
        column_name = os.path.basename(file).split('_table.csv')[0]
        df = pd.read_csv(file, delimiter='\t', header=None)
        combined_df[column_name] = df.iloc[:, 1]

    # Concatenate 'axis' DataFrame with 'combined_df'
    axis = df.iloc[:, 0]
    combined_df2 = pd.concat([axis, combined_df], axis=1)
    combined_df2.rename(columns={0: "RT(min)"}, inplace=True)

    # Select and trim the data range
    start_index = (combined_df2["RT(min)"] - retention_time_start).abs().idxmin()
    end_index = (combined_df2["RT(min)"] - retention_time_end).abs().idxmin()
    combined_df2 = combined_df2.loc[start_index:end_index].copy()

    # Save the combined DataFrame to a CSV file
    if not os.path.exists('data'):     # Rename the folder name to your specific case. Keep it organized.
        os.mkdir('data')
    combined_df2.to_csv(os.path.join(output_folder, "combined_data.csv"), sep=";", index=False)

    return combined_df2

# Example usage
# input_folder = 'path_to_input_folder'
# output_folder = 'path_to_output_folder'
# retention_time_start = 2
# retention_time_end = 30
# combined_df2 = combine_and_trim_data(input_folder, output_folder, retention_time_start, retention_time_end)



### REMOVE UNWANTED REGIONS
def remove_unwanted_regions(df, start_value, end_value):
    """
    Remove unwanted regions by substituting sample values with zeros between specified 
    start and end values in the RT(min) axis.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    start_value (float/int): Starting value of the RT(min) range.
    end_value (float/int): Ending value of the RT(min) range.

    Returns:
    pd.DataFrame: DataFrame with substituted values.
    
    # Example usage
    start = 2  # Starting value of RT(min) for substitution
    end = 5    # Ending value of RT(min) for substitution
    modified_df = remove_unwanted_regions(combined_df.copy(), start, end)
    """
    # Identify the rows where RT(min) is within the specified range
    rows_to_substitute = df['RT(min)'].between(start_value, end_value)

    # Columns to be modified (excluding RT(min))
    sample_columns = [col for col in df.columns if col != 'RT(min)']

    # Substitute values with zeros in the specified range for all sample columns
    df.loc[rows_to_substitute, sample_columns] = 0

    return df
##

## 
def volumns_HPLC(gradient_segments, flow_rate_mL_min):
    # Initialize total volumes for A and B
    total_volume_A = 0
    total_volume_B = 0

    # Calculate volumes for each segment
    for start_time, end_time, start_percentage_B, end_percentage_B, in gradient_segments:
        # Duration of segment in minutes
        duration_min = end_time - start_time
    
        # Average percentage of B in this segment
        average_percentage_B = (start_percentage_B + end_percentage_B) / 2
    
        # Total volume for this segment
        segment_volume_mL = flow_rate_mL_min * duration_min
    
        # Volume of B in this segment
        segment_volume_B_mL = segment_volume_mL * (average_percentage_B / 100)
    
        # Volume of A in this segment
        segment_volume_A_mL = segment_volume_mL - segment_volume_B_mL
    
        # Add to total volumes
        total_volume_A += segment_volume_A_mL
        total_volume_B += segment_volume_B_mL
    
    print(f"Total volume of Solvent A: {total_volume_A:.1f} mL")
    print(f"Total volume of Solvent B: {total_volume_B:.1f} mL")
    print(f"                           Total volume : {total_volume_A + total_volume_B:.1f} mL")

    #return total_volume_A, total_volume_B
    ##
    
##
def gradient_plot(gradient):
    # Prepare data for plotting
    times_list = []
    percentages_B_list = []

    for index, row in gradient.iterrows():
        if index == 0 or (index > 0 and gradient.loc[index-1, 'end_time'] != row['start_time']):
            times_list.append(row['start_time'])
            percentages_B_list.append(row['start_B%'])
        times_list.append(row['end_time'])
        percentages_B_list.append(row['end_B%'])

    times_array = np.array(times_list)
    percentages_B_array = np.array(percentages_B_list)

    df_gradient_corrected = pd.DataFrame({
        'Time (minutes)': times_array,
        'Percentage of Solvent B (%)': percentages_B_array
    })
    # Interpolate to create denser data points for a smoother line
    # Let's assume you want to interpolate such that there's a point every 0.1 minute
    time_range = np.arange(gradient['start_time'].min(), gradient['end_time'].max(), 0.1)
    percentage_B_interpolated = np.interp(time_range, times_array, percentages_B_array)
    
    # Create a new DataFrame for the interpolated data
    df_interpolated = pd.DataFrame({
        'Time (minutes)': time_range,
        'Percentage of Solvent B (%)': percentage_B_interpolated
    })

    # Create the Plotly figure
    fig = go.Figure()
    
    # Add the line trace
    fig.add_trace(go.Scatter(x=df_interpolated['Time (minutes)'], y=df_interpolated['Percentage of Solvent B (%)'],
                            mode='lines', name='Solvent B %'))
    
    # Enhance hover data
    fig.update_traces(hoverinfo='x+y', hovertemplate='Time: %{x} min<br>Percentage of B: %{y}%')

    # Update layout for better readability
    fig.update_layout(title='Interactive Gradient Profile of Solvent B Over Time',
                    xaxis_title='Time (minutes)',
                    yaxis_title='Percentage of Solvent B (%)',
                    template='plotly_white')
    
    # Display the figure
    fig.show()

##



def plot_gradient_and_chromatogram(gradient, directory_path, start_column, end_column, retention_time_start, retention_time_end):
    import os
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import gradient_processing as gp  # Assuming 'gp' has 'combine_and_trim_data'
    # Set the working directory to where your data is
    os.chdir(directory_path)

    input_folder = directory_path
    output_folder = os.path.join(input_folder, 'data')
    
    # Combine and trim chromatogram data
    combined_df2 = gp.combine_and_trim_data(input_folder, output_folder, retention_time_start, retention_time_end)

    # Create the Plotly figure with subplots and a secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot chromatograms on the primary y-axis
    for column in combined_df2.columns[start_column:end_column + 1]:
        fig.add_trace(go.Scatter(x=combined_df2['RT(min)'], y=combined_df2[column], mode='lines', name=column), secondary_y=False)

    # Prepare data for plotting the gradient
    times_list = []
    percentages_B_list = []

    for index, row in gradient.iterrows():
        if index == 0 or (index > 0 and gradient.loc[index-1, 'end_time'] != row['start_time']):
            times_list.append(row['start_time'])
            percentages_B_list.append(row['start_B%'])
        times_list.append(row['end_time'])
        percentages_B_list.append(row['end_B%'])

    times_array = np.array(times_list)
    percentages_B_array = np.array(percentages_B_list)
    
    # Interpolate gradient data
    time_range = np.arange(gradient['start_time'].min(), gradient['end_time'].max(), 0.1)
    percentage_B_interpolated = np.interp(time_range, times_array, percentages_B_array)
    
    # Add gradient trace on a secondary y-axis
    fig.add_trace(go.Scatter(x=time_range, y=percentage_B_interpolated, mode='lines', name='Gradient of Solvent B (%)',
                             line=dict(color='red', dash='dot')), secondary_y=True)

    # Configure plot layout with dual y-axes
    fig.update_layout(
        title='Chromatogram and Gradient Profile Over Time',
        xaxis_title='RT (min)',
        yaxis_title='Chromatogram Intensity',
        legend_title='Samples / Gradient',
        hovermode='closest'
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text='Chromatogram Intensity', secondary_y=False)
    fig.update_yaxes(title_text='Percentage of Solvent B (%)', secondary_y=True)
    
    # Display the figure
    fig.show()

# Example call (make sure you have a 'gradient' DataFrame ready and adjust paths)
# plot_gradient_and_chromatogram(gradient_df, r'C:\Users\borge\Desktop\HPLC DATA ARISTOLOCHIA_ANDREW', 1, 1, 4, 30)