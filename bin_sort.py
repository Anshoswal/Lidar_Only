import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('FINAL DATA - Sheet1.csv')

# Remove rows with any missing values
df.dropna(inplace=True)

# --- NEW: Group by both frame_id and cluster_id ---
# The primary grouping keys for the analysis
grouping_keys = ['frame_id', 'cluster_id']

# Determine the classification for each unique (frame, cluster) pair
# This creates a MultiIndex Series for accurate mapping later
pair_classification = df.groupby(grouping_keys)['classification'].apply(lambda x: x.mode()[0])

# Define the 100 bins for the z_coordinate as before
min_z = df['z_coordinate'].min()
max_z = df['z_coordinate'].max()
bin_edges = np.linspace(min_z, max_z, 51)

# Assign each data point to a bin
df['bin'] = pd.cut(df['z_coordinate'], bins=bin_edges, labels=False, include_lowest=True)

# --- MODIFIED: Group by frame, cluster, AND bin ---
# Calculate the mean intensity for each group
grouped = df.groupby(grouping_keys + ['bin'])['normalized_intensity'].mean()

# --- MORE EFFICIENT: Use unstack to pivot the bins into columns ---
# This creates the final structure with bin columns and fills missing values with -1
output_df = grouped.unstack(level='bin', fill_value=-1)

# Rename columns to have the 'bin_' prefix
output_df.columns = [f'bin_{col}' for col in output_df.columns]

# Add the 'classification' column to the output DataFrame
output_df['classification'] = pair_classification

# Reorder columns to make 'classification' the first column
cols = ['classification'] + [col for col in output_df if col != 'classification']
output_df = output_df[cols]

# Save the final, correctly structured DataFrame to a new CSV file
output_df.to_csv('frame_cluster_intensity.csv')

# Display a sample of the final output
print("Sample of the final output, grouped by frame and cluster:")
print(output_df.head())