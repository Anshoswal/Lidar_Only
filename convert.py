import pandas as pd

def process_data(input_file, output_file):
    """
    Reads a CSV file, filters out clusters with less than 8 points,
    sorts the remaining data by frame_id and cluster_id, and writes
    a new CSV with a blank line inserted after each cluster group.
    """
    try:
        # Load the dataset from the specified input file
        df = pd.read_csv(input_file)
        print(f"Successfully loaded '{input_file}'.")

        # --- Data Cleaning ---
        # Drop rows with any missing values to ensure data integrity
        original_rows = len(df)
        df.dropna(inplace=True)
        print(f"Dropped {original_rows - len(df)} rows with missing values.")

        # Convert IDs to integer types for accurate sorting and comparison
        df['frame_id'] = df['frame_id'].astype(int)
        df['cluster_id'] = df['cluster_id'].astype(int)

        # --- Filtering ---
        # Group by frame and cluster, and then filter out any groups that have fewer than 8 members.
        rows_before_filter = len(df)
        df_filtered = df.groupby(['frame_id', 'cluster_id']).filter(lambda x: len(x) >= 5)
        print(f"Discarded {rows_before_filter - len(df_filtered)} rows from clusters with less than 8 points.")

        # Sort the filtered dataframe first by frame_id, then by cluster_id
        df_sorted = df_filtered.sort_values(by=['frame_id', 'cluster_id'])
        print("Filtered data sorted by frame_id and cluster_id.")

        # --- Process and Write Output ---
        with open(output_file, 'w', newline='') as f_out:
            # Write the header row to the output file
            f_out.write(','.join(df_sorted.columns) + '\n')

            # Initialize trackers for the previous row's IDs
            last_frame_id = None
            last_cluster_id = None

            # Iterate over each row in the sorted dataframe
            for index, row in df_sorted.iterrows():
                current_frame_id = row['frame_id']
                current_cluster_id = row['cluster_id']

                # Check if this is the start of a new cluster
                # A new cluster starts if it's the very first row, or if the frame/cluster ID changes.
                is_new_cluster = (last_frame_id is None) or \
                                 (current_frame_id != last_frame_id) or \
                                 (current_cluster_id != last_cluster_id)

                # If it's a new cluster (and not the very first line), write a blank line before it
                if not (last_frame_id is None) and is_new_cluster:
                    f_out.write('\n')

                # Write the current row's data to the file
                f_out.write(','.join(map(str, row.values)) + '\n')

                # Update the trackers to the current row's IDs
                last_frame_id = current_frame_id
                last_cluster_id = current_cluster_id
        
        print(f"Processing complete. Filtered output saved to '{output_file}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Main execution ---
if __name__ == "__main__":
    # Define the input and output filenames
    input_csv = 'cone_detections_2025.csv'
    output_csv = 'filteredd_data.csv'
    
    # Run the processing function
    process_data(input_csv, output_csv)

