import pandas as pd
import csv
import time

def convert_csv_to_grouped_csv(input_filepath, output_filepath):
    """
    Reads a flat CSV, groups it by frame and cluster, sorts the points
    within each group by height, and saves it to a new CSV file.

    Args:
        input_filepath (str): The path to the source CSV file.
        output_filepath (str): The path where the grouped CSV file will be saved.
    """
    print(f"Reading data from '{input_filepath}'...")
    start_time = time.time()
    
    try:
        df = pd.read_csv(input_filepath)
        print(f"Read {len(df):,} rows in {time.time() - start_time:.2f} seconds.")

        print(f"Restructuring and sorting data, then writing to '{output_filepath}'...")
        
        # Open the destination file to write the new format
        with open(output_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Group by frame and cluster, then iterate through each group
            for (frame_id, cluster_id), group in df.groupby(['frame_id', 'cluster_id']):
                
                # Write a single row to identify the frame and cluster
                writer.writerow([f'frame_id', frame_id, 'cluster_id', cluster_id])
                
                # Write the header for the point data that follows
                writer.writerow(['radial_distance', 'height', 'normalized_intensity'])
                
                # <<< NEW >>> Sort the group by the 'height' column
                sorted_group = group.sort_values('height')
                
                # Write the data for each point in the *sorted* group
                for _, row in sorted_group.iterrows():
                    writer.writerow([
                        row['radial_distance'],
                        row['height'],
                        row['normalized_intensity']
                    ])
                
                # Write a blank line to separate groups
                writer.writerow([])

        end_time = time.time()
        print(f"✅ Success! Process finished in {end_time - start_time:.2f} seconds.")

    except FileNotFoundError:
        print(f"❌ Error: The file '{input_filepath}' was not found.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == '__main__':
    # --- CONFIGURE YOUR FILE PATHS HERE ---
    source_file = 'cone_profile_2D.csv'
    destination_file = 'grouped_and_sorted_cone_data.csv'

    convert_csv_to_grouped_csv(source_file, destination_file)

