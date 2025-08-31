# # # import rclpy
# # # from rclpy.node import Node
# # # from sklearn.cluster import DBSCAN
# # # from rclpy.duration import Duration
# # # from sklearn.linear_model import RANSACRegressor
# # # import numpy as np
# # # from sensor_msgs.msg import PointCloud
# # # import csv

# # # class ClusterProjectorAndLogger(Node):
# # #     def __init__(self):
# # #         super().__init__('cluster_projector_and_logger')
# # #         self.subscriber = self.create_subscription(PointCloud, '/carmaker/pointcloud', self.process_cloud_callback, 10)
# # #         self.frame_count = 0
        
# # #         self.csv_file = open("projected_points.csv", "w", newline="")
# # #         self.csv_writer = csv.writer(self.csv_file)
# # #         self.csv_writer.writerow(["frame", "cluster_id", "proj_x", "proj_y", "proj_z", "intensity"])
# # #         self.get_logger().info("Node has started. Logging projected points to projected_points.csv")

# # #     def __del__(self):
# # #         """Ensure the CSV file is closed properly on shutdown."""
# # #         if hasattr(self, "csv_file") and not self.csv_file.closed:
# # #             self.csv_file.close()

# # #     def remove_ground_ransac(self, pc_data):
# # #         """Removes the ground plane from point cloud data using RANSAC."""
# # #         # RANSAC works on the XYZ coordinates, but the mask is applied to the full data
# # #         xyz_data = pc_data[:, :3]
# # #         if xyz_data.shape[0] < 10:
# # #             return pc_data
        
# # #         try:
# # #             ransac = RANSACRegressor(residual_threshold=0.05)
# # #             ransac.fit(xyz_data[:, :2], xyz_data[:, 2])
# # #             # Return the original data (with intensity) for non-ground points
# # #             return pc_data[~ransac.inlier_mask_]
        
# # #         except ValueError as e:
# # #             self.get_logger().warn(f"RANSAC failed: {e}. Returning no points.")
# # #             return np.array([])

# # #     def process_cloud_callback(self, msg):
# # #         """Processes each point cloud, finds clusters, projects them, and logs to CSV."""
# # #         self.frame_count += 1
# # #         if not msg.points:
# # #             return

# # #         # Capture intensity along with XYZ coordinates
# # #         pc_data_xyz = np.array([[p.x, p.y, p.z] for p in msg.points])
# # #         if msg.channels and msg.channels[0].name.lower() == 'intensity':
# # #             intensities = np.array(msg.channels[0].values).reshape(-1, 1)
# # #             pc_data = np.hstack((pc_data_xyz, intensities))
# # #         else:
# # #             pc_data = np.hstack((pc_data_xyz, np.zeros((pc_data_xyz.shape[0], 1))))

# # #         pc_data_no_ground = self.remove_ground_ransac(pc_data)
        
# # #         if pc_data_no_ground.shape[0] < 5:
# # #             return

# # #         # DBSCAN still only uses XY for clustering
# # #         labels = DBSCAN(eps=0.5, min_samples=2).fit_predict(pc_data_no_ground[:, :2])

# # #         unique_labels = set(labels)
# # #         if -1 in unique_labels:
# # #             unique_labels.remove(-1)

# # #         if not unique_labels:
# # #             return

# # #         self.get_logger().info(f"Frame {self.frame_count}: Found {len(unique_labels)} non-ground clusters")

# # #         for label in unique_labels:
# # #             # cluster_points now has 4 columns: [x, y, z, intensity]
# # #             cluster_points = pc_data_no_ground[labels == label]
            
# # #             # Find closest point using only XYZ coordinates
# # #             squared_distances = np.sum(cluster_points[:, :3]**2, axis=1)
# # #             closest_point_index = np.argmin(squared_distances)
# # #             closest_point = cluster_points[closest_point_index, :3] # Get only XYZ
            
# # #             plane_point = closest_point
# # #             plane_normal = np.array([closest_point[0], closest_point[1], 0.0])
            
# # #             norm = np.linalg.norm(plane_normal)
# # #             if norm < 1e-6:
# # #                 continue
# # #             plane_normal /= norm

# # #             # Project only the XYZ coordinates
# # #             vectors_to_project = cluster_points[:, :3] - plane_point
# # #             distances_from_plane = np.dot(vectors_to_project, plane_normal)
# # #             projected_points = cluster_points[:, :3] - np.outer(distances_from_plane, plane_normal)

# # #             for i in range(len(projected_points)):
# # #                 p = projected_points[i]
# # #                 intensity = cluster_points[i, 3] # Get intensity from the original point
# # #                 self.csv_writer.writerow([self.frame_count,label,f"{p[0]:.4f}",f"{p[1]:.4f}",f"{p[2]:.4f}",f"{intensity:.4f}"])

# # # def main(args=None):
# # #     rclpy.init(args=args)
# # #     node = ClusterProjectorAndLogger()
# # #     rclpy.spin(node)
# # #     node.destroy_node()
# # #     rclpy.shutdown()

# # # if __name__ == '__main__':
# # #     main()

# # import rclpy
# # from rclpy.node import Node
# # from sklearn.cluster import DBSCAN
# # from rclpy.duration import Duration
# # from sklearn.linear_model import RANSACRegressor
# # import numpy as np
# # from sensor_msgs.msg import PointCloud
# # import csv

# # class ClusterProjectorAndLogger(Node):
# #     def __init__(self):
# #         super().__init__('cluster_projector_and_logger')
# #         self.subscriber = self.create_subscription(PointCloud, '/carmaker/pointcloud', self.process_cloud_callback, 10)
# #         self.frame_count = 0
        
# #         # CSV for raw projected points
# #         self.points_csv_file = open("projected_points.csv", "w", newline="")
# #         self.points_csv_writer = csv.writer(self.points_csv_file)
# #         self.points_csv_writer.writerow(["frame", "cluster_id", "proj_x", "proj_y", "proj_z", "intensity"])
        
# #         # --- NEW: CSV for grid visualization ---
# #         self.grid_csv_file = open("projected_grids.csv", "w", newline="")
# #         self.grid_csv_writer = csv.writer(self.grid_csv_file)
# #         self.GRID_SIZE = 30 # Resolution of the grid

# #         self.get_logger().info("Node has started. Logging points and grids.")

# #     def __del__(self):
# #         """Ensure the CSV files are closed properly on shutdown."""
# #         if hasattr(self, "points_csv_file") and not self.points_csv_file.closed:
# #             self.points_csv_file.close()
# #         if hasattr(self, "grid_csv_file") and not self.grid_csv_file.closed:
# #             self.grid_csv_file.close()

# #     def remove_ground_ransac(self, pc_data):
# #         xyz_data = pc_data[:, :3]
# #         if xyz_data.shape[0] < 10: return pc_data
# #         try:
# #             ransac = RANSACRegressor(residual_threshold=0.05)
# #             ransac.fit(xyz_data[:, :2], xyz_data[:, 2])
# #             return pc_data[~ransac.inlier_mask_]
# #         except ValueError:
# #             return np.array([])

# #     def process_cloud_callback(self, msg):
# #         self.frame_count += 1
# #         if not msg.points: return

# #         pc_data_xyz = np.array([[p.x, p.y, p.z] for p in msg.points])
# #         if msg.channels and msg.channels[0].name.lower() == 'intensity':
# #             intensities = np.array(msg.channels[0].values).reshape(-1, 1)
# #             pc_data = np.hstack((pc_data_xyz, intensities))
# #         else:
# #             pc_data = np.hstack((pc_data_xyz, np.zeros((pc_data_xyz.shape[0], 1))))

# #         pc_data_no_ground = self.remove_ground_ransac(pc_data)
# #         if pc_data_no_ground.shape[0] < 5: return

# #         labels = DBSCAN(eps=0.5, min_samples=2).fit_predict(pc_data_no_ground[:, :2])
# #         unique_labels = set(labels)
# #         if -1 in unique_labels: unique_labels.remove(-1)
# #         if not unique_labels: return

# #         self.get_logger().info(f"--- Frame {self.frame_count}: Found {len(unique_labels)} non-ground clusters ---")

# #         for label in unique_labels:
# #             cluster_points = pc_data_no_ground[labels == label]
            
# #             squared_distances = np.sum(cluster_points[:, :3]**2, axis=1)
# #             closest_point_index = np.argmin(squared_distances)
# #             closest_point = cluster_points[closest_point_index, :3]
            
# #             plane_point = closest_point
# #             plane_normal = np.array([closest_point[0], closest_point[1], 0.0])
# #             norm = np.linalg.norm(plane_normal)
# #             if norm < 1e-6: continue
# #             plane_normal /= norm

# #             vectors_to_project = cluster_points[:, :3] - plane_point
# #             distances_from_plane = np.dot(vectors_to_project, plane_normal)
# #             projected_points = cluster_points[:, :3] - np.outer(distances_from_plane, plane_normal)
            
# #             for i in range(len(projected_points)):
# #                 p = projected_points[i]
# #                 intensity = cluster_points[i, 3]
# #                 self.points_csv_writer.writerow([self.frame_count, label, f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}", f"{intensity:.4f}"])

# #             # --- NEW: Call the function to create and save the grid ---
# #             self.create_and_save_grid(projected_points, cluster_points[:, 3], plane_point, plane_normal, label)

# #     # --- NEW: Function to create and save the grid visualization ---
# #     def create_and_save_grid(self, projected_points, intensities, plane_origin, plane_normal, cluster_label):
# #         """Converts projected points into a trimmed 2D grid of intensities and saves to CSV."""
# #         z_coords = projected_points[:, 2]
# #         tangent_vector = np.array([-plane_normal[1], plane_normal[0], 0])
# #         h_coords = np.dot(projected_points - plane_origin, tangent_vector)

# #         z_min, z_max = np.min(z_coords), np.max(z_coords)
# #         h_min, h_max = np.min(h_coords), np.max(h_coords)
# #         range_z, range_h = z_max - z_min, h_max - h_min
# #         if range_z < 1e-6 or range_h < 1e-6: return

# #         scale = (self.GRID_SIZE - 1) / max(range_z, range_h)

# #         intensity_sum_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=float)
# #         point_count_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
# #         for i in range(len(h_coords)):
# #             col = int((h_coords[i] - h_min) * scale)
# #             row = int((z_coords[i] - z_min) * scale)
# #             grid_row_idx = self.GRID_SIZE - 1 - row
# #             intensity_sum_grid[grid_row_idx, col] += intensities[i]
# #             point_count_grid[grid_row_idx, col] += 1

# #         final_grid = np.full((self.GRID_SIZE, self.GRID_SIZE), '0', dtype=object)
# #         non_empty_cells = point_count_grid > 0
# #         average_intensities = np.divide(intensity_sum_grid, point_count_grid, where=non_empty_cells)
# #         final_grid[non_empty_cells] = np.char.mod('%.1f', average_intensities[non_empty_cells])

# #         row_mask = (final_grid != '0').any(axis=1)
# #         col_mask = (final_grid != '0').any(axis=0)
# #         if np.any(row_mask) and np.any(col_mask):
# #             trimmed_grid = final_grid[row_mask][:, col_mask]
# #         else:
# #             trimmed_grid = np.array([['0']])

# #         self.grid_csv_writer.writerow([])
# #         self.grid_csv_writer.writerow([f"Frame {self.frame_count}", f"Cluster {cluster_label}"])
# #         for grid_row in trimmed_grid:
# #             self.grid_csv_writer.writerow(grid_row)

# # def main(args=None):
# #     rclpy.init(args=args)
# #     node = ClusterProjectorAndLogger()
# #     try:
# #         rclpy.spin(node)
# #     except KeyboardInterrupt:
# #         pass
# #     finally:
# #         node.destroy_node()
# #         rclpy.shutdown()

# # if __name__ == '__main__':
# #     main()

# import rclpy
# from rclpy.node import Node
# from sklearn.cluster import DBSCAN
# from rclpy.duration import Duration
# from sklearn.linear_model import RANSACRegressor
# import numpy as np
# from sensor_msgs.msg import PointCloud
# import csv

# class ClusterProjectorAndLogger(Node):
#     def __init__(self):
#         super().__init__('cluster_projector_and_logger')
#         self.subscriber = self.create_subscription(PointCloud, '/carmaker/pointcloud', self.process_cloud_callback, 10)
#         self.frame_count = 0
        
#         self.points_csv_file = open("projected_points.csv", "w", newline="")
#         self.points_csv_writer = csv.writer(self.points_csv_file)
#         self.points_csv_writer.writerow(["frame", "cluster_id", "proj_x", "proj_y", "proj_z", "intensity"])
        
#         self.grid_csv_file = open("projected_grids.csv", "w", newline="")
#         self.grid_csv_writer = csv.writer(self.grid_csv_file)
#         self.GRID_SIZE = 30

#         self.get_logger().info("Node has started. Logging points and grids.")

#     def __del__(self):
#         if hasattr(self, "points_csv_file") and not self.points_csv_file.closed:
#             self.points_csv_file.close()
#         if hasattr(self, "grid_csv_file") and not self.grid_csv_file.closed:
#             self.grid_csv_file.close()

#     def remove_ground_ransac(self, pc_data):
#         xyz_data = pc_data[:, :3]
#         if xyz_data.shape[0] < 10: return pc_data
#         try:
#             ransac = RANSACRegressor(residual_threshold=0.05)
#             ransac.fit(xyz_data[:, :2], xyz_data[:, 2])
#             return pc_data[~ransac.inlier_mask_]
#         except ValueError:
#             return np.array([])

#     def process_cloud_callback(self, msg):
#         self.frame_count += 1
#         if not msg.points: return

#         pc_data_xyz = np.array([[p.x, p.y, p.z] for p in msg.points])
#         if msg.channels and msg.channels[0].name.lower() == 'intensity':
#             intensities = np.array(msg.channels[0].values).reshape(-1, 1)
#             pc_data = np.hstack((pc_data_xyz, intensities))
#         else:
#             pc_data = np.hstack((pc_data_xyz, np.zeros((pc_data_xyz.shape[0], 1))))

#         pc_data_no_ground = self.remove_ground_ransac(pc_data)
#         if pc_data_no_ground.shape[0] < 5: return

#         labels = DBSCAN(eps=0.5, min_samples=2).fit_predict(pc_data_no_ground[:, :2])
#         unique_labels = set(labels)
#         if -1 in unique_labels: unique_labels.remove(-1)
#         if not unique_labels: return

#         self.get_logger().info(f"--- Frame {self.frame_count}: Found {len(unique_labels)} non-ground clusters ---")

#         for label in unique_labels:
#             cluster_points = pc_data_no_ground[labels == label]
            
#             squared_distances = np.sum(cluster_points[:, :3]**2, axis=1)
#             closest_point_index = np.argmin(squared_distances)
#             closest_point = cluster_points[closest_point_index, :3]
            
#             plane_point = closest_point
#             plane_normal = np.array([closest_point[0], closest_point[1], 0.0])
#             norm = np.linalg.norm(plane_normal)
#             if norm < 1e-6: continue
#             plane_normal /= norm

#             vectors_to_project = cluster_points[:, :3] - plane_point
#             distances_from_plane = np.dot(vectors_to_project, plane_normal)
#             projected_points = cluster_points[:, :3] - np.outer(distances_from_plane, plane_normal)
            
#             for i in range(len(projected_points)):
#                 p = projected_points[i]
#                 intensity = cluster_points[i, 3]
#                 self.points_csv_writer.writerow([self.frame_count, label, f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}", f"{intensity:.4f}"])

#             self.create_and_save_grid(projected_points, cluster_points[:, 3], plane_point, plane_normal, label)

#     # --- MODIFIED: Function updated to fill holes in the grid ---
#     def create_and_save_grid(self, projected_points, intensities, plane_origin, plane_normal, cluster_label):
#         z_coords = projected_points[:, 2]
#         tangent_vector = np.array([-plane_normal[1], plane_normal[0], 0])
#         h_coords = np.dot(projected_points - plane_origin, tangent_vector)

#         z_min, z_max = np.min(z_coords), np.max(z_coords)
#         h_min, h_max = np.min(h_coords), np.max(h_coords)
#         range_z, range_h = z_max - z_min, h_max - h_min
#         if range_z < 1e-6 or range_h < 1e-6: return

#         scale = (self.GRID_SIZE - 1) / max(range_z, range_h)

#         intensity_sum_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=float)
#         point_count_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
#         for i in range(len(h_coords)):
#             col = int((h_coords[i] - h_min) * scale)
#             row = int((z_coords[i] - z_min) * scale)
#             grid_row_idx = self.GRID_SIZE - 1 - row
#             intensity_sum_grid[grid_row_idx, col] += intensities[i]
#             point_count_grid[grid_row_idx, col] += 1

#         final_grid = np.full((self.GRID_SIZE, self.GRID_SIZE), '0', dtype=object)
#         non_empty_cells = point_count_grid > 0
#         average_intensities = np.divide(intensity_sum_grid, point_count_grid, where=non_empty_cells)
#         final_grid[non_empty_cells] = np.char.mod('%.1f', average_intensities[non_empty_cells])

#         # --- NEW: Hole-filling logic ---
#         filled_grid = final_grid.copy()
#         for i in range(self.GRID_SIZE): # For each row
#             row = filled_grid[i, :]
#             non_zero_indices = np.where(row != '0')[0]
            
#             # If there's a gap to fill (at least a start and end point)
#             if len(non_zero_indices) >= 2:
#                 left_edge = non_zero_indices[0]
#                 right_edge = non_zero_indices[-1]
                
#                 # Get all non-zero values in the row to calculate a fill value
#                 row_values = [float(val) for val in row[non_zero_indices]]
#                 fill_value = np.mean(row_values)
                
#                 # Fill the gaps
#                 for j in range(left_edge, right_edge):
#                     if filled_grid[i, j] == '0':
#                         filled_grid[i, j] = f"{fill_value:.1f}"
#         # --- End of new logic ---

#         row_mask = (filled_grid != '0').any(axis=1)
#         col_mask = (filled_grid != '0').any(axis=0)
#         if np.any(row_mask) and np.any(col_mask):
#             trimmed_grid = filled_grid[row_mask][:, col_mask]
#         else:
#             trimmed_grid = np.array([['0']])

#         self.grid_csv_writer.writerow([])
#         self.grid_csv_writer.writerow([f"Frame {self.frame_count}", f"Cluster {cluster_label}"])
#         for grid_row in trimmed_grid:
#             self.grid_csv_writer.writerow(grid_row)


# def main(args=None):
#     rclpy.init(args=args)
#     node = ClusterProjectorAndLogger()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()

import rclpy
from rclpy.node import Node
from sklearn.cluster import DBSCAN
from rclpy.duration import Duration
from sklearn.linear_model import RANSACRegressor
import numpy as np
from sensor_msgs.msg import PointCloud
import csv

class ClusterMerger(Node):
    def __init__(self):
        super().__init__('cluster_merger')
        self.subscriber = self.create_subscription(PointCloud, '/carmaker/pointcloud', self.process_cloud_callback, 10)
        self.frame_count = 0
        
        # --- NEW: CSV setup for the merged 1D row ---
        self.csv_file = open("merged_rows.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        # The header is simpler now
        self.csv_writer.writerow(["frame", "cluster_id", "intensity_profile"])
        
        self.GRID_SIZE = 40 # Using a slightly wider grid for better horizontal resolution

        self.get_logger().info("Node has started. Logging merged rows to merged_rows.csv")

    def __del__(self):
        """Ensure the CSV file is closed properly on shutdown."""
        if hasattr(self, "csv_file") and not self.csv_file.closed:
            self.csv_file.close()

    def remove_ground_ransac(self, pc_data):
        xyz_data = pc_data[:, :3]
        if xyz_data.shape[0] < 10: return pc_data
        try:
            ransac = RANSACRegressor(residual_threshold=0.05)
            ransac.fit(xyz_data[:, :2], xyz_data[:, 2])
            return pc_data[~ransac.inlier_mask_]
        except ValueError:
            return np.array([])

    def process_cloud_callback(self, msg):
        self.frame_count += 1
        if not msg.points: return

        pc_data_xyz = np.array([[p.x, p.y, p.z] for p in msg.points])
        if msg.channels and msg.channels[0].name.lower() == 'intensity':
            intensities = np.array(msg.channels[0].values).reshape(-1, 1)
            pc_data = np.hstack((pc_data_xyz, intensities))
        else:
            pc_data = np.hstack((pc_data_xyz, np.zeros((pc_data_xyz.shape[0], 1))))

        pc_data_no_ground = self.remove_ground_ransac(pc_data)
        if pc_data_no_ground.shape[0] < 5: return

        labels = DBSCAN(eps=0.5, min_samples=2).fit_predict(pc_data_no_ground[:, :2])
        unique_labels = set(labels)
        if -1 in unique_labels: unique_labels.remove(-1)
        if not unique_labels: return

        self.get_logger().info(f"--- Frame {self.frame_count}: Found {len(unique_labels)} non-ground clusters ---")

        for label in unique_labels:
            cluster_points = pc_data_no_ground[labels == label]
            
            squared_distances = np.sum(cluster_points[:, :3]**2, axis=1)
            closest_point_index = np.argmin(squared_distances)
            closest_point = cluster_points[closest_point_index, :3]
            
            plane_point = closest_point
            plane_normal = np.array([closest_point[0], closest_point[1], 0.0])
            norm = np.linalg.norm(plane_normal)
            if norm < 1e-6: continue
            plane_normal /= norm

            vectors_to_project = cluster_points[:, :3] - plane_point
            distances_from_plane = np.dot(vectors_to_project, plane_normal)
            projected_points = cluster_points[:, :3] - np.outer(distances_from_plane, plane_normal)
            
            # --- NEW: Call the function to create and save the merged row ---
            self.create_and_save_merged_row(projected_points, cluster_points[:, 3], plane_point, plane_normal, label)
    
    # --- NEW: Function to create and save the merged row ---
    def create_and_save_merged_row(self, projected_points, intensities, plane_origin, plane_normal, cluster_label):
        # Step 1: Create the 2D grid in memory first
        z_coords = projected_points[:, 2]
        tangent_vector = np.array([-plane_normal[1], plane_normal[0], 0])
        h_coords = np.dot(projected_points - plane_origin, tangent_vector)

        z_min, z_max = np.min(z_coords), np.max(z_coords)
        h_min, h_max = np.min(h_coords), np.max(h_coords)
        range_z, range_h = z_max - z_min, h_max - h_min
        if range_z < 1e-6 or range_h < 1e-6: return

        scale = (self.GRID_SIZE - 1) / max(range_z, range_h)
        
        # We only need one grid this time, to store the max intensity for each cell
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=float)

        for i in range(len(h_coords)):
            col = int((h_coords[i] - h_min) * scale)
            row = int((z_coords[i] - z_min) * scale)
            grid_row_idx = self.GRID_SIZE - 1 - row
            
            # Place the intensity, but only if it's larger than what's already there
            grid[grid_row_idx, col] = max(grid[grid_row_idx, col], intensities[i])

        # Step 2: "Squash" the 2D grid into a 1D row by taking the max of each column
        # The np.max function with axis=0 finds the maximum value in each column
        merged_row = np.max(grid, axis=0)

        # Step 3: Write the single merged row to the CSV
        # We prepend the frame and cluster ID to the row of intensity values
        output_row = [self.frame_count, cluster_label] + [f"{val:.1f}" for val in merged_row]
        self.csv_writer.writerow(output_row)


def main(args=None):
    rclpy.init(args=args)
    node = ClusterMerger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()