# import rclpy
# import numpy as np
# from rclpy.node import Node
# from sklearn.cluster import DBSCAN
# from sensor_msgs.msg import PointCloud
# import rclpy.duration
# import csv
# from datetime import datetime

# class ConeDetector(Node):
#     def __init__(self):
#         super().__init__('cone_detector')

#         self.subscriber = self.create_subscription(PointCloud, '/carmaker/pointcloud', self.process_cloud_callback, 10)
#         self.get_logger().info("ConeDetector node started. Logging z and normalized intensity for points in each cluster.")

#         # --- CSV Logging Setup ---
#         timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.csv_filename = f'cone_detections_{timestamp_str}.csv'
        
#         self.csv_file = open(self.csv_filename, 'w', newline='')
#         self.csv_writer = csv.writer(self.csv_file)
        
#         header = ['frame_id', 'cluster_id', 'z_coordinate', 'normalized_intensity']
#         self.csv_writer.writerow(header)
#         self.get_logger().info(f"Saving detection data to: {self.csv_filename}")

#         self.frame_id = 0

#     def destroy_node(self):
#         """Override destroy_node to cleanly close the CSV file."""
#         self.get_logger().info("Shutting down, closing CSV file.")
#         if self.csv_file:
#             self.csv_file.close()
#         super().destroy_node()

#     def remove_ground_by_height(self, pcd, ground_height_threshold=0.1):
#         """Removes ground points based on a z-axis height threshold."""
#         non_ground_mask = pcd[:, 2] > ground_height_threshold
#         return pcd[non_ground_mask]

#     def process_cloud_callback(self, msg: PointCloud):
#         self.frame_id += 1

#         if not msg.points:
#             return
            
#         intensity_values = None
#         for channel in msg.channels:
#             if channel.name == 'intensity':
#                 intensity_values = np.array(channel.values)
#                 break
        
#         if intensity_values is None:
#             self.get_logger().warn("'intensity' channel not found. Using 0 for all points.")
#             intensity_values = np.zeros(len(msg.points))

#         pcd_xyz = np.array([[p.x, p.y, p.z] for p in msg.points])
#         pcd_full = np.hstack((pcd_xyz, intensity_values.reshape(-1, 1)))

#         pcd = self.remove_ground_by_height(pcd_full, ground_height_threshold=-0.1629)
        
#         if pcd.shape[0] == 0:
#             return

#         clusters = DBSCAN(min_samples=3, eps=0.3).fit(pcd[:, :2])
#         labels = clusters.labels_
#         unique_labels = set(labels)
        
#         num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
#         if num_clusters == 0:
#             return

#         self.get_logger().info(f"Frame {self.frame_id}: Found {num_clusters} potential cone(s).")
        
#         for label in unique_labels:
#             if label == -1:
#                 continue

#             cluster_points = pcd[labels == label].copy()
            
#             cluster_intensities = cluster_points[:, 3]
#             min_intensity = np.min(cluster_intensities)
#             max_intensity = np.max(cluster_intensities)
            
#             if (max_intensity - min_intensity) > 1e-6:
#                 normalized_intensities = (cluster_intensities - min_intensity) / (max_intensity - min_intensity)
#             else:
#                 normalized_intensities = np.zeros_like(cluster_intensities)
            
#             cluster_points[:, 3] = normalized_intensities
#             cluster_points = cluster_points[cluster_points[:, 2].argsort()]

#             self.get_logger().info(f"--> Cluster {label} ({len(cluster_points)} points):")
            
#             for i, point in enumerate(cluster_points):
#                 z_coordinate = point[2]
#                 intensity_value = point[3]
#                 self.get_logger().info(f"    Point {i}: z={z_coordinate:.4f}, intensity={intensity_value:.2f}")

#                 row_data = [self.frame_id, label, z_coordinate, intensity_value]
#                 self.csv_writer.writerow(row_data)
            
#             # --- ADDED: Write a blank line to the CSV after processing all points in a cluster ---
#             self.csv_writer.writerow([])


# def main(args=None):
#     rclpy.init(args=args)
#     node = ConeDetector()
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
import numpy as np
from rclpy.node import Node
from sklearn.cluster import DBSCAN
from sensor_msgs.msg import PointCloud
import rclpy.duration
import csv
from datetime import datetime


class ConeDetector(Node):
    def __init__(self):
        super().__init__('cone_detector')

        self.subscriber = self.create_subscription(PointCloud, '/carmaker/pointcloud', self.process_cloud_callback, 10)
        self.get_logger().info("ConeDetector node started. Logging z, normalized intensity, and classification for points in each cluster.")

        # --- CSV Logging Setup ---
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f'cone_detections_{timestamp_str}.csv'
        
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # --- MODIFIED: Added 'classification' to the header ---
        header = ['frame_id', 'cluster_id', 'z_coordinate', 'normalized_intensity', 'classification']
        self.csv_writer.writerow(header)
        self.get_logger().info(f"Saving detection data to: {self.csv_filename}")

        self.frame_id = 0

    def destroy_node(self):
        """Override destroy_node to cleanly close the CSV file."""
        self.get_logger().info("Shutting down, closing CSV file.")
        if self.csv_file:
            self.csv_file.close()
        super().destroy_node()

    def remove_ground_by_height(self, pcd, ground_height_threshold=0.1):
        """Removes ground points based on a z-axis height threshold."""
        non_ground_mask = pcd[:, 2] > ground_height_threshold
        return pcd[non_ground_mask]

    def process_cloud_callback(self, msg: PointCloud):
        self.frame_id += 1

        if not msg.points:
            return
            
        intensity_values = None
        for channel in msg.channels:
            if channel.name == 'intensity':
                intensity_values = np.array(channel.values)
                break
        
        if intensity_values is None:
            self.get_logger().warn("'intensity' channel not found. Using 0 for all points.")
            intensity_values = np.zeros(len(msg.points))

        pcd_xyz = np.array([[p.x, p.y, p.z] for p in msg.points])
        pcd_full = np.hstack((pcd_xyz, intensity_values.reshape(-1, 1)))

        pcd = self.remove_ground_by_height(pcd_full, ground_height_threshold=-0.1629)
        
        if pcd.shape[0] == 0:
            return

        # --- MODIFIED: Changed min_samples from 3 to 8 to discard smaller clusters ---
        clusters = DBSCAN(min_samples=8, eps=0.3).fit(pcd[:, :2])
        labels = clusters.labels_
        unique_labels = set(labels)
        
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        if num_clusters == 0:
            return

        self.get_logger().info(f"Frame {self.frame_id}: Found {num_clusters} potential cone(s).")
        
        for label in unique_labels:
            if label == -1:
                continue

            cluster_points = pcd[labels == label].copy()
            
            # --- ADDED: Classification based on Y-coordinate ---
            # Calculate the average y-coordinate of the cluster
            mean_y = np.mean(cluster_points[:, 1])
            # Classify as 1 if on the right (negative y) and 0 if on the left (positive y)
            classification = 1 if mean_y < 0 else 0
            side = "right" if classification == 1 else "left"
            
            cluster_intensities = cluster_points[:, 3]
            min_intensity = np.min(cluster_intensities)
            max_intensity = np.max(cluster_intensities)
            
            if (max_intensity - min_intensity) > 1e-6:
                normalized_intensities = (cluster_intensities - min_intensity) / (max_intensity - min_intensity)
            else:
                normalized_intensities = np.zeros_like(cluster_intensities)
            
            cluster_points[:, 3] = normalized_intensities
            cluster_points = cluster_points[cluster_points[:, 2].argsort()]

            self.get_logger().info(f"--> Cluster {label} ({len(cluster_points)} points) on the {side}:")
            
            for i, point in enumerate(cluster_points):
                z_coordinate = point[2]
                intensity_value = point[3]
                self.get_logger().info(f"    Point {i}: z={z_coordinate:.4f}, intensity={intensity_value:.2f}")

                # --- MODIFIED: Added classification to the CSV row ---
                row_data = [self.frame_id, label, z_coordinate, intensity_value, classification]
                self.csv_writer.writerow(row_data)
            
            # Write a blank line to the CSV after processing all points in a cluster
            self.csv_writer.writerow([])


def main(args=None):
    rclpy.init(args=args)
    node = ConeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
