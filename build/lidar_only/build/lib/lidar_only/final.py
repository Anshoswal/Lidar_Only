# import rclpy
# import numpy as np
# import csv
# from rclpy.node import Node
# from sklearn.cluster import DBSCAN
# from sklearn.linear_model import RANSACRegressor
# from sensor_msgs.msg import PointCloud

# class ConeDetector(Node):
#     def __init__(self):
#         super().__init__('cone_detector')
        
#         # Subscriber for the point cloud
#         self.subscriber = self.create_subscription(PointCloud, '/carmaker/pointcloud', self.process_cloud_callback, 10)

#         self.frame_id = 0

#         #  Setup for a 2D profile CSV file
#         self.csv_filename = "cone_profile_2D.csv"
#         # Clear the file and write the new header row
#         with open(self.csv_filename, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(["frame_id", "cluster_id", "radial_distance", "height", "intensity"])
        
#         self.get_logger().info(f"Node initialised. Logging 2D profile data to {self.csv_filename}.")

#     def remove_ground_ransac(self, pcd):
#         """Removes ground points from a point cloud using RANSAC."""
#         xy = pcd[:, :2]
#         z = pcd[:, 2]
#         if len(xy) < 10: 
#             return pcd
#         ransac = RANSACRegressor(residual_threshold=0.03)
#         ransac.fit(xy, z)
#         inlier_mask = ransac.inlier_mask_
#         return pcd[~inlier_mask]

#     def process_cloud_callback(self, msg: PointCloud):
#         if not msg.points:
#             return

#         # Convert PointCloud message to a NumPy array with intensity
#         pcd_xyz = np.array([[p.x, p.y, p.z] for p in msg.points])
#         intensities = np.zeros((len(msg.points), 1))
#         for channel in msg.channels:
#             if channel.name.lower() == "intensity":
#                 intensities = np.array(channel.values).reshape(-1, 1)
#                 break
#         pcd = np.hstack((pcd_xyz, intensities))
        
#         # Filter out ground points
#         pcd = self.remove_ground_ransac(pcd)
#         if pcd.shape[0] == 0:
#             return

#         # Perform clustering on the remaining points
#         clusters = DBSCAN(min_samples=2, eps=0.5).fit(pcd[:, :2])
#         labels = clusters.labels_
#         unique_labels = set(labels)
        
#         csv_point_rows = []

#         for label in unique_labels:
#             if label == -1:  # Skip noise points
#                 continue

#             cluster_points = pcd[labels == label]
#             cluster_points_xyz = cluster_points[:, :3]
#             cluster_intensities = cluster_points[:, 3]

#             distances = np.linalg.norm(cluster_points_xyz, axis=1)
#             min_index = np.argmin(distances)
#             nearest_point = cluster_points_xyz[min_index]
            
#             a, b = nearest_point[0], nearest_point[1]
#             plane_normal = np.array([a, b, 0])
#             norm_squared = np.dot(plane_normal, plane_normal)
#             if np.isclose(norm_squared, 0): 
#                 continue
            
#             vec_to_points = cluster_points_xyz - nearest_point
#             scales = np.dot(vec_to_points, plane_normal) / norm_squared
#             projected_points_xyz = cluster_points_xyz - scales[:, np.newaxis] * plane_normal
            
#             centroid = np.mean(projected_points_xyz, axis=0)
#             transformed_points = projected_points_xyz - centroid
            
#             # Prepare data for CSV logging
#             for i, point in enumerate(transformed_points):
#                 radial_distance = np.sqrt(point[0]**2 + point[1]**2)
#                 height = point[2]
#                 csv_point_rows.append([self.frame_id,label,radial_distance,height,cluster_intensities[i]])

#         # Write all collected rows to the CSV file
#         if csv_point_rows:
#             with open(self.csv_filename, 'a', newline='\n') as f:
#                 writer = csv.writer(f)
#                 writer.writerows(csv_point_rows)
#             self.get_logger().info(f"Frame {self.frame_id}: Appended {len(csv_point_rows)} profile points to CSV.")

#         self.frame_id += 1

# def main(args=None):
#     rclpy.init(args=args)
#     node = ConeDetector()    
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

import rclpy
import numpy as np
import csv
from rclpy.node import Node
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from sensor_msgs.msg import PointCloud

class ConeDetector(Node):
    def __init__(self):
        super().__init__('cone_detector')
        
        # Subscriber for the point cloud
        self.subscriber = self.create_subscription(PointCloud, '/carmaker/pointcloud', self.process_cloud_callback, 10)

        self.frame_id = 0

        #  Setup for a 2D profile CSV file with the new column
        self.csv_filename = "cone_profile_2D.csv"
        # Clear the file and write the new header row
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # <<< MODIFIED >>> Added 'normalized_intensity' column
            writer.writerow(["frame_id", "cluster_id", "radial_distance", "height", "intensity", "normalized_intensity"])
        
        self.get_logger().info(f"Node initialised. Logging 2D profile data to {self.csv_filename}.")

    def remove_ground_ransac(self, pcd):
        """Removes ground points from a point cloud using RANSAC."""
        xy = pcd[:, :2]
        z = pcd[:, 2]
        if len(xy) < 10: 
            return pcd
        ransac = RANSACRegressor(residual_threshold=0.03)
        ransac.fit(xy, z)
        inlier_mask = ransac.inlier_mask_
        return pcd[~inlier_mask]

    def process_cloud_callback(self, msg: PointCloud):
        if not msg.points:
            return

        # Convert PointCloud message to a NumPy array with intensity
        pcd_xyz = np.array([[p.x, p.y, p.z] for p in msg.points])
        intensities = np.zeros((len(msg.points), 1))
        for channel in msg.channels:
            if channel.name.lower() == "intensity":
                intensities = np.array(channel.values).reshape(-1, 1)
                break
        pcd = np.hstack((pcd_xyz, intensities))
        
        # Filter out ground points
        pcd = self.remove_ground_ransac(pcd)
        if pcd.shape[0] == 0:
            return

        # Perform clustering on the remaining points
        clusters = DBSCAN(min_samples=6, eps=0.20).fit(pcd[:, :2])
        labels = clusters.labels_
        unique_labels = set(labels)
        
        csv_point_rows = []

        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue

            cluster_points = pcd[labels == label]
            cluster_points_xyz = cluster_points[:, :3]
            cluster_intensities = cluster_points[:, 3]

            # <<< NEW >>> Normalize intensities within this specific cluster
            max_intensity = np.max(cluster_intensities)
            
            # --- Geometric Transformation Logic (unchanged) ---
            distances = np.linalg.norm(cluster_points_xyz, axis=1)
            min_index = np.argmin(distances)
            nearest_point = cluster_points_xyz[min_index]
            
            a, b = nearest_point[0], nearest_point[1]
            plane_normal = np.array([a, b, 0])
            norm_squared = np.dot(plane_normal, plane_normal)
            if np.isclose(norm_squared, 0): 
                continue
            
            vec_to_points = cluster_points_xyz - nearest_point
            scales = np.dot(vec_to_points, plane_normal) / norm_squared
            projected_points_xyz = cluster_points_xyz - scales[:, np.newaxis] * plane_normal
            
            centroid = np.mean(projected_points_xyz, axis=0)
            transformed_points = projected_points_xyz - centroid
            
            # Prepare data for CSV logging
            for i, point in enumerate(transformed_points):
                radial_distance = np.sqrt(point[0]**2 + point[1]**2)
                height = point[2]
                actual_intensity = cluster_intensities[i]
                
                # <<< NEW >>> Calculate normalized intensity, handle division by zero
                if max_intensity > 0:
                    normalized_intensity = actual_intensity / max_intensity
                else:
                    normalized_intensity = 0.0
                
                # <<< MODIFIED >>> Append the new normalized value to the row
                csv_point_rows.append([
                    self.frame_id,
                    label,
                    radial_distance,
                    height,
                    actual_intensity,
                    normalized_intensity
                ])

        # Write all collected rows to the CSV file
        if csv_point_rows:
            with open(self.csv_filename, 'a', newline='\n') as f:
                writer = csv.writer(f)
                writer.writerows(csv_point_rows)
            self.get_logger().info(f"Frame {self.frame_id}: Appended {len(csv_point_rows)} profile points to CSV.")

        self.frame_id += 1

def main(args=None):
    rclpy.init(args=args)
    node = ConeDetector()    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Gracefully shutdown the node
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()