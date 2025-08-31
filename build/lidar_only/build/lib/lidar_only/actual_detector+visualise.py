# # import rclpy
# # import numpy as np
# # import csv
# # from rclpy.node import Node
# # from sklearn.cluster import DBSCAN
# # from sklearn.linear_model import RANSACRegressor
# # from sensor_msgs.msg import PointCloud

# # class ConeDetector(Node):
# #     def __init__(self):
# #         super().__init__('cone_detector')
# #         self.subscriber = self.create_subscription(PointCloud, '/carmaker/pointcloud', self.process_cloud_callback, 10)

# #         self.frame_id = 0

# #         #  Setup for a 2D profile CSV file
# #         self.csv_filename = "cone_profile_2D.csv"
# #         # Clear the file and write the new header row
# #         with open(self.csv_filename, 'w', newline='') as f:
# #             writer = csv.writer(f)
# #             writer.writerow(["frame_id", "cluster_id", "radial_distance", "height", "intensity"])
        
# #         self.get_logger().info(f"Node initialised. Logging 2D profile data to {self.csv_filename}.")

# #     def remove_ground_ransac(self, pcd):
# #         # This function remains unchanged
# #         xy = pcd[:, :2]
# #         z = pcd[:, 2]
# #         if len(xy) < 10: return pcd
# #         ransac = RANSACRegressor(residual_threshold=0.03)
# #         ransac.fit(xy, z)
# #         inlier = ransac.inlier_mask_
# #         return pcd[~inlier]

# #     def process_cloud_callback(self, msg: PointCloud):
# #         if not msg.points:
# #             return
        
# #         # Point cloud processing to include intensity (unchanged)
# #         pcd_xyz = np.array([[p.x, p.y, p.z] for p in msg.points])
# #         intensities = np.zeros((len(msg.points), 1))
# #         for channel in msg.channels:
# #             if channel.name.lower() == "intensity":
# #                 intensities = np.array(channel.values).reshape(-1, 1)
# #                 break
# #         pcd = np.hstack((pcd_xyz, intensities))
# #         pcd = self.remove_ground_ransac(pcd)
# #         if pcd.shape[0] == 0:
# #             return

# #         # Clustering (unchanged)
# #         clusters = DBSCAN(min_samples=2, eps=0.5).fit(pcd[:, :2])
# #         labels = clusters.labels_
# #         unique_labels = set(labels)
        
# #         csv_point_rows = []

# #         for label in unique_labels:
# #             if label == -1:
# #                 continue

# #             # Geometric transformation logic (unchanged)
# #             cluster_points = pcd[labels == label]
# #             cluster_points_xyz = cluster_points[:, :3]
# #             cluster_intensities = cluster_points[:, 3]

# #             distances = np.linalg.norm(cluster_points_xyz, axis=1)
# #             min_index = np.argmin(distances)
# #             nearest_point = cluster_points_xyz[min_index]
# #             a, b = nearest_point[0], nearest_point[1]
# #             plane_normal = np.array([a, b, 0])
# #             norm_squared = np.dot(plane_normal, plane_normal)
# #             if np.isclose(norm_squared, 0): continue
# #             vec_to_points = cluster_points_xyz - nearest_point
# #             scales = np.dot(vec_to_points, plane_normal) / norm_squared
# #             projected_points_xyz = cluster_points_xyz - scales[:, np.newaxis] * plane_normal
            
# #             centroid = np.mean(projected_points_xyz, axis=0)
# #             transformed_points = projected_points_xyz - centroid
            
# #             # <<< MODIFIED >>> Prepare a 2D profile row for each point
# #             for i, point in enumerate(transformed_points):
# #                 # Calculate the radial distance from the new central axis (t_x, t_y)
# #                 radial_distance = np.sqrt(point[0]**2 + point[1]**2)
                
# #                 # The height is the z-component of the transformed point (t_z)
# #                 height = point[2]
                
# #                 # Append the new 2D profile data to the list
# #                 csv_point_rows.append([
# #                     self.frame_id,
# #                     label,
# #                     radial_distance,
# #                     height,
# #                     cluster_intensities[i]
# #                 ])
        
# #         # Write all collected rows to the CSV file (unchanged)
# #         if csv_point_rows:
# #             with open(self.csv_filename, 'a', newline='\n') as f:
# #                 writer = csv.writer(f)
# #                 writer.writerows(csv_point_rows)
# #                 self.get_logger().info(f"Frame {self.frame_id}: Appended {len(csv_point_rows)} profile points to {self.csv_filename}.")

# #         self.frame_id += 1

# # def main(args=None):
# #     # This function remains unchanged
# #     rclpy.init(args=args)
# #     node = ConeDetector()
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
# import numpy as np
# import csv
# from rclpy.node import Node
# from sklearn.cluster import DBSCAN
# from sklearn.linear_model import RANSACRegressor

# # ROS message imports
# from sensor_msgs.msg import PointCloud
# from visualization_msgs.msg import Marker, MarkerArray
# from geometry_msgs.msg import Point
# from std_msgs.msg import ColorRGBA
# import rclpy.duration

# class ConeDetector(Node):
#     def __init__(self):
#         super().__init__('cone_detector')
        
#         # Subscriber for the point cloud
#         self.subscriber = self.create_subscription(
#             PointCloud, 
#             '/carmaker/pointcloud', 
#             self.process_cloud_callback, 
#             10
#         )
        
#         # <<< NEW >>> Publisher for visualization markers
#         self.marker_publisher = self.create_publisher(MarkerArray, '/cone_markers', 10)

#         self.frame_id = 0

#         #  Setup for a 2D profile CSV file
#         self.csv_filename = "cone_profile_2D.csv"
#         # Clear the file and write the new header row
#         with open(self.csv_filename, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(["frame_id", "cluster_id", "radial_distance", "height", "intensity"])
        
#         self.get_logger().info(f"Node initialised. Logging 2D profile data to {self.csv_filename}.")
#         self.get_logger().info("Publishing cluster markers to /cone_markers topic.")

#     def remove_ground_ransac(self, pcd):
#         # This function remains unchanged
#         xy = pcd[:, :2]
#         z = pcd[:, 2]
#         if len(xy) < 10: return pcd
#         ransac = RANSACRegressor(residual_threshold=0.03)
#         ransac.fit(xy, z)
#         inlier = ransac.inlier_mask_
#         return pcd[~inlier]

#     def get_color(self, label):
#         """Generates a unique color for each cluster label."""
#         # Simple color cycling
#         colors = [
#             (1.0, 0.0, 0.0), # Red
#             (0.0, 1.0, 0.0), # Green
#             (0.0, 0.0, 1.0), # Blue
#             (1.0, 1.0, 0.0), # Yellow
#             (1.0, 0.0, 1.0), # Magenta
#             (0.0, 1.0, 1.0), # Cyan
#         ]
#         color_index = label % len(colors)
#         r, g, b = colors[color_index]
#         return ColorRGBA(r=r, g=g, b=b, a=0.8)

#     def process_cloud_callback(self, msg: PointCloud):
#         if not msg.points:
#             return
            
#         # <<< NEW >>> Initialize a MarkerArray to hold all markers for this frame
#         marker_array = MarkerArray()
        
#         # <<< NEW >>> Create a marker to delete all previous markers in this namespace
#         # This ensures that markers from old frames don't persist in RViz
#         delete_marker = Marker()
#         delete_marker.action = Marker.DELETEALL
#         marker_array.markers.append(delete_marker)
#         self.marker_publisher.publish(marker_array)
        
#         # Re-initialize the array for the new markers
#         marker_array = MarkerArray()

#         # Point cloud processing to include intensity (unchanged)
#         pcd_xyz = np.array([[p.x, p.y, p.z] for p in msg.points])
#         intensities = np.zeros((len(msg.points), 1))
#         for channel in msg.channels:
#             if channel.name.lower() == "intensity":
#                 intensities = np.array(channel.values).reshape(-1, 1)
#                 break
#         pcd = np.hstack((pcd_xyz, intensities))
#         pcd = self.remove_ground_ransac(pcd)
#         if pcd.shape[0] == 0:
#             return

#         # Clustering (unchanged)
#         clusters = DBSCAN(min_samples=2, eps=0.5).fit(pcd[:, :2])
#         labels = clusters.labels_
#         unique_labels = set(labels)
        
#         csv_point_rows = []

#         for label in unique_labels:
#             if label == -1:
#                 continue

#             # Geometric transformation logic (unchanged)
#             cluster_points = pcd[labels == label]
#             cluster_points_xyz = cluster_points[:, :3]
#             cluster_intensities = cluster_points[:, 3]

#             distances = np.linalg.norm(cluster_points_xyz, axis=1)
#             min_index = np.argmin(distances)
#             nearest_point = cluster_points_xyz[min_index]
#             a, b = nearest_point[0], nearest_point[1]
#             plane_normal = np.array([a, b, 0])
#             norm_squared = np.dot(plane_normal, plane_normal)
#             if np.isclose(norm_squared, 0): continue
#             vec_to_points = cluster_points_xyz - nearest_point
#             scales = np.dot(vec_to_points, plane_normal) / norm_squared
#             projected_points_xyz = cluster_points_xyz - scales[:, np.newaxis] * plane_normal
            
#             centroid = np.mean(projected_points_xyz, axis=0)
#             transformed_points = projected_points_xyz - centroid
            
#             # <<< MODIFIED >>> Prepare a 2D profile row for each point
#             for i, point in enumerate(transformed_points):
#                 radial_distance = np.sqrt(point[0]**2 + point[1]**2)
#                 height = point[2]
#                 csv_point_rows.append([
#                     self.frame_id,
#                     label,
#                     radial_distance,
#                     height,
#                     cluster_intensities[i]
#                 ])

#             # <<< NEW >>> Create a SPHERE_LIST marker for the cluster points
#             cluster_marker = Marker()
#             cluster_marker.header = msg.header
#             cluster_marker.ns = "cone_clusters"
#             cluster_marker.id = int(label)
#             cluster_marker.type = Marker.SPHERE_LIST
#             cluster_marker.action = Marker.ADD
#             cluster_marker.scale.x = 0.08  # Sphere diameter
#             cluster_marker.scale.y = 0.08
#             cluster_marker.scale.z = 0.08
#             cluster_marker.color = self.get_color(int(label))
#             cluster_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
            
#             # Add all points from the cluster to the marker
#             cluster_marker.points = [Point(x=p[0], y=p[1], z=p[2]) for p in cluster_points_xyz]
#             marker_array.markers.append(cluster_marker)

#             # <<< NEW >>> Create a TEXT_VIEW_FACING marker for the cluster ID
#             text_marker = Marker()
#             text_marker.header = msg.header
#             text_marker.ns = "cluster_ids"
#             text_marker.id = int(label)
#             text_marker.type = Marker.TEXT_VIEW_FACING
#             text_marker.action = Marker.ADD
#             text_marker.text = f"F:{self.frame_id} C:{label}"
            
#             # Position the text slightly above the cluster's centroid
#             text_marker.pose.position.x = centroid[0]
#             text_marker.pose.position.y = centroid[1]
#             text_marker.pose.position.z = np.max(cluster_points_xyz[:, 2]) + 0.3  # Place above highest point
            
#             text_marker.scale.z = 0.2  # Text height
#             text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0) # White
#             text_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
#             marker_array.markers.append(text_marker)

#         # <<< NEW >>> Publish the complete marker array for this frame
#         if marker_array.markers:
#             self.marker_publisher.publish(marker_array)

#         # Write all collected rows to the CSV file (unchanged)
#         if csv_point_rows:
#             with open(self.csv_filename, 'a', newline='\n') as f:
#                 writer = csv.writer(f)
#                 writer.writerows(csv_point_rows)
#                 self.get_logger().info(f"Frame {self.frame_id}: Appended {len(csv_point_rows)} profile points to {self.csv_filename}.")

#         self.frame_id += 1

# def main(args=None):
#     # This function remains unchanged
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
import csv
from rclpy.node import Node
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor

# ROS message imports
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import rclpy.duration

class ConeDetector(Node):
    def __init__(self):
        super().__init__('cone_detector')
        
        # Subscriber for the point cloud
        self.subscriber = self.create_subscription(
            PointCloud, 
            '/carmaker/pointcloud', 
            self.process_cloud_callback, 
            10
        )
        
        # Publisher for visualization markers
        self.marker_publisher = self.create_publisher(MarkerArray, '/cone_markers', 10)

        self.frame_id = 0

        #  Setup for a 2D profile CSV file
        self.csv_filename = "cone_profile_2D.csv"
        # Clear the file and write the new header row
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "cluster_id", "radial_distance", "height", "intensity"])
        
        self.get_logger().info(f"Node initialised. Logging 2D profile data to {self.csv_filename}.")
        self.get_logger().info("Publishing cluster markers to /cone_markers topic.")

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

    def get_color(self, label):
        """Generates a unique color for each cluster label."""
        # Simple color cycling
        colors = [
            (1.0, 0.0, 0.0), # Red
            (0.0, 1.0, 0.0), # Green
            (0.0, 0.0, 1.0), # Blue
            (1.0, 1.0, 0.0), # Yellow
            (1.0, 0.0, 1.0), # Magenta
            (0.0, 1.0, 1.0), # Cyan
        ]
        color_index = label % len(colors)
        r, g, b = colors[color_index]
        return ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)

    def process_cloud_callback(self, msg: PointCloud):
        if not msg.points:
            return
            
        # Initialize a MarkerArray to hold all markers for this frame
        marker_array = MarkerArray()
        
        # Create a marker to delete all previous markers.
        # This is crucial for when using infinite lifetimes.
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        self.marker_publisher.publish(marker_array)
        
        # Re-initialize the array for the new markers
        marker_array = MarkerArray()

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
        clusters = DBSCAN(min_samples=2, eps=0.5).fit(pcd[:, :2])
        labels = clusters.labels_
        unique_labels = set(labels)
        
        csv_point_rows = []

        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue

            # --- Geometric Transformation Logic ---
            cluster_points = pcd[labels == label]
            cluster_points_xyz = cluster_points[:, :3]
            cluster_intensities = cluster_points[:, 3]

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
                csv_point_rows.append([
                    self.frame_id,
                    label,
                    radial_distance,
                    height,
                    cluster_intensities[i]
                ])

            # --- Visualization Marker for Cluster Points ---
            cluster_marker = Marker()
            cluster_marker.header = msg.header
            cluster_marker.ns = "cone_clusters"
            cluster_marker.id = int(label)
            cluster_marker.type = Marker.SPHERE_LIST
            cluster_marker.action = Marker.ADD
            cluster_marker.scale.x = 0.08  # Sphere diameter
            cluster_marker.scale.y = 0.08
            cluster_marker.scale.z = 0.08
            cluster_marker.color = self.get_color(int(label))
            cluster_marker.lifetime = rclpy.duration.Duration(seconds=0.0).to_msg() # Infinite lifetime
            cluster_marker.points = [Point(x=p[0], y=p[1], z=p[2]) for p in cluster_points_xyz]
            marker_array.markers.append(cluster_marker)

            # --- Visualization Marker for Text ID ---
            text_marker = Marker()
            text_marker.header = msg.header
            text_marker.ns = "cluster_ids"
            text_marker.id = int(label)
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.text = f"F:{self.frame_id} C:{label}"
            text_marker.pose.position.x = centroid[0]
            text_marker.pose.position.y = centroid[1]
            text_marker.pose.position.z = np.max(cluster_points_xyz[:, 2]) + 0.3  # Place above highest point
            text_marker.scale.z = 0.2  # Text height
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0) # White
            text_marker.lifetime = rclpy.duration.Duration(seconds=0.0).to_msg() # Infinite lifetime
            marker_array.markers.append(text_marker)

        # Publish the complete marker array for this frame
        if marker_array.markers:
            self.marker_publisher.publish(marker_array)

        # Write all collected rows to the CSV file
        if csv_point_rows:
            with open(self.csv_filename, 'a', newline='\n') as f:
                writer = csv.writer(f)
                writer.writerows(csv_point_rows)
            self.get_logger().info(f"Frame {self.frame_id}: Found {len(unique_labels)-1} clusters. Appended {len(csv_point_rows)} points to CSV.")

        self.frame_id += 1

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