# import rclpy
# import numpy as np
# from rclpy.node import Node
# from sklearn.cluster import DBSCAN
# from sklearn.linear_model import RANSACRegressor
# from sensor_msgs.msg import PointCloud
# from visualization_msgs.msg import Marker, MarkerArray

# class ConeDetector(Node):

#     def __init__(self):
#         super().__init__('cone_detector')
#         self.subscriber = self.create_subscription(PointCloud, '/carmaker/pointcloud', self.process_cloud_callback, 10)
#         self.publisher = self.create_publisher(MarkerArray, 'vis_db_and_ran', 10)
#         self.frame_id = 0


#     def remove_ground_ransac(self, pcd):
#         # This function remains unchanged
#         xy = pcd[:, :2]
#         z = pcd[:, 2]
#         if len(xy) < 10: return pcd
#         ransac = RANSACRegressor(residual_threshold=0.03)
#         ransac.fit(xy, z)
#         inlier = ransac.inlier_mask_
#         return pcd[~inlier]

#     def process_cloud_callback(self, msg: PointCloud):
#         if not msg.points:
#             return
        
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
from rclpy.node import Node
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker, MarkerArray

class ConeDetector(Node):

    def __init__(self):
        super().__init__('cone_detector')
        # Subscribes to the incoming point cloud data
        self.subscriber = self.create_subscription(PointCloud, '/carmaker/pointcloud', self.process_cloud_callback, 10)
        # Publishes the visualization markers to RViz2
        self.publisher = self.create_publisher(MarkerArray, 'vis_db_and_ran', 10)
        self.get_logger().info("Node initialized. Publishing markers on '/vis_db_and_ran'.")


    def remove_ground_ransac(self, pcd):
        # This function remains unchanged
        xy = pcd[:, :2]
        z = pcd[:, 2]
        if len(xy) < 10: return pcd
        ransac = RANSACRegressor(residual_threshold=0.03)
        ransac.fit(xy, z)
        inlier = ransac.inlier_mask_
        # Return only the points that are not part of the ground
        return pcd[~inlier]

    def process_cloud_callback(self, msg: PointCloud):
        if not msg.points:
            return
        
        # 1. Convert PointCloud to a NumPy array (XYZI)
        pcd_xyz = np.array([[p.x, p.y, p.z] for p in msg.points])
        intensities = np.zeros((len(msg.points), 1))
        for channel in msg.channels:
            if channel.name.lower() == "intensity":
                intensities = np.array(channel.values).reshape(-1, 1)
                break
        pcd = np.hstack((pcd_xyz, intensities))

        # 2. Remove the ground plane using RANSAC
        pcd_no_ground = self.remove_ground_ransac(pcd)
        if pcd_no_ground.shape[0] == 0:
            # Clear previous markers if no points are left
            marker_array = MarkerArray()
            delete_marker = Marker()
            delete_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_marker)
            self.publisher.publish(marker_array)
            return

        # 3. Cluster the remaining points
        clusters = DBSCAN(min_samples=2, eps=0.5).fit(pcd_no_ground[:, :2])
        labels = clusters.labels_
        unique_labels = set(labels)
        
        # <<< MODIFIED SECTION START >>>
        
        # 4. Create and publish markers for each cluster
        marker_array = MarkerArray()

        # It's good practice to clear old markers first
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        self.publisher.publish(marker_array) # Publish the deletion command

        # Create a new array for the current markers
        marker_array = MarkerArray()

        for label in unique_labels:
            # -1 is the label for noise points, which we ignore
            if label == -1:
                continue

            # Get all points belonging to the current cluster
            cluster_points = pcd_no_ground[labels == label]
            
            # Calculate the center of the cluster to place the marker
            centroid = np.mean(cluster_points[:, :3], axis=0)
            
            # --- Create a CYLINDER marker for the cluster ---
            cylinder_marker = Marker()
            cylinder_marker.header.frame_id = msg.header.frame_id
            cylinder_marker.header.stamp = self.get_clock().now().to_msg()
            cylinder_marker.ns = "clusters"
            cylinder_marker.id = int(label) # Use the cluster label as the marker ID
            cylinder_marker.type = Marker.CYLINDER
            cylinder_marker.action = Marker.ADD
            
            # Position the marker at the centroid
            cylinder_marker.pose.position.x = float(centroid[0])
            cylinder_marker.pose.position.y = float(centroid[1])
            cylinder_marker.pose.position.z = float(centroid[2])
            cylinder_marker.pose.orientation.w = 1.0

            # Set the marker's scale and color (e.g., a blue cylinder)
            cylinder_marker.scale.x = 0.1  # Diameter
            cylinder_marker.scale.y = 0.1  # Diameter
            cylinder_marker.scale.z = 0.3  # Height
            cylinder_marker.color.a = 0.8  # Alpha (transparency)
            cylinder_marker.color.r = 0.0
            cylinder_marker.color.g = 0.0
            cylinder_marker.color.b = 1.0
            
            # --- Create a TEXT marker for the cluster ID ---
            text_marker = Marker()
            text_marker.header.frame_id = msg.header.frame_id
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "cluster_ids"
            # Give the text a unique ID to avoid conflicting with the cylinder
            text_marker.id = int(label) + 1000 
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.text = str(label)

            # Position the text slightly above the cylinder
            text_marker.pose.position.x = float(centroid[0])
            text_marker.pose.position.y = float(centroid[1])
            text_marker.pose.position.z = float(centroid[2]) + 0.6 # Place above cylinder
            
            # Set the text's scale and color (e.g., white)
            text_marker.scale.z = 0.5 # Text height
            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0

            # Add both the cylinder and its text ID to the array
            marker_array.markers.append(cylinder_marker)
            marker_array.markers.append(text_marker)
        
        # Publish the complete marker array to RViz2
        if marker_array.markers:
            self.publisher.publish(marker_array)
            self.get_logger().info(f"Published markers for {len(unique_labels)-1} clusters.")
        
        # <<< MODIFIED SECTION END >>>

def main(args=None):
    # This function remains unchanged
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