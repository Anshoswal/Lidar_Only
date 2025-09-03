# This code removes ground by height.
# Then it removes ground by height
# Gets color by using left and right.
# Publishes as array. 

import rclpy
import numpy as np
from rclpy.node import Node
from sklearn.cluster import DBSCAN

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

        self.get_logger().info("ConeDetector node started. Visualizing cones in RViz.")

    def remove_ground_by_height(self, pcd, ground_height_threshold=0.1):
        # Create a boolean mask where True indicates the point's z-value is above the threshold
        non_ground_mask = pcd[:, 2] > ground_height_threshold
        
        # Return only the points that are not on the ground
        return pcd[non_ground_mask]

    def get_color(self, y_value):
        """Assigns blue if cone is on the left (y > 0), yellow if on the right (y <= 0)."""
        if y_value > 0.05:  # Use a small threshold for robustness
            return ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)  # Blue
        else:
            return ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)  # Yellow

    def process_cloud_callback(self, msg: PointCloud):
        if not msg.points:
            return
            
        # Clear previous markers
        delete_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_array.markers.append(delete_marker)
        self.marker_publisher.publish(delete_array)
        
        # Convert PointCloud message to a NumPy array
        pcd_xyz = np.array([[p.x, p.y, p.z] for p in msg.points])
        
        # --- MODIFICATION: Using z-thresholding instead of RANSAC ---
        # You can adjust the threshold (e.g., 0.1 for 10cm) based on your sensor's height and noise
        pcd = self.remove_ground_by_height(pcd_xyz, ground_height_threshold = -0.1629 )
        
        if pcd.shape[0] == 0:
            self.get_logger().warn("All points were filtered as ground plane.")
            return

        # Cluster remaining points (2D clustering on x,y)
        clusters = DBSCAN(min_samples=3, eps=0.3).fit(pcd[:, :2])
        labels = clusters.labels_
        unique_labels = set(labels)
        
        # Check if any clusters were found
        if len(unique_labels) <= 1 and -1 in unique_labels:
            self.get_logger().info("No clusters found after ground removal.")
            return

        marker_array = MarkerArray()

        for label in unique_labels:
            if label == -1:  # Skip noise
                continue

            cluster_points = pcd[labels == label]
            centroid = np.mean(cluster_points, axis=0)

            cyl_marker = Marker()
            cyl_marker.header = msg.header
            cyl_marker.ns = "cone_centroids"
            cyl_marker.id = int(label)
            cyl_marker.type = Marker.CYLINDER
            cyl_marker.action = Marker.ADD
            cyl_marker.pose.position.x = float(centroid[0])
            cyl_marker.pose.position.y = float(centroid[1])
            cyl_marker.pose.position.z = float(centroid[2]) + 0.25  # lifted a bit
            cyl_marker.pose.orientation.w = 1.0
            cyl_marker.scale.x = 0.3
            cyl_marker.scale.y = 0.3
            cyl_marker.scale.z = 0.6
            cyl_marker.color = self.get_color(centroid[1])
            cyl_marker.lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()
            
            marker_array.markers.append(cyl_marker)
            self.get_logger().warn("marker published")

        # Publish markers
        if marker_array.markers:
            self.get_logger().info(f"Publishing {len(marker_array.markers)} cone markers.")
            self.marker_publisher.publish(marker_array)


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