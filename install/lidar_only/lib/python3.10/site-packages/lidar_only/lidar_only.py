import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
import numpy as np
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker, MarkerArray

class ConesDetector(Node):
    def __init__(self):
        super().__init__('cones_detector')
        self.subscriber = self.create_subscription(PointCloud, '/carmaker/pointcloud', self.cluster, 10)
        self.publisher = self.create_publisher(MarkerArray, '/visualise', 10)
        self.marker_id = 0

    def remove_ground_threshold(self, pc_data, z_thresh=-0.325):
        """Remove ground using z-thresholding"""
        if len(pc_data) < 10:
            return pc_data
        mask = pc_data[:, 2] > z_thresh
        return pc_data[mask]

    def remove_ground_ransac(self, pc_data):
        """Remove ground using RANSAC plane fitting"""
        X = pc_data[:, :2]
        y = pc_data[:, 2]
        if len(X) < 10:
            return pc_data
        ransac = RANSACRegressor(residual_threshold=0.02)
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_
        return pc_data[~inlier_mask]

    def cluster(self, msg):
        if not msg.points:
            return

        # Convert PointCloud to numpy
        pc_data = np.array([[p.x, p.y, p.z] for p in msg.points])
        intensities = np.array(msg.channels[0].values)
        intensities /= np.max(intensities) if np.max(intensities) != 0 else 1.0
        pc_data = np.column_stack((pc_data, intensities))

        # Remove ground
        pc_data = self.remove_ground_threshold(pc_data, z_thresh=-0.325)
        if pc_data.shape[0] == 0:
            return

        # Cluster points
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(pc_data[:, :2])
        labels = clustering.labels_

        # Clear old markers
        markerarr = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        markerarr.markers.append(delete_marker)
        self.marker_id = 0

        # Loop over clusters
        for label in set(labels):
            if label == -1:  # noise
                continue
            cluster = pc_data[labels == label]
            if cluster.shape[0] < 5:
                continue

            # Centroid
            x, y, z = np.mean(cluster[:, :3], axis=0)

            # Sizes
            x_size = np.max(cluster[:, 0]) - np.min(cluster[:, 0])
            y_size = np.max(cluster[:, 1]) - np.min(cluster[:, 1])
            z_size = np.max(cluster[:, 2]) - np.min(cluster[:, 2])

            # Marker
            marker = Marker()
            marker.header.frame_id = "Lidar_F"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "cones"
            marker.id = self.marker_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = float(z + z_size / 2.0)
            marker.pose.orientation.w = 1.0
            marker.scale.x = max(x_size, 0.2)
            marker.scale.y = max(y_size, 0.2)
            marker.scale.z = max(z_size, 0.31)
            marker.color.r, marker.color.g, marker.color.b = (0.0, 0.0, 1.0)
            marker.color.a = 1.0
            marker.lifetime = Duration(seconds=1.0).to_msg()

            markerarr.markers.append(marker)
            self.marker_id += 1

        self.publisher.publish(markerarr)


def main(args=None):
    rclpy.init(args=args)
    node = ConesDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
