import rclpy
from rclpy.node import Node
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
import numpy as np
from sensor_msgs.msg import PointCloud
import csv

class ClusterProjectorAndLogger(Node):
    def __init__(self):
        super().__init__('cluster_projector_and_logger')
        self.subscriber = self.create_subscription(
            PointCloud, '/carmaker/pointcloud', self.process_cloud_callback, 10)
        self.frame_count = 0

        # CSV setup
        self.csv_file = open("projected_triangle_grid.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["frame", "cluster_id", "x_local", "y_local", "intensity"])
        self.get_logger().info("Node started. Logging rasterized clusters to projected_triangle_grid.csv")

    def __del__(self):
        if hasattr(self, "csv_file") and not self.csv_file.closed:
            self.csv_file.close()

    def remove_ground_ransac(self, pc_data):
        xyz_data = pc_data[:, :3]
        if xyz_data.shape[0] < 10:
            return pc_data
        try:
            ransac = RANSACRegressor(residual_threshold=0.05)
            ransac.fit(xyz_data[:, :2], xyz_data[:, 2])
            return pc_data[~ransac.inlier_mask_]
        except ValueError as e:
            self.get_logger().warn(f"RANSAC failed: {e}. Returning no points.")
            return np.array([])

    def fit_plane_svd(self, points_xyz):
        if points_xyz.shape[0] < 3:
            return None
        centroid = points_xyz.mean(axis=0)
        cov = points_xyz - centroid
        try:
            _, _, vh = np.linalg.svd(cov, full_matrices=False)
            normal = vh[-1, :]
            norm = np.linalg.norm(normal)
            if norm < 1e-8:
                return None
            return centroid, normal / norm
        except Exception:
            return None

    def rasterize_triangle_grid(self, points_3d, intensities, plane_point, plane_normal, grid_res=0.02):
        """Convert projected 3D points into a triangular 2D grid."""
        if points_3d.shape[0] == 0:
            return None, None, None

        # Local axes on plane
        v = points_3d.mean(axis=0) - plane_point
        v -= np.dot(v, plane_normal) * plane_normal
        if np.linalg.norm(v) < 1e-8:
            v = np.array([1.0, 0.0, 0.0])
            v -= np.dot(v, plane_normal) * plane_normal
        y_axis = v / np.linalg.norm(v)
        x_axis = np.cross(plane_normal, y_axis)
        x_axis /= np.linalg.norm(x_axis)

        # Project points to local 2D plane
        rel = points_3d - plane_point
        coords2d = np.column_stack((rel.dot(x_axis), rel.dot(y_axis)))

        # Keep only forward points (y >=0)
        mask_forward = coords2d[:, 1] >= 0
        coords2d = coords2d[mask_forward]
        intensities = intensities[mask_forward]

        if coords2d.shape[0] == 0:
            return None, None, None

        max_y = coords2d[:, 1].max()
        max_half_width = np.max(np.abs(coords2d[:, 0]))
        k = max_half_width / max_y if max_y > 0 else 0.0

        y_steps = int(np.ceil(max_y / grid_res)) + 1
        x_half = max_half_width
        x_steps = int(np.ceil((2 * x_half) / grid_res)) + 1

        xs = (np.arange(x_steps) * grid_res) - x_half + grid_res/2.0
        ys = (np.arange(y_steps) * grid_res) + grid_res/2.0
        gx, gy = np.meshgrid(xs, ys)
        H, W = gy.shape

        # Fill grid using nearest neighbor
        from sklearn.neighbors import KDTree
        kdt = KDTree(coords2d)
        cell_points = np.column_stack((gx.ravel(), gy.ravel()))
        dists, inds = kdt.query(cell_points, k=1)
        dists = dists.ravel()
        inds = inds.ravel()

        # Only fill cells inside triangle
        abs_x = np.abs(cell_points[:, 0])
        yy = cell_points[:, 1]
        inside_triangle = (yy >= 0) & (abs_x <= (k * yy))
        img_flat = np.zeros(H * W)
        img_flat[inside_triangle] = intensities[inds[inside_triangle]]
        img = img_flat.reshape(H, W)

        return img, xs, ys

    def process_cloud_callback(self, msg):
        self.frame_count += 1
        if not msg.points:
            return

        # Extract points and intensity
        pc_xyz = np.array([[p.x, p.y, p.z] for p in msg.points])
        if msg.channels and msg.channels[0].name.lower() == 'intensity':
            intensities = np.array(msg.channels[0].values)
        else:
            intensities = np.zeros(pc_xyz.shape[0])

        pc_data = np.hstack((pc_xyz, intensities.reshape(-1,1)))

        # Remove ground
        pc_nog = self.remove_ground_ransac(pc_data)
        if pc_nog.shape[0] < 5:
            return

        # DBSCAN clustering
        labels = DBSCAN(eps=0.5, min_samples=2).fit_predict(pc_nog[:, :2])
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        if not unique_labels:
            return

        self.get_logger().info(f"Frame {self.frame_count}: {len(unique_labels)} clusters")

        for label in unique_labels:
            cluster_points = pc_nog[labels == label]
            pts_3d = cluster_points[:, :3]
            intens = cluster_points[:, 3]

            # Fit plane
            plane = self.fit_plane_svd(pts_3d)
            if plane is None:
                continue
            plane_point, plane_normal = plane

            # Project points onto plane
            vecs = pts_3d - plane_point
            distances = np.dot(vecs, plane_normal)
            projected_pts = pts_3d - np.outer(distances, plane_normal)

            # Rasterize into triangle grid
            img, xs, ys = self.rasterize_triangle_grid(projected_pts, intens, plane_point, plane_normal)
            if img is None:
                continue

            # Write grid to CSV
            H, W = img.shape
            for i in range(H):
                for j in range(W):
                    self.csv_writer.writerow([
                        self.frame_count,
                        int(label),
                        f"{xs[j]:.4f}",
                        f"{ys[i]:.4f}",
                        f"{img[i,j]:.4f}"
                    ])
        # Flush occasionally
        if self.frame_count % 10 == 0:
            self.csv_file.flush()


def main(args=None):
    rclpy.init(args=args)
    node = ClusterProjectorAndLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()