# import atexit
# import csv
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import PointCloud
# from sklearn.cluster import DBSCAN
# from sklearn.linear_model import RANSACRegressor
# from sklearn.neighbors import KDTree
# import os


# class ClusterProjectorAndLogger(Node):
#     """ROS2 node that removes ground, clusters points, fits a plane per cluster (via SVD),
#     projects all original cluster points onto that plane, and logs projected points to CSV.
#     Also generates per-cluster triangular raster CSVs.
#     """

#     def __init__(self):
#         super().__init__('cluster_projector_and_logger')

#         # Parameters
#         self.declare_parameter('voxel_size', 0.05)
#         self.declare_parameter('max_downsample_points', 3000)
#         self.declare_parameter('dbscan_eps', 0.5)
#         self.declare_parameter('dbscan_min_samples', 2)
#         self.declare_parameter('ransac_threshold', 0.05)
#         self.declare_parameter('max_ransac_trials', 100)

#         self.voxel_size = float(self.get_parameter('voxel_size').value)
#         self.max_downsample_points = int(self.get_parameter('max_downsample_points').value)
#         self.dbscan_eps = float(self.get_parameter('dbscan_eps').value)
#         self.dbscan_min_samples = int(self.get_parameter('dbscan_min_samples').value)
#         self.ransac_threshold = float(self.get_parameter('ransac_threshold').value)
#         self.max_ransac_trials = int(self.get_parameter('max_ransac_trials').value)

#         # Subscriber
#         self.subscriber = self.create_subscription(
#             PointCloud, '/carmaker/pointcloud', self.process_cloud_callback, 10)

#         # CSV setup
#         self.csv_path = 'projected_points.csv'
#         self.csv_file = open(self.csv_path, 'w', newline='')
#         atexit.register(self._safe_close_csv)
#         self.csv_writer = csv.writer(self.csv_file)
#         self.csv_writer.writerow(['frame', 'cluster_id', 'proj_x', 'proj_y', 'proj_z', 'intensity'])

#         self.frame_count = 0
#         self.get_logger().info('Node started — logging to %s' % self.csv_path)

#     def _safe_close_csv(self):
#         try:
#             if hasattr(self, 'csv_file') and not self.csv_file.closed:
#                 self.csv_file.close()
#         except Exception:
#             pass

#     def destroy_node(self):
#         self._safe_close_csv()
#         return super().destroy_node()

#     # ---------------- utility helpers -----------------
#     def voxel_downsample(self, points, voxel_size, max_points=None):
#         if points.shape[0] == 0:
#             return np.empty((0, points.shape[1])), np.empty((0,), dtype=int)
#         vox = np.floor(points[:, :3] / voxel_size).astype(np.int64)
#         _, idx, inv = np.unique(vox, axis=0, return_index=True, return_inverse=True)

#         reps = []
#         rep_indices = []
#         for i in range(len(idx)):
#             inds = np.where(inv == i)[0]
#             centroid = points[inds, :3].mean(axis=0)
#             reps.append(centroid)
#             rep_indices.append(inds[0])
#         reps = np.vstack(reps)
#         rep_indices = np.array(rep_indices, dtype=int)

#         if max_points is not None and reps.shape[0] > max_points:
#             chosen = np.random.choice(reps.shape[0], size=max_points, replace=False)
#             reps = reps[chosen]
#             rep_indices = rep_indices[chosen]

#         return reps, rep_indices

#     def remove_ground_ransac(self, pc_data):
#         xyz = pc_data[:, :3]
#         if xyz.shape[0] < 30:
#             return pc_data
#         sample_size = min(2000, xyz.shape[0])
#         subset_idx = np.random.choice(xyz.shape[0], size=sample_size, replace=False)
#         try:
#             ransac = RANSACRegressor(residual_threshold=self.ransac_threshold,
#                                      max_trials=self.max_ransac_trials)
#             ransac.fit(xyz[subset_idx, :2], xyz[subset_idx, 2])
#             inlier_mask = ransac.predict(xyz[:, :2])
#             residuals = np.abs(xyz[:, 2] - inlier_mask)
#             non_ground_mask = residuals > self.ransac_threshold
#             return pc_data[non_ground_mask]
#         except Exception as e:
#             self.get_logger().warning(f'RANSAC failed: {e} — skipping ground removal')
#             return pc_data

#     def fit_plane_svd(self, points_xyz):
#         if points_xyz.shape[0] < 3:
#             return None
#         centroid = points_xyz.mean(axis=0)
#         cov = points_xyz - centroid
#         try:
#             _, _, vh = np.linalg.svd(cov, full_matrices=False)
#             normal = vh[-1, :]
#             norm = np.linalg.norm(normal)
#             if norm < 1e-8:
#                 return None
#             return centroid, normal / norm
#         except Exception:
#             return None

#     # -------------- main callback ------------------
#     def process_cloud_callback(self, msg):
#         self.frame_count += 1
#         if not msg.points:
#             return

#         pc_xyz = np.array([[p.x, p.y, p.z] for p in msg.points])
#         if msg.channels and msg.channels[0].name.lower() == 'intensity':
#             intensities = np.array(msg.channels[0].values).reshape(-1, 1)
#         else:
#             intensities = np.zeros((pc_xyz.shape[0], 1))
#         pc_data = np.hstack((pc_xyz, intensities))

#         pc_nog = self.remove_ground_ransac(pc_data)
#         if pc_nog.shape[0] < 5:
#             return

#         reps, rep_indices = self.voxel_downsample(pc_nog, self.voxel_size, self.max_downsample_points)
#         if reps.shape[0] == 0:
#             return

#         db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
#         rep_labels = db.fit_predict(reps[:, :2])
#         unique = set(rep_labels)
#         if -1 in unique:
#             unique.remove(-1)
#         if not unique:
#             return

#         self.get_logger().info(f'Frame {self.frame_count}: {len(unique)} clusters (downsampled)')

#         kdt = KDTree(reps[:, :2])
#         dists, nearest_rep = kdt.query(pc_nog[:, :2], k=1)
#         assigned_labels = rep_labels[nearest_rep.flatten()]

#         rows_to_write = []
#         for label in sorted(unique):
#             mask = assigned_labels == label
#             if np.count_nonzero(mask) < 3:
#                 continue

#             pts_full = pc_nog[mask][:, :3]
#             plane = self.fit_plane_svd(pts_full)
#             if plane is None:
#                 continue
#             plane_point, plane_normal = plane

#             vecs = pts_full - plane_point
#             distances = np.dot(vecs, plane_normal)
#             projected = pts_full - np.outer(distances, plane_normal)

#             intens = pc_nog[mask][:, 3]

#             # write to main CSV
#             for p, inten in zip(projected, intens):
#                 rows_to_write.append([
#                     self.frame_count,
#                     int(label),
#                     f'{p[0]:.4f}',
#                     f'{p[1]:.4f}',
#                     f'{p[2]:.4f}',
#                     f'{inten:.4f}'
#                 ])

#             # write triangular raster CSV
#             try:
#                 self.rasterize_cone_to_csv(
#                     projected_pts=projected,
#                     intensities=intens,
#                     plane_point=plane_point,
#                     plane_normal=plane_normal,
#                     frame_idx=self.frame_count,
#                     label=label,
#                     grid_res=self.voxel_size,
#                     csv_folder='cone_csvs'
#                 )
#             except Exception as e:
#                 self.get_logger().warning(f"Failed to rasterize cone for frame {self.frame_count}, label {label}: {e}")

#         if rows_to_write:
#             try:
#                 self.csv_writer.writerows(rows_to_write)
#                 if self.frame_count % 10 == 0:
#                     self.csv_file.flush()
#             except Exception as e:
#                 self.get_logger().warning(f'Failed to write CSV: {e}')

#     # ----------------- rasterize to CSV helper -----------------
#     def rasterize_cone_to_csv(self, projected_pts, intensities, plane_point, plane_normal,
#                                frame_idx=0, label=0, grid_res=0.02, csv_folder='cone_csvs',
#                                max_nn_distance=None, agg='nearest'):
#         if projected_pts.shape[0] == 0:
#             return None

#         centroid = projected_pts.mean(axis=0)
#         v = centroid - plane_point
#         v = v - np.dot(v, plane_normal) * plane_normal
#         vy_norm = np.linalg.norm(v)
#         if vy_norm < 1e-8:
#             arbitrary = np.array([1.0, 0.0, 0.0])
#             v = arbitrary - np.dot(arbitrary, plane_normal) * plane_normal
#             vy_norm = np.linalg.norm(v)
#             if vy_norm < 1e-8:
#                 return None
#         y_axis = v / vy_norm
#         x_axis = np.cross(plane_normal, y_axis)
#         x_axis /= np.linalg.norm(x_axis)

#         rel = projected_pts - plane_point
#         coords2d = np.column_stack((rel.dot(x_axis), rel.dot(y_axis)))

#         if np.median(coords2d[:, 1]) < 0:
#             coords2d[:, 1] *= -1
#             coords2d[:, 0] *= -1

#         mask_forward = coords2d[:, 1] >= 0
#         if np.count_nonzero(mask_forward) == 0:
#             return None
#         coords2d = coords2d[mask_forward]
#         intensities = intensities[mask_forward]

#         max_y = coords2d[:, 1].max()
#         if max_y <= 1e-6:
#             return None
#         max_half_width = np.max(np.abs(coords2d[:, 0]))
#         k = max_half_width / max_y if max_y > 0 else 0.0

#         y_steps = int(np.ceil(max_y / grid_res)) + 1
#         x_half = max_half_width
#         x_steps = int(np.ceil((2 * x_half) / grid_res)) + 1

#         xs = (np.arange(x_steps) * grid_res) - x_half
#         ys = (np.arange(y_steps) * grid_res)
#         gx, gy = np.meshgrid(xs + grid_res/2.0, ys + grid_res/2.0)
#         H, W = gy.shape

#         kdt = KDTree(coords2d)
#         cell_points = np.column_stack((gx.ravel(), gy.ravel()))
#         dists, inds = kdt.query(cell_points, k=1)
#         dists = dists.ravel()
#         inds = inds.ravel()

#         if max_nn_distance is None:
#             max_nn_distance = np.sqrt(2) * grid_res * 1.5

#         img_flat = np.zeros(H * W, dtype=float)

#         abs_x = np.abs(cell_points[:, 0])
#         yy = cell_points[:, 1]
#         inside_triangle = (yy >= 0) & (abs_x <= (k * yy))
#         valid_cells = inside_triangle & (dists <= max_nn_distance)

#         if agg in ['nearest', 'max', 'mean']:
#             img_flat[valid_cells] = intensities[inds[valid_cells]]
#         else:
#             img_flat[valid_cells] = intensities[inds[valid_cells]]

#         img = img_flat.reshape(H, W)

#         os.makedirs(csv_folder, exist_ok=True)
#         csv_path = os.path.join(csv_folder, f'cone_f{frame_idx}_l{label}.csv')

#         x_centers = xs + grid_res/2.0
#         y_centers = ys + grid_res/2.0
#         try:
#             with open(csv_path, 'w', newline='') as cf:
#                 writer = csv.writer(cf)
#                 header = [''] + [f'{xc:.4f}' for xc in x_centers]
#                 writer.writerow(header)
#                 for yi, yv in enumerate(y_centers):
#                     row_vals = [f'{yv:.4f}'] + [f'{val:.4f}' for val in img[yi, :]]
#                     writer.writerow(row_vals)
#         except Exception as e:
#             self.get_logger().warning(f'Failed to write cone CSV {csv_path}: {e}')


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
