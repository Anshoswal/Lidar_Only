# # # import rclpy
# # # import numpy as np
# # # import pandas as pd
# # # from rclpy.node import Node
# # # from sklearn.cluster import DBSCAN
# # # from sensor_msgs.msg import PointCloud
# # # from datetime import datetime
# # # from visualization_msgs.msg import Marker, MarkerArray
# # # from tensorflow import keras   
# # # import joblib                  


# # # class Cone_Detector(Node):
# # #     def __init__(self):
# # #         super().__init__('full_run')
# # #         self.subscriber = self.create_subscription(PointCloud, '/carmaker/pointcloud', self.overall_callback, 10)
# # #         self.publisher = self.create_publisher(MarkerArray, 'visualise_cones', 10)
# # #         self.model = keras.models.load_model("my_model.h5")

# # #         try:
# # #             self.scaler = joblib.load("scaler.pkl")  
# # #             self.get_logger().info("Loaded scaler for preprocessing")
# # #         except:
# # #             self.scaler = None
# # #             self.get_logger().warn("No scaler found. Make sure preprocessing matches training.")


# # #     def remove_z_threshold(self, pcd, z_threshold):
# # #         non_ground_mask = pcd[:, 2] > z_threshold
# # #         return pcd[non_ground_mask]

# # #     def overall_callback(self, msg):

# # #         pass

# # # ==============================================================================
# # # Lidar Cone Classifier ROS 2 Node
# # # ==============================================================================
# # # This node subscribes to a PointCloud, performs ground removal and clustering,
# # # and uses a pre-trained neural network to classify each cluster in real-time.
# # # The classification result is then visualized in RViz with different colors.
# # #
# # # Files required in the launch directory:
# # # - lidar_classifier.keras (The trained model)
# # # - scaler.joblib (The data scaler)
# # # - bin_edges.npy (The bin definitions for preprocessing)
# # # ==============================================================================

# # import rclpy
# # import numpy as np
# # import pandas as pd
# # import joblib
# # import tensorflow as tf
# # from rclpy.node import Node
# # from sklearn.cluster import DBSCAN

# # # ROS message imports
# # from sensor_msgs.msg import PointCloud
# # from visualization_msgs.msg import Marker, MarkerArray
# # from geometry_msgs.msg import Point
# # from std_msgs.msg import ColorRGBA
# # import rclpy.duration

# # # ==============================================================================
# # # Helper Class: LidarClassifier
# # # ==============================================================================
# # class LidarClassifier:
# #     """
# #     A class to load a trained lidar classification model and make predictions.
# #     This class handles the entire data preprocessing pipeline for a given cluster.
# #     """
# #     def __init__(self, model_path, scaler_path, bins_path, logger):
# #         self.logger = logger
# #         self.logger.info("Initializing LidarClassifier...")
# #         try:
# #             self.model = tf.keras.models.load_model(model_path)
# #             self.scaler = joblib.load(scaler_path)
# #             self.bin_edges = np.load(bins_path)
# #             self.num_bins = len(self.bin_edges) - 1
# #             self.logger.info("Classifier initialized successfully.")
# #         except Exception as e:
# #             self.logger.error(f"Error during classifier initialization: {e}")
# #             raise

# #     def _preprocess_data(self, cluster_df):
# #         """
# #         Private method to preprocess a cluster's data into a feature vector.
# #         This includes binning, pivoting, and scaling.
# #         """
# #         # Step 1: Divide the cluster's points into 50 bins based on z_coordinate
# #         df = cluster_df.copy()
# #         df['bin'] = pd.cut(df['z_coordinate'], bins=self.bin_edges, labels=False, include_lowest=True)
# #         df.dropna(subset=['bin'], inplace=True)
# #         df['bin'] = df['bin'].astype(int)

# #         # Step 2: Calculate the average intensity for each bin
# #         grouped = df.groupby(['cluster_id', 'bin'])['normalized_intensity'].mean()
# #         processed_df = grouped.unstack(level='bin', fill_value=0) # Use 0 for empty bins

# #         # Step 3: Ensure the feature vector has exactly 50 columns
# #         for i in range(self.num_bins):
# #             if i not in processed_df.columns:
# #                 processed_df[i] = 0
        
# #         # Sort columns to ensure consistent order before scaling
# #         processed_df = processed_df.reindex(sorted(processed_df.columns), axis=1)
        
# #         # Step 4: Scale the feature vector using the loaded scaler
# #         return self.scaler.transform(processed_df.values)

# #     def predict(self, cluster_df):
# #         """
# #         Makes a classification prediction on a DataFrame of points from one cluster.
# #         """
# #         if cluster_df.empty:
# #             return {"error": "Input data is empty."}
            
# #         # Create the feature vector from the raw cluster points
# #         processed_features = self._preprocess_data(cluster_df)
        
# #         # Use the neural network to predict
# #         prediction_proba = self.model.predict(processed_features, verbose=0)[0][0]
# #         predicted_class = 1 if prediction_proba > 0.5 else 0
        
# #         return {
# #             "predicted_class": predicted_class,
# #             "probability": float(prediction_proba)
# #         }

# # # ==============================================================================
# # # Main ROS 2 Node
# # # ==============================================================================
# # class ConeClassifierNode(Node):
# #     def __init__(self):
# #         super().__init__('cone_classifier_node')
        
# #         # === 1. LOAD THE TRAINED NEURAL NETWORK ===
# #         # The LidarClassifier class handles loading the model, scaler, and bins.
# #         # This is done only once when the node starts up for efficiency.
# #         try:
# #             self.classifier = LidarClassifier(
# #                 model_path='lidar_classifier.keras',
# #                 scaler_path='scaler.joblib',
# #                 bins_path='bin_edges.npy',
# #                 logger=self.get_logger()
# #             )
# #         except Exception:
# #             self.get_logger().error("Node startup failed: Could not initialize the classifier. Please ensure model files are present. Shutting down.")
# #             rclpy.shutdown()
# #             return
            
# #         # ROS subscriber for the point cloud
# #         self.subscriber = self.create_subscription(
# #             PointCloud, 
# #             '/carmaker/pointcloud', 
# #             self.process_cloud_callback, 
# #             10
# #         )
        
# #         # ROS publisher for visualization markers
# #         self.marker_publisher = self.create_publisher(MarkerArray, '/cone_markers', 10)

# #         self.get_logger().info("ConeClassifierNode started. Using NN to classify and visualize cones.")

# #     def get_color_from_prediction(self, predicted_class):
# #         """Assigns a color based on the NN's classification output."""
# #         if predicted_class == 0:
# #             # BLUE for class 0
# #             return ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.9)
# #         else:
# #             # YELLOW for class 1
# #             return ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.9)

# #     def process_cloud_callback(self, msg: PointCloud):
# #         # === 2. GET THE DATA CALLBACK ===
# #         if not msg.points:
# #             return
            
# #         # Clear previous markers for a clean visualization
# #         delete_marker = Marker(action=Marker.DELETEALL)
# #         self.marker_publisher.publish(MarkerArray(markers=[delete_marker]))
        
# #         pcd_xyz = np.array([[p.x, p.y, p.z] for p in msg.points])
# #         intensity_channel = next((ch for ch in msg.channels if ch.name == 'intensity'), None)
        
# #         if intensity_channel is None:
# #             self.get_logger().warn("'intensity' channel not found. Predictions may be inaccurate.")
# #             intensities = np.zeros(len(pcd_xyz))
# #         else:
# #             intensities = np.array(intensity_channel.values)

# #         pcd_full = np.c_[pcd_xyz, intensities]

# #         # === 3. REMOVE GROUND USING THRESHOLDING ===
# #         non_ground_mask = pcd_full[:, 2] > -0.1629 # Z-height threshold
# #         pcd = pcd_full[non_ground_mask]
        
# #         if pcd.shape[0] < 5:
# #             return

# #         # === 4. FORM CLUSTERS ===
# #         clusters = DBSCAN(min_samples=5, eps=0.3).fit(pcd[:, :2])
# #         labels = clusters.labels_
# #         unique_labels = set(labels)
        
# #         marker_array = MarkerArray()
        
# #         for label in unique_labels:
# #             if label == -1:  # -1 is noise in DBSCAN, so we skip it
# #                 continue

# #             cluster_points = pcd[labels == label]
            
# #             # Preprocessing: Normalize intensity for this specific cluster
# #             cluster_intensities = cluster_points[:, 3]
# #             min_i, max_i = np.min(cluster_intensities), np.max(cluster_intensities)
# #             normalized_intensities = (cluster_intensities - min_i) / (max_i - min_i) if (max_i - min_i) > 1e-6 else np.zeros_like(cluster_intensities)

# #             # Create a DataFrame in the format required by the classifier
# #             cluster_df = pd.DataFrame({
# #                 'cluster_id': label,
# #                 'z_coordinate': cluster_points[:, 2],
# #                 'normalized_intensity': normalized_intensities
# #             })
            
# #             # === 5. USE THE MODEL TO CLASSIFY ===
# #             # The classifier handles binning and prediction internally
# #             result = self.classifier.predict(cluster_df)
# #             predicted_class = result.get("predicted_class", -1) 

# #             # === 6. PUBLISH MARKER OF DIFFERENT COLOR ===
# #             centroid = np.mean(cluster_points[:, :3], axis=0)
            
# #             cyl_marker = Marker()
# #             cyl_marker.header = msg.header
# #             cyl_marker.ns = "classified_cones"
# #             cyl_marker.id = int(label)
# #             cyl_marker.type = Marker.CYLINDER
# #             cyl_marker.action = Marker.ADD
# #             cyl_marker.pose.position = Point(x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2]) + 0.25)
# #             cyl_marker.pose.orientation.w = 1.0
# #             cyl_marker.scale.x, cyl_marker.scale.y, cyl_marker.scale.z = 0.3, 0.3, 0.6
# #             cyl_marker.color = self.get_color_from_prediction(predicted_class)
# #             cyl_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
            
# #             marker_array.markers.append(cyl_marker)

# #         if marker_array.markers:
# #             self.marker_publisher.publish(marker_array)

# # def main(args=None):
# #     rclpy.init(args=args)
# #     node = ConeClassifierNode()
# #     if rclpy.ok():
# #         try:
# #             rclpy.spin(node)
# #         except KeyboardInterrupt:
# #             pass
# #         finally:
# #             node.destroy_node()
# #             rclpy.shutdown()

# # if __name__ == '__main__':
# #     main()

# import rclpy
# import numpy as np
# import pandas as pd
# import joblib
# import tensorflow as tf
# from rclpy.node import Node
# from sklearn.cluster import DBSCAN
# from ament_index_python.packages import get_package_share_directory
# import os

# # ROS message imports
# from sensor_msgs.msg import PointCloud
# from visualization_msgs.msg import Marker, MarkerArray
# from geometry_msgs.msg import Point
# from std_msgs.msg import ColorRGBA
# import rclpy.duration

# class LidarClassifier:
#     """ Loads the trained model and handles the preprocessing pipeline. """
#     def __init__(self, model_path, scaler_path, bins_path, logger):
#         self.logger = logger
#         self.logger.info("Initializing LidarClassifier...")
#         try:
#             # Use the .h5 file path here
#             self.model = tf.keras.models.load_model(model_path)
#             self.scaler = joblib.load(scaler_path)
#             self.bin_edges = np.load(bins_path)
#             self.num_bins = len(self.bin_edges) - 1
#             self.logger.info("Classifier initialized successfully.")
#         except Exception as e:
#             self.logger.error(f"Error during classifier initialization: {e}")
#             raise

#     def _preprocess_data(self, cluster_df):
#         """ Preprocesses a cluster's data into a feature vector. """
#         df = cluster_df.copy()
#         df['bin'] = pd.cut(df['z_coordinate'], bins=self.bin_edges, labels=False, include_lowest=True)
#         df.dropna(subset=['bin'], inplace=True)
#         df['bin'] = df['bin'].astype(int)

#         grouped = df.groupby(['cluster_id', 'bin'])['normalized_intensity'].mean()
#         processed_df = grouped.unstack(level='bin', fill_value=0)

#         for i in range(self.num_bins):
#             if i not in processed_df.columns:
#                 processed_df[i] = 0
        
#         processed_df = processed_df.reindex(sorted(processed_df.columns), axis=1)
#         return self.scaler.transform(processed_df.values)

#     def predict(self, cluster_df):
#         """ Makes a classification prediction on a cluster. """
#         processed_features = self._preprocess_data(cluster_df)
#         prediction_proba = self.model.predict(processed_features, verbose=0)[0][0]
#         predicted_class = 1 if prediction_proba > 0.5 else 0
#         return {"predicted_class": predicted_class}

# class ConeClassifierNode(Node):
#     def __init__(self):
#         super().__init__('cone_classifier_node')
        
#         # === 1. LOAD THE TRAINED NEURAL NETWORK ===
#         try:
#             package_share_path = get_package_share_directory('lidar_only')
#             MODEL_PATH = os.path.join(package_share_path, 'models', 'lidar_classifier.h5')
#             SCALER_PATH = os.path.join(package_share_path, 'models', 'scaler.joblib')
#             BINS_PATH = os.path.join(package_share_path, 'models', 'bin_edges.npy')
            
#             self.classifier = LidarClassifier(
#                 model_path=MODEL_PATH,
#                 scaler_path=SCALER_PATH,
#                 bins_path=BINS_PATH,
#                 logger=self.get_logger()
#             )
#         except Exception as e:
#             self.get_logger().error(f"Node startup failed: {e}. Shutting down.")
#             rclpy.shutdown()
#             return
            
#         self.subscriber = self.create_subscription(PointCloud, '/carmaker/pointcloud', self.process_cloud_callback, 10)
#         self.marker_publisher = self.create_publisher(MarkerArray, '/cone_markers', 10)
#         self.get_logger().info("ConeClassifierNode started and model loaded.")

#     def get_color_from_prediction(self, predicted_class):
#         return ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.9) if predicted_class == 1 else ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.9)

#     def process_cloud_callback(self, msg: PointCloud):
#         # === 2. GET DATA, 3. REMOVE GROUND, 4. FORM CLUSTERS ===
#         if not msg.points: return
#         delete_marker = Marker(action=Marker.DELETEALL)
#         self.marker_publisher.publish(MarkerArray(markers=[delete_marker]))
        
#         pcd_xyz = np.array([[p.x, p.y, p.z] for p in msg.points])
#         intensity_channel = next((ch for ch in msg.channels if ch.name == 'intensity'), None)
#         intensities = np.array(intensity_channel.values) if intensity_channel else np.zeros(len(pcd_xyz))
#         pcd_full = np.c_[pcd_xyz, intensities]

#         pcd = pcd_full[pcd_full[:, 2] > -0.1629]
#         if pcd.shape[0] < 5: return

#         clusters = DBSCAN(min_samples=5, eps=0.3).fit(pcd[:, :2])
#         labels = clusters.labels_
        
#         marker_array = MarkerArray()
#         for label in set(labels):
#             if label == -1: continue

#             cluster_points = pcd[labels == label]
            
#             # Normalize intensity for this cluster
#             cluster_intensities = cluster_points[:, 3]
#             min_i, max_i = np.min(cluster_intensities), np.max(cluster_intensities)
#             normalized_intensities = (cluster_intensities - min_i) / (max_i - min_i) if (max_i - min_i) > 1e-6 else np.zeros_like(cluster_intensities)

#             cluster_df = pd.DataFrame({
#                 'cluster_id': label, 'z_coordinate': cluster_points[:, 2], 'normalized_intensity': normalized_intensities
#             })
            
#             # === 5. USE MODEL TO CLASSIFY & 6. PUBLISH MARKER ===
#             result = self.classifier.predict(cluster_df)
#             predicted_class = result.get("predicted_class")

#             centroid = np.mean(cluster_points[:, :3], axis=0)
#             cyl_marker = Marker()
#             cyl_marker.header, cyl_marker.ns, cyl_marker.id = msg.header, "classified_cones", int(label)
#             cyl_marker.type, cyl_marker.action = Marker.CYLINDER, Marker.ADD
#             cyl_marker.pose.position = Point(x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2]) + 0.25)
#             cyl_marker.pose.orientation.w = 1.0
#             cyl_marker.scale.x, cyl_marker.scale.y, cyl_marker.scale.z = 0.3, 0.3, 0.6
#             cyl_marker.color = self.get_color_from_prediction(predicted_class)
#             cyl_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
#             marker_array.markers.append(cyl_marker)

#         if marker_array.markers: self.marker_publisher.publish(marker_array)

# def main(args=None):
#     rclpy.init(args=args)
#     node = ConeClassifierNode()
#     if rclpy.ok():
#         try: rclpy.spin(node)
#         except KeyboardInterrupt: pass
#         finally:
#             node.destroy_node()
#             rclpy.shutdown()

# if __name__ == '__main__':
#     main()

import rclpy
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from rclpy.node import Node
from sklearn.cluster import DBSCAN
from ament_index_python.packages import get_package_share_directory
import os

from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import rclpy.duration

class LidarClassifier:
    """ Loads the trained model and handles the full preprocessing pipeline. """
    def __init__(self, model_path, scaler_path, bins_path, feature_cols_path, logger):
        self.logger = logger
        self.logger.info("Initializing LidarClassifier...")
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.bin_edges = np.load(bins_path)
            self.feature_columns = np.load(feature_cols_path)
            self.num_bins = len(self.bin_edges) - 1
            self.logger.info(f"Classifier initialized. Expecting {len(self.feature_columns)} features.")
        except Exception as e:
            self.logger.error(f"Error during classifier initialization: {e}")
            raise

    def _preprocess_data(self, cluster_df):
        """ Preprocesses a cluster's data into the feature vector the model expects. """
        df = cluster_df.copy()
        df['bin'] = pd.cut(df['z_coordinate'], bins=self.bin_edges, labels=False, include_lowest=True)
        df.dropna(subset=['bin'], inplace=True)
        df['bin'] = df['bin'].astype(int)

        grouped = df.groupby(['cluster_id', 'bin'])['normalized_intensity'].mean()
        processed_df = grouped.unstack(level='bin', fill_value=0)
        
        # Create a full DataFrame with all 50 possible bins to handle any live data
        full_df = pd.DataFrame(0.0, index=processed_df.index, columns=range(self.num_bins))
        full_df.update(processed_df)
        
        # Select only the exact feature columns the scaler was trained on
        df_for_scaler = full_df[self.feature_columns]
        
        return self.scaler.transform(df_for_scaler.values)

    def predict(self, cluster_df):
        """ Makes a classification prediction on a cluster. """
        if cluster_df.empty: return {"error": "Input data is empty."}
        processed_features = self._preprocess_data(cluster_df)
        prediction_proba = self.model.predict(processed_features, verbose=0)[0][0]
        predicted_class = 1 if prediction_proba > 0.5 else 0
        return {"predicted_class": predicted_class}

class ConeClassifierNode(Node):
    def __init__(self):
        super().__init__('cone_classifier_node')
        
        try:
            package_share_path = get_package_share_directory('lidar_only')
            MODEL_PATH = os.path.join(package_share_path, 'models', 'lidar_classifier.h5')
            SCALER_PATH = os.path.join(package_share_path, 'models', 'scaler.joblib')
            BINS_PATH = os.path.join(package_share_path, 'models', 'bin_edges.npy')
            FEATURE_COLS_PATH = os.path.join(package_share_path, 'models', 'feature_columns.npy')

            self.classifier = LidarClassifier(
                model_path=MODEL_PATH,
                scaler_path=SCALER_PATH,
                bins_path=BINS_PATH,
                feature_cols_path=FEATURE_COLS_PATH,
                logger=self.get_logger()
            )
        except Exception as e:
            self.get_logger().error(f"Node startup failed: {e}. Shutting down.")
            rclpy.shutdown()
            return
            
        self.subscriber = self.create_subscription(PointCloud, '/carmaker/pointcloud', self.process_cloud_callback, 10)
        self.marker_publisher = self.create_publisher(MarkerArray, '/cone_markers', 10)
        self.get_logger().info("ConeClassifierNode started and model loaded successfully.")

    def get_color_from_prediction(self, predicted_class):
        return ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.9) if predicted_class == 1 else ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.9)

    def process_cloud_callback(self, msg: PointCloud):
        if not msg.points: return
        delete_marker = Marker(action=Marker.DELETEALL)
        self.marker_publisher.publish(MarkerArray(markers=[delete_marker]))
        
        pcd_xyz = np.array([[p.x, p.y, p.z] for p in msg.points])
        intensity_channel = next((ch for ch in msg.channels if ch.name == 'intensity'), None)
        intensities = np.array(intensity_channel.values) if intensity_channel else np.zeros(len(pcd_xyz))
        pcd_full = np.c_[pcd_xyz, intensities]

        pcd = pcd_full[pcd_full[:, 2] > -0.1629]
        if pcd.shape[0] < 5: return

        clusters = DBSCAN(min_samples=5, eps=0.3).fit(pcd[:, :2])
        labels = clusters.labels_
        
        marker_array = MarkerArray()
        for label in set(labels):
            if label == -1: continue

            cluster_points = pcd[labels == label]
            
            cluster_intensities = cluster_points[:, 3]
            min_i, max_i = np.min(cluster_intensities), np.max(cluster_intensities)
            normalized_intensities = (cluster_intensities - min_i) / (max_i - min_i) if (max_i - min_i) > 1e-6 else np.zeros_like(cluster_intensities)

            cluster_df = pd.DataFrame({
                'cluster_id': label, 'z_coordinate': cluster_points[:, 2], 'normalized_intensity': normalized_intensities
            })
            
            result = self.classifier.predict(cluster_df)
            predicted_class = result.get("predicted_class")

            centroid = np.mean(cluster_points[:, :3], axis=0)
            cyl_marker = Marker()
            cyl_marker.header, cyl_marker.ns, cyl_marker.id = msg.header, "classified_cones", int(label)
            cyl_marker.type, cyl_marker.action = Marker.CYLINDER, Marker.ADD
            cyl_marker.pose.position = Point(x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2]) + 0.25)
            cyl_marker.pose.orientation.w = 1.0
            cyl_marker.scale.x, cyl_marker.scale.y, cyl_marker.scale.z = 0.3, 0.3, 0.6
            cyl_marker.color = self.get_color_from_prediction(predicted_class)
            cyl_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
            marker_array.markers.append(cyl_marker)

        if marker_array.markers: self.marker_publisher.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = ConeClassifierNode()
    if rclpy.ok():
        try: rclpy.spin(node)
        except KeyboardInterrupt: pass
        finally:
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()