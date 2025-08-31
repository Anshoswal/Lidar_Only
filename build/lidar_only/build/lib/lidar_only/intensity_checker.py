import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud

class RawDataViewer(Node):
    def __init__(self):
        super().__init__('raw_data_viewer')
        self.subscriber = self.create_subscription(
            PointCloud, 
            '/carmaker/pointcloud', 
            self.data_callback, 
            10)
        self.frame_count = 0
        self.get_logger().info("Raw Data Viewer started. Waiting for messages...")

    def data_callback(self, msg):
        """
        This function is called for every message and prints its raw data.
        """
        self.frame_count += 1
        self.get_logger().info(f"--- FRAME {self.frame_count} | Total Points: {len(msg.points)} ---")

        # Check if there is an intensity channel
        has_intensity = msg.channels and msg.channels[0].name.lower() == 'intensity'

        # Loop through each point in the message
        for i in range(len(msg.points)):
            point = msg.points[i]
            
            # Get the corresponding intensity value
            # If no intensity channel exists, default to 0.0
            intensity = msg.channels[0].values[i] if has_intensity else 0.0
            
            # Print the raw values directly from the message
            self.get_logger().info(
                f"  Point {i}: x={point.x}, y={point.y}, z={point.z}, intensity={intensity}"
            )
        
        # We only want to see the first message, so we shut down the node.
        # Remove the two lines below if you want it to run continuously.
        self.get_logger().warn(">>> Printed first message. Shutting down to prevent spam. <<<")
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RawDataViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # The node might already be destroyed if it received a message,
        # so we check if it's still valid.
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()