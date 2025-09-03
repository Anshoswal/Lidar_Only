import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# The message type for both subscribing and publishing
from visualization_msgs.msg import Marker, MarkerArray

class MarkerRepublisher(Node):
    """
    This node subscribes to a MarkerArray on '/carmaker/ObjectList',
    extracts the position and color of each marker, and then republishes
    a new, simplified sphere marker with the IDENTICAL position and color
    to the '/my_visualization' topic.
    """
    def __init__(self):
        super().__init__('marker_republisher_node')

        # QoS profile to match the reliability of rosbag publishers
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # --- SUBSCRIBER ---
        # Listens to the original marker data from your bag file
        self.marker_subscriber = self.create_subscription(
            MarkerArray,
            '/carmaker/ObjectList',
            self.marker_callback,
            qos_profile
        )
        
        # --- PUBLISHER ---
        # Publishes our new, custom markers to a clean topic for RViz
        self.new_marker_publisher = self.create_publisher(
            MarkerArray, 
            '/my_visualization',
            10
        )
        
        self.get_logger().info("Marker Republisher Node has started. ✅")
        self.get_logger().info("Subscribing to '/carmaker/ObjectList'")
        self.get_logger().info("Publishing to '/my_visualization'")


    def marker_callback(self, incoming_msg: MarkerArray):
        """
        This function is called every time a message is received.
        It processes the data and publishes the new markers.
        """
        # Create a new, empty MarkerArray to hold our custom markers
        new_marker_array = MarkerArray()

        # Loop through every single marker that was in the received message
        for original_marker in incoming_msg.markers:
            # Ignore any instructions to delete markers
            if original_marker.action != Marker.ADD:
                continue

            # --- Create our new marker ---
            new_marker = Marker()
            
            # Basic marker setup
            new_marker.header.frame_id = original_marker.header.frame_id
            new_marker.header.stamp = self.get_clock().now().to_msg()
            new_marker.ns = "republished_spheres"
            new_marker.id = original_marker.id
            new_marker.type = Marker.SPHERE
            new_marker.action = Marker.ADD
            new_marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
            
            # --- SETTING THE POSITION (X, Y, Z) ---
            # This line copies the EXACT x, y, z position from the original marker
            # to our new marker.
            new_marker.pose.position = original_marker.pose.position
            
            # Set a default orientation, as spheres are symmetrical
            new_marker.pose.orientation.w = 1.0

            # Set a fixed size for our sphere for visual clarity
            new_marker.scale.x = 0.5
            new_marker.scale.y = 0.5
            new_marker.scale.z = 0.5

            # --- SETTING THE COLOR (R, G, B) ---
            # This line copies the EXACT r, g, b, and a (transparency) color
            # from the original marker to our new marker.
            new_marker.color = original_marker.color
            
            # Add our completed new marker to the array
            new_marker_array.markers.append(new_marker)

        # Publish the full array of newly created markers
        if new_marker_array.markers:
            self.new_marker_publisher.publish(new_marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = MarkerRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
