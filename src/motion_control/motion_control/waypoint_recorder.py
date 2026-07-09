"""Record waypoint coordinates from RViz interactive markers."""
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from geometry_msgs.msg import Pose
from interactive_markers.interactive_marker_server import InteractiveMarkerServer


class WaypointRecorder(Node):
    def __init__(self):
        super().__init__('waypoint_recorder')
        
        self.server = InteractiveMarkerServer(self, "waypoint_markers")
        self.waypoints = []
        self.waypoint_count = 0
        
        self.get_logger().info('Waypoint Recorder started. Drag the green sphere to record waypoints.')
        self.get_logger().info('Release the mouse to save each waypoint.')
        self.get_logger().info('Press Ctrl+C when done.')
        
        self._create_add_waypoint_marker()

    def _create_add_waypoint_marker(self):
        """Create a marker to add new waypoints."""
        marker = InteractiveMarker()
        marker.header.frame_id = "base_link"
        marker.pose.position.x = 0.5
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.6
        marker.name = "add_waypoint"
        marker.description = "Drag to position, release to record"

        control = InteractiveMarkerControl()
        control.name = "move_3d"
        control.interaction_mode = InteractiveMarkerControl.MOVE_3D
        control.always_visible = True
        
        sphere = Marker()
        sphere.type = Marker.SPHERE
        sphere.scale.x = 0.1
        sphere.scale.y = 0.1
        sphere.scale.z = 0.1
        sphere.color.r = 0.0
        sphere.color.g = 1.0
        sphere.color.b = 0.0
        sphere.color.a = 0.8
        control.markers.append(sphere)

        marker.controls.append(control)
        self.server.insert(marker, feedback_callback=self._marker_feedback)
        self.server.applyChanges()

    def _marker_feedback(self, feedback):
        """Handle marker feedback."""
        if feedback.event_type == feedback.MOUSE_UP:
            pos = feedback.pose.position
            waypoint = ([round(pos.x, 4), round(pos.y, 4), round(pos.z, 4)], [0.0, 0.707, 0.0, 0.707])
            self.waypoints.append(waypoint)
            self.waypoint_count += 1
            
            self.get_logger().info(f'Waypoint {self.waypoint_count}: [{pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f}]')

    def print_code(self):
        """Print waypoints as Python code."""
        print("\n" + "="*70)
        print("WAYPOINTS RECORDED - COPY THIS CODE INTO move_to_pose.py:")
        print("="*70 + "\n")
        
        print("# In __init__, add:")
        print("self.waypoints = [")
        for wp in self.waypoints:
            print(f"    ({wp[0]}, {wp[1]}),")
        print("]\n")
        
        print("# In main(), replace the move_to calls with:")
        print("for i, waypoint in enumerate(node.waypoints):")
        print("    node.get_logger().info(f'Moving to waypoint {i+1}...')")
        print("    node.move_to(waypoint)\n")
        
        print("="*70 + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = WaypointRecorder()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.print_code()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()