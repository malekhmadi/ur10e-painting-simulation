"""ROS2 node to visualize walls in RViz and add them to the MoveIt planning scene."""
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
import math
from environment.wall import create_default_circle, create_default_wall


class WallVisualizer(Node):
    """Node that publishes wall markers to RViz and collision objects to MoveIt."""

    def __init__(self):
        super().__init__('wall_visualizer')

        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)
        self.scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)

        # Markers republish continuously (RViz needs periodic refresh)
        self.marker_timer = self.create_timer(1.0, self.publish_markers)
        # Collision objects only need to be sent once (MoveIt keeps them until told otherwise)
        self.scene_timer = self.create_timer(1.0, self.publish_collision_objects)
        self.scene_published = False

        self.frame_id = "base_link"

        self.object_type = 'circle'  # Default to circle; can be changed via parameter
        self.objects = self._build_objects()

        self.get_logger().info(f'Wall visualizer initialized for {self.object_type}')

    def _build_objects(self):
        if self.object_type == 'circle':
            return [create_default_circle()]
        return [create_default_wall()]

    def publish_markers(self):
        """Publish the selected object as an RViz marker (cosmetic only)."""
        for i, obj in enumerate(self.objects):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "walls"
            marker.id = i
            marker.action = Marker.ADD

            marker.pose.position.x = obj.x
            marker.pose.position.y = obj.y
            marker.pose.position.z = obj.z

            cos_half = math.cos(obj.orientation / 2)
            sin_half = math.sin(obj.orientation / 2)
            marker.pose.orientation.w = cos_half
            marker.pose.orientation.z = sin_half

            if self.object_type == 'circle':
                marker.type = Marker.CYLINDER
                marker.scale.x = obj.radius * 2.0
                marker.scale.y = obj.radius * 2.0
                marker.scale.z = obj.height
            else:
                marker.type = Marker.CUBE
                marker.scale.x = obj.length
                marker.scale.y = obj.thickness
                marker.scale.z = obj.height

            marker.color.r = 0.8
            marker.color.g = 0.2
            marker.color.b = 0.2
            marker.color.a = 0.7

            self.marker_pub.publish(marker)

    def publish_collision_objects(self):
        """Publish the selected object as a MoveIt collision object (once)."""
        if self.scene_published:
            return

        scene = PlanningScene()
        scene.is_diff = True

        for i, obj in enumerate(self.objects):
            collision_obj = CollisionObject()
            collision_obj.header.frame_id = self.frame_id
            collision_obj.id = f"{self.object_type}_{i}"

            primitive = SolidPrimitive()
            if self.object_type == 'circle':
                primitive.type = SolidPrimitive.CYLINDER
                primitive.dimensions = [obj.height, obj.radius]
            else:
                primitive.type = SolidPrimitive.BOX
                primitive.dimensions = [obj.length, obj.thickness, obj.height]

            pose = Pose()
            pose.position.x = obj.x
            pose.position.y = obj.y
            pose.position.z = obj.z
            pose.orientation.z = math.sin(obj.orientation / 2)
            pose.orientation.w = math.cos(obj.orientation / 2)

            collision_obj.primitives.append(primitive)
            collision_obj.primitive_poses.append(pose)
            collision_obj.operation = CollisionObject.ADD

            scene.world.collision_objects.append(collision_obj)

        self.scene_pub.publish(scene)
        self.get_logger().info(f'Published {len(self.objects)} {self.object_type}(s) to planning scene')
        self.scene_published = True


def main(args=None):
    rclpy.init(args=args)
    node = WallVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

