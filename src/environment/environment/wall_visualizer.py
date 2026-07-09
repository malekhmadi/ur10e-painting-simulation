"""ROS2 node to visualize walls in RViz and add them to the MoveIt planning scene."""
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
import math
from environment.wall import Wall


class WallVisualizer(Node):
    """Node that publishes wall markers to RViz and collision objects to MoveIt."""

    def __init__(self):
        super().__init__('wall_visualizer')

        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)
        self.scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)

        # Markers republish continuously (RViz needs periodic refresh)
        self.marker_timer = self.create_timer(1.0, self.publish_markers)
        # Collision objects only need to be sent once (MoveIt keeps them until told otherwise)
        #self.scene_timer = self.create_timer(1.0, self.publish_collision_objects)
        #self.scene_published = False

        self.frame_id = "base_link"  # match this to your robot's planning frame

        self.walls = [
            Wall(x=0.0, y=0.8, z=1.3, length=0.6, height=0.6, thickness=0.03, orientation=0),
        ]

        self.get_logger().info('Wall visualizer initialized')

    def publish_markers(self):
        """Publish all walls as RViz markers (cosmetic only)."""
        for i, wall in enumerate(self.walls):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "walls"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = wall.x
            marker.pose.position.y = wall.y
            marker.pose.position.z = wall.z

            cos_half = math.cos(wall.orientation / 2)
            sin_half = math.sin(wall.orientation / 2)
            marker.pose.orientation.w = cos_half
            marker.pose.orientation.z = sin_half

            marker.scale.x = wall.length
            marker.scale.y = wall.thickness
            marker.scale.z = wall.height

            marker.color.r = 0.8
            marker.color.g = 0.2
            marker.color.b = 0.2
            marker.color.a = 0.7

            self.marker_pub.publish(marker)

""" def publish_collision_objects(self):
        Publish all walls as MoveIt collision objects (once).
        if self.scene_published:
            return

        scene = PlanningScene()
        scene.is_diff = True

        for i, wall in enumerate(self.walls):
            obj = CollisionObject()
            obj.header.frame_id = self.frame_id
            obj.id = f"wall_{i}"

            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [wall.length, wall.thickness, wall.height]

            pose = Pose()
            pose.position.x = wall.x
            pose.position.y = wall.y
            pose.position.z = wall.z
            pose.orientation.z = math.sin(wall.orientation / 2)
            pose.orientation.w = math.cos(wall.orientation / 2)

            obj.primitives.append(primitive)
            obj.primitive_poses.append(pose)
            obj.operation = CollisionObject.ADD

            scene.world.collision_objects.append(obj)

        self.scene_pub.publish(scene)
        self.get_logger().info(f'Published {len(self.walls)} wall(s) to planning scene')
        self.scene_published = True
"""

def main(args=None):
    rclpy.init(args=args)
    node = WallVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()