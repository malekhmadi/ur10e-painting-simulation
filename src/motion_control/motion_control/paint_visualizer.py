#!/usr/bin/env python3
"""Publish the full painting path to RViz as a continuous colored ribbon,
with width matching ROW_SPACING — showing actual paint coverage as a
connected swath rather than discrete disks. All path-generation logic is
reused directly from cal_pattern_ap4.py."""

from pathlib import Path

import numpy as np
import trimesh
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

from ament_index_python.packages import get_package_share_directory

from motion_control.cal_patten_ap4 import (
    read_mesh_scale_from_urdf,
    read_object_pose_from_launch,
    euler_to_quaternion,
    quaternion_to_matrix4,
    generate_surface_based_path,
    build_ribbon_quads,
    ROW_SPACING,
    ROBOT_POSITION,
    ANGLE_RANGES,
    LIFT_DISTANCE,
    PAINT_DISTANCE,
)


class PaintPathVisualizer(Node):
    def __init__(self):
        super().__init__('paint_path_visualizer')

        qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(MarkerArray, '/paint_coverage_markers', qos)

        self.points, self.normals = self._build_path()

        # publish once immediately, then keep refreshing periodically
        self._publish_ribbon()
        self.timer = self.create_timer(2.0, self._publish_ribbon)

    def _build_path(self):
        scale = read_mesh_scale_from_urdf()

        mesh_path = Path(get_package_share_directory('environment')) / 'meshes' / 'obj.stl'
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        mesh.apply_scale(scale)

        translation, (roll, pitch, yaw) = read_object_pose_from_launch()
        quat = euler_to_quaternion(roll, pitch, yaw)
        mesh.apply_transform(quaternion_to_matrix4(quat))

        points, normals = generate_surface_based_path(
            mesh,
            row_spacing=ROW_SPACING,
            robot_position=ROBOT_POSITION,
            angle_ranges=ANGLE_RANGES,
            lift_distance=LIFT_DISTANCE,
            paint_distance=PAINT_DISTANCE,
        )
        points = points + translation

        self.get_logger().info(f'Built path with {len(points)} points, row_spacing={ROW_SPACING}')
        return points, normals

    def _publish_ribbon(self):
        quads, colors = build_ribbon_quads(self.points, self.normals, ROW_SPACING)

        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'paint_path'
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        for quad, color in zip(quads, colors):
            p0, p1, p2, p3 = quad
            triangle_points = [p0, p1, p2, p0, p2, p3]
            rgba = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
            for p in triangle_points:
                marker.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
                marker.colors.append(rgba)

        marker.lifetime.sec = 0

        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = PaintPathVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()