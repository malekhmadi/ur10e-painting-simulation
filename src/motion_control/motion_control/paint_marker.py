#!/usr/bin/env python3
"""Publish the simulated paint coverage from cal_pattern_ap4 as RViz markers,
so you can inspect coverage directly in the 3D scene alongside the robot."""

from pathlib import Path

import numpy as np
import trimesh
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from ament_index_python.packages import get_package_share_directory

from motion_control.cal_patten_ap4 import (
    read_mesh_scale_from_urdf,
    read_object_pose_from_launch,
    euler_to_quaternion,
    quaternion_to_matrix4,
    generate_surface_based_path,
    ROW_SPACING,
    ROBOT_POSITION,
    ANGLE_RANGES,
    LIFT_DISTANCE,
)


def align_z_to_vector(vector):
    """
    Build a quaternion (x, y, z, w) that rotates the +Z axis to align with
    the given vector — used so a cylinder marker's flat face lies tangent
    to the surface, oriented along the surface normal.
    """
    vector = vector / np.linalg.norm(vector)
    z_axis = np.array([0.0, 0.0, 1.0])

    # axis-angle between +Z and target vector
    axis = np.cross(z_axis, vector)
    axis_norm = np.linalg.norm(axis)
    dot = np.clip(np.dot(z_axis, vector), -1.0, 1.0)

    if axis_norm < 1e-9:
        # already aligned (or exactly opposite)
        if dot > 0:
            return [0.0, 0.0, 0.0, 1.0]
        else:
            return [1.0, 0.0, 0.0, 0.0]  # 180 deg flip around X

    axis = axis / axis_norm
    angle = np.arccos(dot)
    s = np.sin(angle / 2.0)
    return [axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)]


def build_paint_marker(point, normal, marker_id, frame_id='world',
                        radius=0.02, thickness=0.02, color=(0.1, 0.4, 0.9, 0.5)):
    """Build a single flat disk Marker (thin cylinder) tangent to the surface
    at `point`, oriented along `normal`."""
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = 'paint_coverage'
    marker.id = marker_id
    marker.type = Marker.CYLINDER
    marker.action = Marker.ADD

    marker.pose.position = Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
    qx, qy, qz, qw = align_z_to_vector(normal)
    marker.pose.orientation.x = qx
    marker.pose.orientation.y = qy
    marker.pose.orientation.z = qz
    marker.pose.orientation.w = qw

    marker.scale.x = radius * 2
    marker.scale.y = radius * 2
    marker.scale.z = thickness

    marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
    marker.lifetime.sec = 0  # 0 = persists until explicitly deleted/overwritten

    return marker


class PaintMarkerVisualizer(Node):
    def __init__(self):
        super().__init__('paint_marker_visualizer')

        # transient local QoS so RViz sees the markers even if it subscribes
        # slightly after this node publishes (standard for one-shot marker dumps)
        qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(MarkerArray, '/paint_coverage_markers', qos)

        points, normals = self._build_path()
        self._publish_markers(points, normals)

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
        )
        points = points + translation

        self.get_logger().info(f'Built {len(points)} paint markers')
        return points, normals

    def _publish_markers(self, points, normals):
        marker_array = MarkerArray()
        for i, (point, normal) in enumerate(zip(points, normals)):
            marker_array.markers.append(build_paint_marker(point, normal, marker_id=i))

        self.marker_pub.publish(marker_array)
        self.get_logger().info(f'Published {len(marker_array.markers)} markers to /paint_coverage_markers')


def main(args=None):
    rclpy.init(args=args)
    node = PaintMarkerVisualizer()
    rclpy.spin(node)  # keep alive so transient-local markers stay available
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()