#!/usr/bin/env python3
"""Paint a mesh-based object using a scan pattern derived from its geometry."""

from pathlib import Path
import xml.etree.ElementTree as ET

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.duration import Duration

from ament_index_python.packages import get_package_share_directory
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from pymoveit2 import MoveIt2
from pymoveit2.robots import ur


class ObjectModel:
    """Simple container for an object's pose and dimensions."""

    def __init__(self, x, y, z, length, height, thickness, orientation=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.length = length
        self.height = height
        self.thickness = thickness
        self.orientation = orientation


class PaintObject(Node):
    def __init__(self):
        super().__init__('paint_object')

        callback_group = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur.joint_names(),
            base_link_name=ur.base_link_name(),
            end_effector_name=ur.end_effector_name(),
            group_name=ur.MOVE_GROUP_ARM,
            callback_group=callback_group,
        )

        # TF setup — needed to look up where object_link actually is in the world
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        placement = self._lookup_object_transform()
        self.object_model = self._load_object_from_urdf(placement)
        self.scan_path = self._build_object_path(self.object_model)

    def _lookup_object_transform(self, parent_frame='world', child_frame='object_link', timeout_sec=10.0):
        """Block briefly, spinning, until the live TF transform for object_link is available."""
        deadline = self.get_clock().now() + Duration(seconds=timeout_sec)

        while rclpy.ok() and self.get_clock().now() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    parent_frame, child_frame, rclpy.time.Time()
                )
                t = transform.transform.translation
                self.get_logger().info(
                    f'Found live transform {parent_frame} -> {child_frame}: '
                    f'x={t.x:.3f}, y={t.y:.3f}, z={t.z:.3f}'
                )
                return (t.x, t.y, t.z)
            except (LookupException, ConnectivityException, ExtrapolationException):
                rclpy.spin_once(self, timeout_sec=0.2)

        raise RuntimeError(
            f'Could not find transform {parent_frame} -> {child_frame} after {timeout_sec}s. '
            f'Is object_tf.launch.py running?'
        )

    def _load_object_from_urdf(self, placement):
        """Load the object mesh from the URDF and derive its dimensions and center with trimesh."""
        try:
            import trimesh
        except ImportError as exc:
            raise RuntimeError(
                "The 'trimesh' Python package is required. Install it with 'pip install trimesh'."
            ) from exc

        urdf_path = Path(get_package_share_directory('environment')) / 'urdf' / 'obj.urdf'
        if not urdf_path.exists():
            raise FileNotFoundError(f'Object URDF not found: {urdf_path}')

        tree = ET.parse(urdf_path)
        root = tree.getroot()

        mesh_elem = None
        for element in root.iter('mesh'):
            mesh_elem = element
            break

        if mesh_elem is None:
            raise ValueError(f'No mesh tag found in URDF: {urdf_path}')

        filename = mesh_elem.get('filename', '')
        scale_attr = mesh_elem.get('scale', '1 1 1')
        scale = [float(value) for value in scale_attr.split()]
        if len(scale) != 3:
            scale = [0.001, 0.001, 0.001]

        mesh_path = self._resolve_mesh_path(filename)
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        extents = mesh.bounding_box.extents
        length = extents[0] * scale[0]
        thickness = extents[1] * scale[1]
        height = extents[2] * scale[2]

        local_centroid = mesh.bounding_box.centroid
        center_offset_x = local_centroid[0] * scale[0]
        center_offset_y = local_centroid[1] * scale[1]
        center_offset_z = local_centroid[2] * scale[2]

        placement_x, placement_y, placement_z = placement

        center_x = placement_x + center_offset_x
        center_y = placement_y + center_offset_y
        center_z = placement_z + center_offset_z

        self.get_logger().info(
            f'Loaded object mesh from {mesh_path} with dimensions: '
            f'length={length:.3f}, thickness={thickness:.3f}, height={height:.3f}, '
            f'center=({center_x:.3f}, {center_y:.3f}, {center_z:.3f})'
        )

        return ObjectModel(
            x=center_x,
            y=center_y,
            z=center_z,
            length=length,
            height=height,
            thickness=thickness,
            orientation=0.0,
        )

    def _resolve_mesh_path(self, filename):
        """Convert a URDF package:// mesh reference to an absolute filesystem path."""
        if not filename.startswith('package://'):
            return Path(filename).expanduser().resolve()

        package_name, relative_path = filename[len('package://'):].split('/', 1)
        return Path(get_package_share_directory(package_name)) / relative_path

    def _build_object_path(self, obj):
        """Create a left-to-right scan pattern over the object surface."""
        quat = [-0.5, 0.5, 0.5, 0.5]
        step = 0.05
        num_steps = max(1, int(round(obj.height / step)))
        poses = []
        z_start = obj.z - obj.height / 2.0
        y_offset = obj.y - 0.1

        for i in range(num_steps):
            z = z_start + i * step
            if i % 2 == 0:
                poses.append(([obj.x - obj.length / 2.0, y_offset, z], quat))
                poses.append(([obj.x + obj.length / 2.0, y_offset, z], quat))
            else:
                poses.append(([obj.x + obj.length / 2.0, y_offset, z], quat))
                poses.append(([obj.x - obj.length / 2.0, y_offset, z], quat))

        return poses

    def move_to(self, pose):
        position, quat_xyzw = pose
        self.moveit2.move_to_pose(
            position=position,
            quat_xyzw=quat_xyzw,
            cartesian=True,
            cartesian_max_step=0.002,
            cartesian_fraction_threshold=0.3,
        )
        success = self.moveit2.wait_until_executed()
        if success:
            self.get_logger().info('Move succeeded')
        else:
            self.get_logger().warn('Move failed')


def main(args=None):
    rclpy.init(args=args)
    node = PaintObject()

    node.get_logger().info('Moving along the object scan pattern...')
    for pose in node.scan_path:
        node.move_to(pose)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()