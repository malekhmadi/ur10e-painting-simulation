import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur
from environment.wall import create_default_circle, create_default_wall


class MoveToPose(Node):
    def __init__(self):
        super().__init__('move_to_pose')

        callback_group = ReentrantCallbackGroup()

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur.joint_names(),
            base_link_name=ur.base_link_name(),
            end_effector_name=ur.end_effector_name(),
            group_name=ur.MOVE_GROUP_ARM,
            callback_group=callback_group,
        )

        # Use the shared default objects defined in the environment module.
        self.object_type = 'wall'

        wall = create_default_wall()
        circle = create_default_circle()
        self.scan_path = self._build_scan_path(wall, circle)

    def _build_scan_path(self, wall, circle=None):
        """Return the correct scan path depending on the object type."""
        if self.object_type == 'wall':
            return self._build_wall_path(wall)
        if self.object_type == 'circle':
            return self._build_circle_path(circle or create_default_circle())
        self.get_logger().warn(
            f'Unknown object_type "{self.object_type}". Defaulting to wall scan.'
        )
        return self._build_wall_path(wall)

    def _build_wall_path(self, wall):
        """Create a simple left-to-right scan pattern for the full wall."""
        quat = [-0.5, 0.5, 0.5, 0.5]
        step = 0.05
        num_steps = max(1, int(round(wall.height / step)))
        poses = []
        z1=wall.z-wall.height/2.0
        for i in range(num_steps):
            z = z1 + i * step
            if i % 2 == 0:
                poses.append(([wall.x - wall.length / 2.0, wall.y-0.1, z], quat))
                poses.append(([wall.x + wall.length / 2.0, wall.y-0.1, z], quat))
            else:
                poses.append(([wall.x + wall.length / 2.0, wall.y-0.1, z], quat))
                poses.append(([wall.x - wall.length / 2.0, wall.y-0.1, z], quat))

        return poses

    def _build_circle_path(self, circle):
        """Create a right-up-left-up scan pattern for the circular object."""
        quat = [-0.5, 0.5, 0.5, 0.5]
        x_left = circle.x - circle.radius
        x_right = circle.x + circle.radius
        z_start = circle.z - circle.height / 2.0
        z_end = circle.z + circle.height / 2.0
        z_step = 0.05
        poses = []

        z = z_start
        row = 0
        while z <= z_end + 1e-6:
            if row % 2 == 0:
                poses.append(([x_left, circle.y, z], quat))
                poses.append(([x_right, circle.y, z], quat))
            else:
                poses.append(([x_right, circle.y, z], quat))
                poses.append(([x_left, circle.y, z], quat))

            z += z_step
            row += 1

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
    node = MoveToPose()

    node.get_logger().info(f'Moving along the {node.object_type} scan pattern...')
    for pose in node.scan_path:
        node.move_to(pose)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()