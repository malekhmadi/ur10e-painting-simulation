import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur
from environment.wall import Wall


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

        # Create wall object with dimensions
        wall = Wall(x=0.0, y=0.7, z=1.1, length=0.6, height=0.6, thickness=0.03, orientation=0)

        # Build a scan path that covers the full wall by moving left-right across the wall
        # at several height levels, changing only the x-axis and z-axis values.
        self.wall_path = self._build_wall_path(wall)

    def _build_wall_path(self, wall):
        """Create a simple left-to-right scan pattern for the full wall."""
        quat = [-0.5, 0.5, 0.5, 0.5]
        step = 0.05
        num_steps = max(1, int(round(wall.height / step)))
        poses = []

        for i in range(num_steps):
            z = wall.z + i * step
            if i % 2 == 0:
                poses.append(([wall.x - wall.length / 2.0, wall.y, z], quat))
                poses.append(([wall.x + wall.length / 2.0, wall.y, z], quat))
            else:
                poses.append(([wall.x + wall.length / 2.0, wall.y, z], quat))
                poses.append(([wall.x - wall.length / 2.0, wall.y, z], quat))

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

    node.get_logger().info('Moving along the full wall pattern...')
    for pose in node.wall_path:
        node.move_to(pose)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()