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

        # Manually set start and end points here.
        # Change the x, y, z values to the points you want the robot to reach.
        self.start_point = (
            [-0.2, 0.7, 1.1],  # [x, y, z]
            [0.0, 0.0, 0.0, 1.0],  # quaternion [x, y, z, w]
        )
        self.end_point = (
            [0.4, 0.7, 1.1],  # [x, y, z]
            [0.0, 0.0, 0.0, 1.0],  # quaternion [x, y, z, w]
        )

    def move_to(self, pose):
        position, quat_xyzw = pose
        self.moveit2.move_to_pose(
            position=position,
            quat_xyzw=quat_xyzw,
            cartesian=True,
            cartesian_max_step=0.002,
            cartesian_fraction_threshold=0.5,
        )
        success = self.moveit2.wait_until_executed()
        if success:
            self.get_logger().info('Move succeeded')
        else:
            self.get_logger().warn('Move failed')


def main(args=None):
    rclpy.init(args=args)
    node = MoveToPose()

    node.get_logger().info('Moving to start point...')
    node.move_to(node.start_point)
    
    node.get_logger().info('Moving to end point...')
    node.move_to(node.end_point)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()