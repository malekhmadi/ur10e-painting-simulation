# GitHub Copilot Instructions

## Project Overview

This repository contains a robotics project built with ROS 2 Humble on Ubuntu 22.04.

The project focuses on autonomous robotic manipulation using a Universal Robots arm with MoveIt 2.

Current objective:
- Simulate the robot in RViz
- Add environment objects as CollisionObjects
- Generate painting trajectories
- Execute trajectories safely with MoveIt
- Later deploy on a real robot

---

## Software Stack

- ROS 2 Humble
- Python (rclpy)
- MoveIt 2
- Universal Robots ROS 2 Driver
- RViz2
- tf2
- URDF/Xacro
- Colcon
- Git

---

## Workspace Structure

The workspace follows the standard ROS 2 layout.

Example:

ros2_ws/
├── src/
│   ├── robot_brain/
│   ├── robot_environment/
│   ├── Universal_Robots_ROS2_Driver/
│   └── ...
└── install/

---

## Coding Style

Always:

- Prefer Python unless explicitly asked for C++.
- Use rclpy.
- Follow ROS 2 Humble APIs.
- Write readable code.
- Add useful comments only when necessary.
- Use descriptive variable names.

Avoid:

- ROS1 syntax.
- Deprecated APIs.
- Hardcoded paths.
- Global variables unless necessary.

---

## ROS 2 Conventions

Prefer:

- Launch files instead of running many nodes manually.
- Parameters instead of hardcoded constants.
- QoS profiles appropriate for the topic.
- tf2 for coordinate transforms.
- Timers instead of while loops whenever possible.

Explain publishers, subscribers, services, actions, and callbacks clearly when requested.

---

## MoveIt 2

Whenever motion planning is required:

- Use MoveIt instead of manual inverse kinematics.
- Respect collision checking.
- Keep planning and execution separate.
- Prefer planning to poses or Cartesian paths using MoveIt APIs.

---

## Environment

Environment objects should be represented using CollisionObjects.

Collision objects should:

- Use the "world" frame unless another frame is specified.
- Have meaningful IDs.
- Be added through PlanningSceneInterface or appropriate ROS topics.

---

## Universal Robots

Robot platform:

- Universal Robots (UR5e/UR10e)

Assume:

- ROS 2 Humble
- Universal Robots ROS 2 Driver
- MoveIt configuration package

---

## Responses

When generating code:

- Explain why the code works.
- Mention important ROS concepts involved.
- Point out common mistakes.
- Suggest debugging commands if appropriate.

---

## Debugging

When something fails:

Prefer checking:

- ros2 node list
- ros2 topic list
- ros2 topic echo
- ros2 param list
- ros2 service list
- ros2 action list
- ros2 doctor

before suggesting major changes.

---

## Build System

Always assume:

colcon build

and source:

source install/setup.bash

after rebuilding.

---

## If information is missing

Do not invent package names or topic names.

Instead:

- Ask for the existing package structure.
- Ask for the launch file.
- Ask for the node implementation if needed.

Accuracy is preferred over guessing.