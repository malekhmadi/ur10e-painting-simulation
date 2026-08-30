"""cal_pattern_ap4.py — surface-segmented painting path generation.

Pipeline:
  STL mesh -> analyze geometry -> detect/segment surfaces -> determine
  painting direction per surface -> generate parallel passes -> connect
  passes smoothly -> convert to robot TCP poses.
"""

import re
import importlib.util
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath
from ament_index_python.packages import get_package_share_directory
import xml.etree.ElementTree as ET

ROW_SPACING = 0.03
ROBOT_POSITION = np.array([0.0, 0.0, -1.0])
ANGLE_RANGES = ((0.0, 180.0),)
LIFT_DISTANCE = 0.05
MIN_FACET_AREA=1e-6
U_RESOLUTION=0.005
PAINT_DISTANCE = 0.05  # fixed tool-to-surface offset in meters
# ---------------------------------------------------------------------------
# Step 0 — orientation / scale utilities (shared with ap3)
# ---------------------------------------------------------------------------

def euler_to_quaternion(roll, pitch, yaw):
    """Convert roll/pitch/yaw (radians) to quaternion (x, y, z, w)."""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w])


def quaternion_to_matrix4(quat_xyzw):
    """Build a 4x4 homogeneous rotation matrix from a quaternion (x, y, z, w)."""
    x, y, z, w = quat_xyzw
    rot3 = np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x**2 + y**2)],
    ])
    matrix4 = np.eye(4)
    matrix4[:3, :3] = rot3
    return matrix4


def normal_to_quaternion(normal, up_hint=np.array([0.0, 0.0, 1.0])):
    """Orient the tool's approach axis (-Z) opposite the given surface normal."""
    normal = normal / np.linalg.norm(normal)
    z_axis = -normal

    if abs(np.dot(z_axis, up_hint)) > 0.95:
        up_hint = np.array([1.0, 0.0, 0.0])

    x_axis = np.cross(up_hint, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    rot_matrix = np.column_stack((x_axis, y_axis, z_axis))

    trace = np.trace(rot_matrix)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot_matrix[2, 1] - rot_matrix[1, 2]) * s
        y = (rot_matrix[0, 2] - rot_matrix[2, 0]) * s
        z = (rot_matrix[1, 0] - rot_matrix[0, 1]) * s
    else:
        i = np.argmax(np.diag(rot_matrix))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2])
            w = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
            z = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2])
            w = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
            x = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1])
            w = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
            x = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
            y = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
            z = 0.25 * s

    return [x, y, z, w]


def read_mesh_scale_from_urdf(package_name='environment', urdf_filename='obj.urdf'):
    """Read the <mesh scale="..."/> value directly from the object's URDF."""
    urdf_path = Path(get_package_share_directory(package_name)) / 'urdf' / urdf_filename
    if not urdf_path.exists():
        raise FileNotFoundError(f'URDF not found: {urdf_path}')

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    mesh_elem = next(root.iter('mesh'), None)
    if mesh_elem is None:
        raise ValueError(f'No <mesh> tag found in URDF: {urdf_path}')

    scale_attr = mesh_elem.get('scale', '1 1 1')
    scale = [float(v) for v in scale_attr.split()]
    if len(scale) != 3:
        raise ValueError(f'Malformed scale attribute: {scale_attr}')
    return scale


def read_object_pose_from_launch(package_name='environment', launch_filename='object_tf.launch.py'):
    """Dynamically import the object's launch file and read its
    DeclareLaunchArgument default values for x/y/z/roll/pitch/yaw."""
    launch_path = Path(get_package_share_directory(package_name)) / 'launch' / launch_filename
    if not launch_path.exists():
        raise FileNotFoundError(f'Launch file not found: {launch_path}')

    spec = importlib.util.spec_from_file_location('object_tf_launch_module', launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    launch_description = module.generate_launch_description()

    defaults = {}
    for entity in launch_description.entities:
        if hasattr(entity, 'name') and hasattr(entity, 'default_value'):
            try:
                value_text = ''.join(
                    s.text if hasattr(s, 'text') else str(s)
                    for s in entity.default_value
                )
                defaults[entity.name] = value_text
            except Exception:
                continue

    x = float(defaults.get('x', 0.0))
    y = float(defaults.get('y', 0.0))
    z = float(defaults.get('z', 0.0))
    roll = float(defaults.get('roll', 0.0))
    pitch = float(defaults.get('pitch', 0.0))
    yaw = float(defaults.get('yaw', 0.0))

    return np.array([x, y, z]), (roll, pitch, yaw)


# ---------------------------------------------------------------------------
# Step 1 — segment the object into meaningful surfaces
# ---------------------------------------------------------------------------

def segment_surfaces(mesh, robot_position=None, angle_ranges=ANGLE_RANGES, min_facet_area=MIN_FACET_AREA):
    """
    Detect planar facets (trimesh's built-in coplanar + connected clustering)
    and keep only the ones that are meaningful to paint: large enough, and
    (if robot_position given) within the accepted facing-angle range(s).

    Returns a list of (face_indices, normal, distance_to_robot_or_None),
    ordered nearest-to-robot first if robot_position was given.
    """
    surfaces = []
    for face_indices, normal in zip(mesh.facets, mesh.facets_normal):
        area = mesh.area_faces[face_indices].sum()
        if area < min_facet_area:
            continue
        
        if abs(normal[0]) > 0.8:  
            continue
        
        distance = None
        if robot_position is not None:
            centroid = mesh.vertices[mesh.faces[face_indices]].reshape(-1, 3).mean(axis=0)
            to_robot = robot_position - centroid
            distance = np.linalg.norm(to_robot)
            to_robot_dir = to_robot / distance if distance > 1e-9 else to_robot

            cos_alignment = np.clip(np.dot(normal, to_robot_dir), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_alignment))

            if not any(lo <= angle_deg <= hi for lo, hi in angle_ranges):
                continue

        surfaces.append((face_indices, normal, distance))

    if robot_position is not None:
        surfaces.sort(key=lambda item: item[2])

    return surfaces


# ---------------------------------------------------------------------------
# Step 2 — determine the painting direction per surface
# ---------------------------------------------------------------------------

def get_tangent_frame(normal, u_hint=None):
    """
    Build two orthogonal tangent vectors (U, V) lying on the plane
    perpendicular to the given normal. If u_hint is given (a preferred
    in-plane direction, e.g. the face's longest dimension), U is aligned
    to it; otherwise falls back to an arbitrary stable reference axis.
    """
    normal = normal / np.linalg.norm(normal)

    if u_hint is not None and np.linalg.norm(u_hint) > 1e-9:
        u = u_hint - np.dot(u_hint, normal) * normal  # project out any normal component
        u = u / np.linalg.norm(u)
    else:
        reference = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(normal, reference)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        u = np.cross(normal, reference)
        u = u / np.linalg.norm(u)

    v = np.cross(normal, u)
    return u, v


def determine_painting_direction(mesh, face_indices, normal):
    """
    Pick the painting stroke direction for a face by finding its longest
    in-plane dimension (via PCA on the projected 2D vertex positions).
    Painting along the long axis minimizes the number of turnarounds.

    Returns:
        u_axis, v_axis: tangent frame where V is the LONG axis (stroke
                        direction) and U is the perpendicular (row-stepping) axis
        origin: reference point used for the 2D projection
    """
    submesh = mesh.submesh([face_indices], append=True)
    vertices = submesh.vertices
    origin = vertices.mean(axis=0)

    # arbitrary starting frame just to project into 2D for PCA
    u0, v0 = get_tangent_frame(normal)
    relative = vertices - origin
    coords_2d = np.column_stack([relative @ u0, relative @ v0])

    # PCA: eigenvector with largest variance = longest dimension
    cov = np.cov(coords_2d.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_axis_2d = eigvecs[:, np.argmax(eigvals)]  # direction of greatest spread, in (u0,v0) coords

    # convert that 2D direction back into a 3D in-plane vector
    long_axis_3d = long_axis_2d[0] * u0 + long_axis_2d[1] * v0

    v_axis, u_axis = get_tangent_frame(normal, u_hint=long_axis_3d)
    # note: stroke direction (long axis) becomes V; row-stepping becomes U
    return u_axis, v_axis, origin


# ---------------------------------------------------------------------------
# Step 3 — generate parallel painting passes per surface
# ---------------------------------------------------------------------------

def generate_passes_for_surface(mesh, face_indices, normal, row_spacing, u_resolution=U_RESOLUTION, paint_distance=PAINT_DISTANCE):
    """
    Generate an edge-to-edge zigzag of parallel passes across one surface,
    swept along that surface's determined long-axis painting direction.
    """
    u_axis, v_axis, origin = determine_painting_direction(mesh, face_indices, normal)

    submesh = mesh.submesh([face_indices], append=True)
    vertices = submesh.vertices
    relative = vertices - origin
    u = relative @ u_axis
    v = relative @ v_axis
    points_2d = np.column_stack([u, v])

    if len(points_2d) < 3:
        return np.empty((0, 3)), np.empty((0, 3))

    try:
        hull = ConvexHull(points_2d)
    except Exception:
        return np.empty((0, 3)), np.empty((0, 3))

    hull_points = points_2d[hull.vertices]
    polygon = MplPath(hull_points)

    v_min, v_max = points_2d[:, 1].min(), points_2d[:, 1].max()
    u_min, u_max = points_2d[:, 0].min(), points_2d[:, 0].max()

    num_rows = max(1, int(round((v_max - v_min) / row_spacing)))
    u_samples = np.arange(u_min, u_max + u_resolution, u_resolution)

    ordered_points = []
    for i in range(num_rows + 1):
        v_level = v_min + i * row_spacing
        test_points = np.column_stack([u_samples, np.full_like(u_samples, v_level)])
        inside = polygon.contains_points(test_points)

        if not inside.any():
            continue

        u_inside = u_samples[inside]
        u_start, u_end = u_inside.min(), u_inside.max()

        if i % 2 == 1:
            u_start, u_end = u_end, u_start

        for uu in (u_start, u_end):
            point_3d = origin + uu * u_axis + v_level * v_axis
            ordered_points.append(point_3d)

    if not ordered_points:
        return np.empty((0, 3)), np.empty((0, 3))

    ordered_points = np.asarray(ordered_points, dtype=float)
    ordered_points = ordered_points + normal * paint_distance
    normals = np.tile(normal, (len(ordered_points), 1))
    return ordered_points, normals


# ---------------------------------------------------------------------------
# Step 4 — connect passes smoothly (across surfaces)
# ---------------------------------------------------------------------------

def connect_surfaces_smoothly(surface_paths, lift_distance, transition_steps=3):
    """
    Concatenate per-surface paths, inserting a short lift-off/retract/approach
    transition between surfaces instead of a raw jump, so the tool doesn't
    drag across empty space between unrelated regions.

    Args:
        surface_paths: list of (points, normals) per surface, already ordered
        lift_distance: how far to retract along the normal before/after
                       traveling between surfaces
        transition_steps: number of interpolated points in each transition

    Returns:
        all_points, all_normals — one continuous trajectory
    """
    if not surface_paths:
        raise RuntimeError('No surface paths to connect.')

    all_points = [surface_paths[0][0]]
    all_normals = [surface_paths[0][1]]

    for i in range(1, len(surface_paths)):
        prev_points, prev_normals = surface_paths[i - 1]
        curr_points, curr_normals = surface_paths[i]

        last_point = prev_points[-1]
        last_normal = prev_normals[-1]
        first_point = curr_points[0]
        first_normal = curr_normals[0]

        retract_point = last_point + last_normal * lift_distance
        approach_point = first_point + first_normal * lift_distance

        transition_points = np.linspace(retract_point, approach_point, transition_steps)
        transition_normals = np.tile(last_normal, (transition_steps, 1))  # orientation blends during transit

        all_points.append(np.vstack([[retract_point], transition_points, [approach_point]]))
        all_normals.append(np.vstack([[last_normal], transition_normals, [first_normal]]))

        all_points.append(curr_points)
        all_normals.append(curr_normals)

    return np.vstack(all_points), np.vstack(all_normals)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def generate_surface_based_path(mesh, row_spacing, min_facet_area=MIN_FACET_AREA,
                                 robot_position=None, angle_ranges=ANGLE_RANGES,
                                 lift_distance=LIFT_DISTANCE, paint_distance=PAINT_DISTANCE):
    """Run the full 5-step pipeline and return one continuous trajectory."""
    surfaces = segment_surfaces(mesh, robot_position, angle_ranges, min_facet_area)
    if not surfaces:
        raise RuntimeError('No surfaces passed segmentation/filtering — check angle_ranges or min_facet_area.')

    surface_paths = []
    for face_indices, normal, _ in surfaces:
        points, normals = generate_passes_for_surface(mesh, face_indices, normal, row_spacing=row_spacing , paint_distance=paint_distance)
        if len(points) == 0:
            continue
        surface_paths.append((points, normals))

    if not surface_paths:
        raise RuntimeError('No surfaces produced a usable path.')

    return connect_surfaces_smoothly(surface_paths, lift_distance=lift_distance)


def check_pose_reachable(moveit2, position, quat_xyzw):
    """
    Two-stage check: does an IK solution exist, and can a collision-free
    plan actually be found to reach it. No robot motion happens (plan only).
    """
    result = {'ik_found': False, 'plan_found': False}

    joint_state = moveit2.compute_ik(position=position.tolist(), quat_xyzw=quat_xyzw)
    if joint_state is None:
        return result
    result['ik_found'] = True

    trajectory = moveit2.plan(position=position.tolist(), quat_xyzw=quat_xyzw)
    result['plan_found'] = trajectory is not None

    return result


# ---------------------------------------------------------------------------
# Visualization (same style as ap3, with equal-scale axes fix applied)
# ---------------------------------------------------------------------------

def visualize_path(mesh, points, normals, output_path='/tmp/surface_path.png', normal_length=0.03):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh_collection = Poly3DCollection(mesh.triangles, alpha=0.15, linewidths=0.2)
    ax.add_collection3d(mesh_collection)

    ax.plot(points[:, 0], points[:, 1], points[:, 2], linewidth=1.5, label="trajectory")

    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=np.arange(len(points)), cmap='viridis', s=15
    )
    fig.colorbar(scatter, ax=ax, label="path order")

    step = max(1, len(points) // 60)
    ax.quiver(
        points[::step, 0], points[::step, 1], points[::step, 2],
        normals[::step, 0], normals[::step, 1], normals[::step, 2],
        length=normal_length, color='red', normalize=True
    )

    ax.scatter(*points[0], s=100, marker='o', color='green', label="START")
    ax.scatter(*points[-1], s=100, marker='X', color='black', label="END")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Surface-segmented painting path\nColors = order, red = normals, green line = lift transitions")
    ax.legend()

    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    max_range = (bounds[1] - bounds[0]).max() / 2
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")
"""

def visualize_paint_coverage(points, normals, output_path='/tmp/surface_paint_coverage.png',
                              spray_radius=0.015, normal_length=0.02, max_disks=3000):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if len(points) > max_disks:
        disk_idx = np.linspace(0, len(points) - 1, max_disks).astype(int)
    else:
        disk_idx = np.arange(len(points))

    theta = np.linspace(0, 2 * np.pi, 20)
    disks = []
    for i in disk_idx:
        p = points[i]
        n = normals[i] / np.linalg.norm(normals[i])
        u, v = get_tangent_frame(n)
        circle = p + spray_radius * (np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v))
        disks.append(circle)

    collection = Poly3DCollection(disks, alpha=0.35)
    ax.add_collection3d(collection)

    ax.plot(points[:, 0], points[:, 1], points[:, 2], linewidth=0.5)

    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=np.arange(len(points)), cmap='viridis', s=6
    )
    fig.colorbar(scatter, ax=ax, label="path order")

    ax.quiver(
        points[:, 0], points[:, 1], points[:, 2],
        normals[:, 0], normals[:, 1], normals[:, 2],
        length=normal_length, color='red', normalize=True
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Paint coverage simulation")

    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    center = (bounds_min + bounds_max) / 2
    max_range = (bounds_max - bounds_min).max() / 2
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved paint coverage visualization: {output_path}")
"""
def build_ribbon_quads(points, normals, width):
    """
    Build a series of quads forming a continuous ribbon along the path,
    tangent to the surface (using each segment's normal) and perpendicular
    to the direction of travel, with the given width (e.g. = row_spacing).

    Returns:
        quads: list of (4, 3) arrays, one per segment, each a flat quad polygon
        colors: list of (r, g, b, a) tuples, one per quad, gradient by order
    """
    quads = []
    n = len(points)

    for i in range(n - 1):
        p0 = points[i]
        p1 = points[i + 1]
        normal = normals[i]
        normal = normal / np.linalg.norm(normal)

        direction = p1 - p0
        seg_length = np.linalg.norm(direction)
        if seg_length < 1e-9:
            continue
        direction = direction / seg_length

        # perpendicular to both travel direction and surface normal —
        # this is the "across the stroke" axis, in the tangent plane
        side = np.cross(direction, normal)
        side_norm = np.linalg.norm(side)
        if side_norm < 1e-9:
            continue
        side = side / side_norm

        half_width = width / 2.0
        offset = side * half_width

        quad = np.array([
            p0 - offset,
            p0 + offset,
            p1 + offset,
            p1 - offset,
        ])
        quads.append(quad)

    colors = []
    for i in range(len(quads)):
        t = i / max(len(quads) - 1, 1)
        colors.append((t, t, 1.0 - t, 0.6))  # blue (start) -> yellow (end)

    return quads, colors


def visualize_paint_ribbon(mesh, points, normals, row_spacing, output_path='/tmp/paint_ribbon.png'):
    """Render the painting path as a continuous colored ribbon (width =
    row_spacing), tangent to the surface, instead of a thin line or disks."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh_collection = Poly3DCollection(mesh.triangles, alpha=0.12, linewidths=0.1, facecolor='gray')
    ax.add_collection3d(mesh_collection)

    quads, colors = build_ribbon_quads(points, normals, row_spacing)
    ribbon = Poly3DCollection(quads, facecolors=colors, edgecolors='none')
    ax.add_collection3d(ribbon)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Paint coverage ribbon — width = row_spacing ({row_spacing} m)")

    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    max_range = (bounds[1] - bounds[0]).max() / 2
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")



def visualize_reachability(mesh, reachable_points, unreachable_points, output_path='/tmp/reachability.png'):
    """Show the object with reachable points as a green path and
    unreachable points marked as red X's."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh_collection = Poly3DCollection(mesh.triangles, alpha=0.12, linewidths=0.1, facecolor='gray')
    ax.add_collection3d(mesh_collection)

    if len(reachable_points) > 0:
        ax.plot(reachable_points[:, 0], reachable_points[:, 1], reachable_points[:, 2],
                color='green', linewidth=1.5, label=f'reachable ({len(reachable_points)})')
        ax.scatter(reachable_points[:, 0], reachable_points[:, 1], reachable_points[:, 2],
                   color='green', s=10)

    if len(unreachable_points) > 0:
        ax.scatter(unreachable_points[:, 0], unreachable_points[:, 1], unreachable_points[:, 2],
                   color='red', marker='x', s=60, label=f'unreachable ({len(unreachable_points)})')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Reachability check — green = reachable path, red X = unreachable/blocked")
    ax.legend()

    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    max_range = (bounds[1] - bounds[0]).max() / 2
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved: {output_path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mesh_path = Path(get_package_share_directory('environment')) / 'meshes' / 'obj.stl'
    print(f"Loading mesh: {mesh_path}")

    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    scale = read_mesh_scale_from_urdf()
    mesh.apply_scale(scale)

    translation, (roll, pitch, yaw) = read_object_pose_from_launch()
    quat = euler_to_quaternion(roll, pitch, yaw)
    mesh.apply_transform(quaternion_to_matrix4(quat))
    print(f"Applied rotation from launch file: roll={roll}, pitch={pitch}, yaw={yaw}")

    print(f"Mesh has {len(mesh.facets)} planar facets")
    print(f"Mesh bounds (meters): {mesh.bounds}")

    points, normals = generate_surface_based_path(
        mesh,
        row_spacing=ROW_SPACING,
        robot_position=ROBOT_POSITION,
        angle_ranges=ANGLE_RANGES,
        lift_distance=LIFT_DISTANCE,
        paint_distance=PAINT_DISTANCE
    )

    print(f"Generated {len(points)} total path points across all surfaces")

    visualize_path(mesh, points, normals, '/tmp/surface_path.png')
    visualize_paint_ribbon(mesh, points, normals, ROW_SPACING, output_path='/tmp/surface_paint_coverage.png')


if __name__ == '__main__':
    main()