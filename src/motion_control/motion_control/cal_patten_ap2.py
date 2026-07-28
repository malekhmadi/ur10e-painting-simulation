"""fcal_pattern_ap1.py — segment a mesh into planar faces, and build a per-face
raster painting path for each one, then concatenate into one full trajectory."""

from pathlib import Path

import numpy as np
import trimesh
from ament_index_python.packages import get_package_share_directory

import xml.etree.ElementTree as ET

def quaternion_to_matrix4(quat_xyzw):
    """Build a 4x4 homogeneous rotation matrix from a quaternion (x, y, z, w),
    suitable for trimesh.apply_transform()."""
    x, y, z, w = quat_xyzw
    rot3 = np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x**2 + y**2)],
    ])
    matrix4 = np.eye(4)
    matrix4[:3, :3] = rot3
    return matrix4

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


def rotate_vectors_by_quaternion(vectors, quat_xyzw):
    """Rotate an (N, 3) array of vectors by a quaternion (x, y, z, w)."""
    x, y, z, w = quat_xyzw
    rot_matrix = np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x**2 + y**2)],
    ])
    return vectors @ rot_matrix.T

def read_mesh_scale_from_urdf(package_name='environment', urdf_filename='obj.urdf'):
    """Read the <mesh scale="..."/> value directly from the object's URDF,
    so painting-pattern code never has to hardcode it separately."""
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
        raise ValueError(f'Malformed scale attribute in URDF: {scale_attr}')

    return scale  # [sx, sy, sz]

def get_tangent_frame(normal):
    """Build two orthogonal tangent vectors (U, V) lying on the plane
    perpendicular to the given normal."""
    normal = normal / np.linalg.norm(normal)
    # pick an arbitrary reference vector not parallel to normal
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, reference)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def sample_face_points(mesh, face_indices, points_per_unit_area=200, min_points=6):
    """Sample points restricted to a single planar face (a group of coplanar,
    connected triangles), scaling point count by the face's surface area."""
    submesh = mesh.submesh([face_indices], append=True)
    area = submesh.area
    num_points = max(min_points, int(area * points_per_unit_area))

    points, sub_face_indices = trimesh.sample.sample_surface_even(
        submesh, count=num_points
    )
    if len(points) == 0:
        # even sampling can occasionally return nothing for very small/thin faces
        points, sub_face_indices = trimesh.sample.sample_surface(submesh, count=min_points)

    normals = submesh.face_normals[sub_face_indices]
    return points, normals


def order_face_path(points, normal, band_count=6):
    """Project a single face's points into its local 2D (u, v) frame,
    raster-order them (bands along V, sorted by U, alternating direction),
    then return the points in that order (still in original 3D coordinates)."""
    u_axis, v_axis = get_tangent_frame(normal)
    origin = points.mean(axis=0)

    relative = points - origin
    u = relative @ u_axis
    v = relative @ v_axis

    v_min, v_max = v.min(), v.max()
    v_range = max(v_max - v_min, 1e-9)  # avoid divide-by-zero on degenerate faces
    band_indices = np.clip(
        ((v - v_min) / v_range * band_count).astype(int), 0, band_count - 1
    )

    ordered_points = []
    for band in range(band_count):
        mask = band_indices == band
        if not np.any(mask):
            continue
        band_points = points[mask]
        band_u = u[mask]

        sort_idx = np.argsort(band_u)
        band_points = band_points[sort_idx]

        if band % 2 == 1:
            band_points = band_points[::-1]

        ordered_points.append(band_points)

    if not ordered_points:
        return points  # fallback: degenerate face, return unordered

    return np.vstack(ordered_points)


def generate_face_based_path(mesh, points_per_unit_area=200, band_count=6, min_facet_area=1e-6):
    """
    Full pipeline: segment mesh into planar faces, sample + order each face's
    points independently, then concatenate into one continuous trajectory.

    Returns:
        all_points: (N, 3) ordered path positions
        all_normals: (N, 3) corresponding surface normals
    """
    all_points = []
    all_normals = []

    for facet_face_indices, facet_normal in zip(mesh.facets, mesh.facets_normal):
        facet_area = mesh.area_faces[facet_face_indices].sum()
        if facet_area < min_facet_area:
            continue  # skip tiny/noise facets

        points, normals = sample_face_points(
            mesh, facet_face_indices, points_per_unit_area=points_per_unit_area
        )
        if len(points) == 0:
            continue

        ordered_points = order_face_path(points, facet_normal, band_count=band_count)
        ordered_normals = np.tile(facet_normal, (len(ordered_points), 1))

        all_points.append(ordered_points)
        all_normals.append(ordered_normals)

    if not all_points:
        raise RuntimeError('No valid facets found — mesh may need repair or facet_threshold tuning.')

    return np.vstack(all_points), np.vstack(all_normals)


def visualize_path(points, normals, output_path='/tmp/face_path_visualization.png', normal_length=0.02):
    """Plot the ordered path and surface normals in 3D, saved to a PNG file."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(points[:, 0], points[:, 1], points[:, 2], color='gray', linewidth=0.5, alpha=0.5)
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=np.arange(len(points)), cmap='viridis', s=8
    )
    fig.colorbar(scatter, ax=ax, label='path order (0 = start)')
    '''
    ax.quiver(
        points[:, 0], points[:, 1], points[:, 2],
        #normals[:, 0], normals[:, 1], normals[:, 2],
        length=normal_length, color='red', linewidth=0.5, normalize=True
    )
    '''
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Face-based painting path — points ordered by color, normals in red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f'Saved visualization to {output_path}')

def visualize_paint_coverage(points, normals, output_path='/tmp/face_paint_visualization.png',
                              spray_radius=0.015, normal_length=0.02, max_disks=3000):
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- simulated paint coverage disks (drawn first, underneath everything else) ---
    if len(points) > max_disks:
        disk_idx = np.linspace(0, len(points) - 1, max_disks).astype(int)
    else:
        disk_idx = np.arange(len(points))

    circle_steps = 10
    theta = np.linspace(0, 2 * np.pi, circle_steps)
    disks = []
    for i in disk_idx:
        p = points[i]
        n = normals[i] / np.linalg.norm(normals[i])
        u, v = get_tangent_frame(n)
        circle = p + spray_radius * (np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v))
        disks.append(circle)

    paint_collection = Poly3DCollection(disks, facecolor='dodgerblue', edgecolor='none', alpha=0.35)
    ax.add_collection3d(paint_collection)

    # --- path line + order coloring (same as before) ---
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color='gray', linewidth=0.5, alpha=0.6)
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=np.arange(len(points)), cmap='viridis', s=6
    )
    fig.colorbar(scatter, ax=ax, label='path order (0 = start)')

    # --- normals ---
    ax.quiver(
        points[:, 0], points[:, 1], points[:, 2],
        normals[:, 0], normals[:, 1], normals[:, 2],
        length=normal_length, color='red', linewidth=0.5, normalize=True
    )

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Face-based painting path — blue = simulated paint coverage, red = normals')

    # auto-scale axes equally so the paint disks aren't visually distorted
    all_bounds = points
    max_range = (all_bounds.max(axis=0) - all_bounds.min(axis=0)).max() / 2.0
    mid = all_bounds.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f'Saved paint coverage visualization to {output_path}')

def main():
    scale = read_mesh_scale_from_urdf()  # or hardcode, whichever version you currently have

    # orientation to test — edit these to try different rotations
    roll, pitch, yaw = 1.57, 0.0, 0.0
    translation = np.array([0.0, 0.0, 0.0])
    
    mesh_path = Path(get_package_share_directory('environment')) / 'meshes' / 'obj.stl'
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    mesh.apply_scale(scale)
    quat = euler_to_quaternion(roll, pitch, yaw)
    mesh.apply_transform(quaternion_to_matrix4(quat))

    print(f'Mesh has {len(mesh.facets)} planar facets')
    print(f'Mesh bounds (meters): {mesh.bounds}')

    points, normals = generate_face_based_path(mesh, points_per_unit_area=120, band_count=6)
    # translation only — no more rotating points/normals after the fact
    points = points + translation
    # apply orientation before visualizing, so the PNG reflects the rotated object
    print(f'Generated {len(points)} total path points across all faces (rotated: roll={roll}, pitch={pitch}, yaw={yaw})')

    visualize_path(points, normals, output_path='/tmp/face_path_visualization.png')
    visualize_paint_coverage(points, normals, output_path='/tmp/face_paint_visualization.png',
                              spray_radius=0.01)


if __name__ == '__main__':
    main()
