"""cal_pattern_ap3.py — deterministic edge-to-edge zigzag path generation for a
single planar face, based on its actual boundary shape (not random sampling)."""

import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath
import os
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
import trimesh


def get_tangent_frame(normal):
    """Build two orthogonal tangent vectors (U, V) lying on the plane
    perpendicular to the given normal."""
    normal = normal / np.linalg.norm(normal)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, reference)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def build_face_polygon(mesh, face_indices, normal):
    """Project a face's vertices into its local 2D (u, v) frame and compute
    the convex hull boundary — the face's actual outline shape."""
    submesh = mesh.submesh([face_indices], append=True)
    vertices = submesh.vertices

    origin = vertices.mean(axis=0)
    u_axis, v_axis = get_tangent_frame(normal)

    relative = vertices - origin
    u = relative @ u_axis
    v = relative @ v_axis
    points_2d = np.column_stack([u, v])

    hull = ConvexHull(points_2d)
    hull_points = points_2d[hull.vertices]

    return origin, u_axis, v_axis, hull_points, points_2d


def generate_zigzag_face_path(mesh, face_indices, normal, row_spacing=0.02, u_resolution=0.005):
    """
    Generate an edge-to-edge zigzag path across a single face: for each
    horizontal row, find exactly where that row enters/exits the face's
    real boundary, take those two points as the row's start/end, and
    alternate sweep direction per row.

    Args:
        row_spacing: vertical distance between rows (real-world units, e.g. meters)
        u_resolution: sampling step used to find each row's polygon boundary
                      (finer = more accurate edge detection, slower)

    Returns:
        points_3d: (M, 3) ordered path points
        normals: (M, 3) matching normals (all equal to `normal`, since it's one flat face)
    """
    origin, u_axis, v_axis, hull_points, points_2d = build_face_polygon(mesh, face_indices, normal)
    polygon = MplPath(hull_points)

    v_min, v_max = points_2d[:, 1].min(), points_2d[:, 1].max()
    u_min, u_max = points_2d[:, 0].min(), points_2d[:, 0].max()

    num_rows = max(1, int(round((v_max - v_min) / row_spacing)))
    u_samples = np.arange(u_min, u_max + u_resolution, u_resolution)

    ordered_points = []

    for i in range(num_rows + 1):
        v = v_min + i * row_spacing
        test_points = np.column_stack([u_samples, np.full_like(u_samples, v)])
        inside = polygon.contains_points(test_points)

        if not inside.any():
            continue  # this row misses the face entirely (can happen near rounded corners)

        u_inside = u_samples[inside]
        u_start, u_end = u_inside.min(), u_inside.max()  # face's actual start/end at this row

        if i % 2 == 1:
            u_start, u_end = u_end, u_start  # alternate direction each row

        for uu in (u_start, u_end):
            point_3d = origin + uu * u_axis + v * v_axis
            ordered_points.append(point_3d)

    ordered_points = np.array(ordered_points)
    normals = np.tile(normal, (len(ordered_points), 1))
    return ordered_points, normals


def generate_all_faces_zigzag(mesh, row_spacing=0.02, min_facet_area=1e-6):
    """Run the zigzag path generator across every detected planar facet and
    concatenate the results into one full trajectory."""
    all_points = []
    all_normals = []

    for face_indices, normal in zip(mesh.facets, mesh.facets_normal):
        area = mesh.area_faces[face_indices].sum()
        if area < min_facet_area:
            continue

        points, normals = generate_zigzag_face_path(mesh, face_indices, normal, row_spacing=row_spacing)
        if len(points) == 0:
            continue

        all_points.append(points)
        all_normals.append(normals)

    if not all_points:
        raise RuntimeError('No valid facets produced a path — check min_facet_area or row_spacing.')

    return np.vstack(all_points), np.vstack(all_normals)

def visualize_path(mesh, points, normals,
                   output_path='/tmp/zigzag_path.png',
                   normal_length=0.05):

    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection


    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')


    # -----------------------------
    # Draw mesh surface
    # -----------------------------
    faces = mesh.triangles

    mesh_collection = Poly3DCollection(
        faces,
        alpha=0.15,
        linewidths=0.2
    )

    ax.add_collection3d(mesh_collection)


    # -----------------------------
    # Draw path
    # -----------------------------
    ax.plot(
        points[:,0],
        points[:,1],
        points[:,2],
        linewidth=2,
        label="zigzag trajectory"
    )


    # Color points by order
    scatter = ax.scatter(
        points[:,0],
        points[:,1],
        points[:,2],
        c=np.arange(len(points)),
        cmap='viridis',
        s=20
    )

    fig.colorbar(
        scatter,
        ax=ax,
        label="path order"
    )


    # -----------------------------
    # Draw direction arrows
    # -----------------------------
    step = max(1, len(points)//50)

    ax.quiver(
        points[::step,0],
        points[::step,1],
        points[::step,2],

        np.diff(points[:,0], append=points[-1,0])[::step],
        np.diff(points[:,1], append=points[-1,1])[::step],
        np.diff(points[:,2], append=points[-1,2])[::step],

        length=0.03,
        normalize=True
    )


    # -----------------------------
    # Draw normals
    # -----------------------------
    ax.quiver(
        points[::step,0],
        points[::step,1],
        points[::step,2],

        normals[::step,0],
        normals[::step,1],
        normals[::step,2],

        length=normal_length,
        color='red',
        normalize=True
    )


    # -----------------------------
    # Mark start and end
    # -----------------------------
    ax.scatter(
        points[0,0],
        points[0,1],
        points[0,2],
        s=100,
        marker='o',
        label="START"
    )


    ax.scatter(
        points[-1,0],
        points[-1,1],
        points[-1,2],
        s=100,
        marker='X',
        label="END"
    )


    # -----------------------------
    # Labels
    # -----------------------------
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.set_title(
        "Edge-to-edge Zigzag Painting Path\n"
        "Black/Gray = mesh | Colors = order | Red = surface normals"
    )

    ax.legend()


    # Equal scale
    bounds = mesh.bounds

    center = bounds.mean(axis=0)

    max_range = (
        bounds[1]-bounds[0]
    ).max()/2


    ax.set_xlim(
        center[0]-max_range,
        center[0]+max_range
    )

    ax.set_ylim(
        center[1]-max_range,
        center[1]+max_range
    )

    ax.set_zlim(
        center[2]-max_range,
        center[2]+max_range
    )


    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    print(f"Saved: {output_path}")


def visualize_paint_coverage(points, normals,
                             output_path='/tmp/paint_coverage.png',
                             spray_radius=0.015,
                             normal_length=0.02,
                             max_disks=3000):

    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection


    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')


    if len(points) > max_disks:
        disk_idx = np.linspace(
            0,
            len(points)-1,
            max_disks
        ).astype(int)
    else:
        disk_idx = np.arange(len(points))


    theta = np.linspace(0,2*np.pi,20)

    disks=[]

    for i in disk_idx:

        p = points[i]

        n = normals[i]/np.linalg.norm(normals[i])

        u,v = get_tangent_frame(n)

        circle = (
            p +
            spray_radius *
            (
                np.outer(np.cos(theta),u)
                +
                np.outer(np.sin(theta),v)
            )
        )

        disks.append(circle)


    collection = Poly3DCollection(
        disks,
        alpha=0.35
    )

    ax.add_collection3d(collection)


    ax.plot(
        points[:,0],
        points[:,1],
        points[:,2],
        linewidth=0.5
    )


    scatter=ax.scatter(
        points[:,0],
        points[:,1],
        points[:,2],
        c=np.arange(len(points)),
        cmap='viridis',
        s=6
    )


    fig.colorbar(
        scatter,
        ax=ax,
        label="path order"
    )


    ax.quiver(
        points[:,0],
        points[:,1],
        points[:,2],
        normals[:,0],
        normals[:,1],
        normals[:,2],
        length=normal_length,
        color='red',
        normalize=True
    )
    ax.set_title(
        "Paint coverage simulation"
    )
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150
    )
    print(
        f"Saved paint coverage visualization: {output_path}"
    )

def main():

    scale = 0.01

    mesh_path = (
        Path(get_package_share_directory('environment'))
        / 'meshes'
        / 'obj.stl'
    )

    print(f"Loading mesh: {mesh_path}")

    mesh = trimesh.load(mesh_path)


    # If STL contains multiple objects
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)


    # Convert STL units to meters if needed
    mesh.apply_scale(scale)


    print(f"Mesh has {len(mesh.facets)} planar facets")
    print(f"Mesh bounds (meters): {mesh.bounds}")


    points, normals = generate_all_faces_zigzag(
        mesh,
        row_spacing=0.02
    )


    print(
        f"Generated {len(points)} total path points across all faces"
    )


    visualize_path(
    mesh,
    points,
    normals,
    '/tmp/zigzag_path.png'
    )


    visualize_paint_coverage(
        points,
        normals,
        output_path='/tmp/face_paint_visualization.png',
        spray_radius=0.02
    )


if __name__ == '__main__':
    main()

