import argparse
import numpy as np
import trimesh
import trimesh.smoothing

# uv pip install pymesh trimesh networkx scikit-image open3d

def decimate(mesh, target_faces):
    import open3d as o3d
    print(f"\nDecimating {len(mesh.faces)} -> ~{target_faces} faces (quadric edge collapse)...")
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
    )
    o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    return trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=False,
    )


def fill_holes(mesh, max_hole_size):
    """Robust hole filling via open3d (handles holes of any boundary length
    up to max_hole_size, unlike trimesh.repair.fill_holes which only fills
    3-edge triangular holes)."""
    import open3d as o3d
    o3d_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.int32)),
    )
    n_before = len(mesh.faces)
    filled = o3d_mesh.fill_holes(hole_size=max_hole_size)
    new_verts = filled.vertex["positions"].numpy()
    new_faces = filled.triangle["indices"].numpy()
    result = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    print(f"Hole-fill: added {len(result.faces) - n_before} faces "
          f"(max hole size {max_hole_size})")
    return result


def cleanup_needles(mesh, max_aspect):
    """Remove triangles with longest_edge / shortest_edge > max_aspect, then
    fill the resulting (typically single-triangle) holes."""
    tri = mesh.vertices[mesh.faces]
    e = np.stack([
        np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
        np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
        np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
    ], axis=1)
    aspect = e.max(axis=1) / np.maximum(e.min(axis=1), 1e-12)
    keep = aspect <= max_aspect
    n_removed = int((~keep).sum())
    if n_removed == 0:
        print(f"No needle faces above aspect ratio {max_aspect}")
        return mesh
    print(f"Removing {n_removed} needle faces (aspect > {max_aspect})...")
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    filled = trimesh.repair.fill_holes(mesh)
    print(f"Hole filling after needle removal: {'ok' if filled else 'partial'}")
    return mesh


def reconstruct_voxel(mesh, pitch):
    print(f"\nVoxelizing with pitch={pitch}...")
    voxels = mesh.voxelized(pitch=pitch)
    print(f"Voxel grid shape: {voxels.matrix.shape}")

    print("Filling interior...")
    voxels = voxels.fill()

    print("Running marching cubes...")
    return voxels.marching_cubes


def _fibonacci_sphere(n):
    indices = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def _outer_hull_pointcloud(mesh, n_views, rays_per_view):
    """Cast parallel rays from viewpoints on a sphere; keep first-hit points
    with their source-triangle normals (flipped to face the viewpoint).
    Preserves original face orientations — a flat diagonal face stays flat."""
    import open3d as o3d

    bbox_center = mesh.bounds.mean(axis=0)
    bbox_radius = float(np.linalg.norm(mesh.extents) / 2.0)
    view_distance = bbox_radius * 3.0

    o3d_mesh_t = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(np.asarray(mesh.faces), dtype=o3d.core.Dtype.Int32),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d_mesh_t)

    tri_normals = np.asarray(mesh.face_normals)
    viewpoints = _fibonacci_sphere(n_views) * view_distance + bbox_center
    grid_n = int(np.ceil(np.sqrt(rays_per_view)))

    print(f"Ray-casting from {n_views} viewpoints, ~{grid_n*grid_n} rays each "
          f"({n_views * grid_n * grid_n} total)...")

    all_points = []
    all_normals = []

    u = np.linspace(-bbox_radius, bbox_radius, grid_n)
    v = np.linspace(-bbox_radius, bbox_radius, grid_n)
    uu, vv = np.meshgrid(u, v)
    plane_uv = np.stack([uu.flatten(), vv.flatten()], axis=1)

    for vp in viewpoints:
        view_dir = bbox_center - vp
        view_dir /= np.linalg.norm(view_dir)

        up_ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(view_dir, up_ref)) > 0.99:
            up_ref = np.array([1.0, 0.0, 0.0])
        right = np.cross(view_dir, up_ref)
        right /= np.linalg.norm(right)
        up = np.cross(right, view_dir)

        offsets = plane_uv[:, 0:1] * right + plane_uv[:, 1:2] * up
        origins = vp + offsets
        directions = np.tile(view_dir, (origins.shape[0], 1))

        rays = np.concatenate([origins, directions], axis=1).astype(np.float32)
        result = scene.cast_rays(o3d.core.Tensor(rays))
        t_hit = result['t_hit'].numpy()
        prim_ids = result['primitive_ids'].numpy()

        valid = np.isfinite(t_hit)
        if not np.any(valid):
            continue

        hit_points = origins[valid] + t_hit[valid, None] * view_dir
        hit_normals = tri_normals[prim_ids[valid]]

        flip = np.einsum('ij,j->i', hit_normals, view_dir) > 0.0
        hit_normals[flip] *= -1.0

        all_points.append(hit_points)
        all_normals.append(hit_normals)

    points = np.concatenate(all_points, axis=0)
    normals = np.concatenate(all_normals, axis=0)
    print(f"Collected {len(points)} outer-surface points (with original normals)")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def reconstruct_poisson(mesh, depth, n_views, rays_per_view, density_quantile,
                         scale, linear_fit, dedup_voxel, bbox_margin,
                         outlier_std, outlier_neighbors):
    import open3d as o3d

    pcd = _outer_hull_pointcloud(mesh, n_views, rays_per_view)

    if dedup_voxel > 0.0:
        before = len(pcd.points)
        pcd = pcd.voxel_down_sample(dedup_voxel)
        print(f"Down-sampled point cloud: {before} -> {len(pcd.points)} points "
              f"(voxel={dedup_voxel})")

    if outlier_std > 0.0:
        before = len(pcd.points)
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=outlier_neighbors, std_ratio=outlier_std
        )
        print(f"Statistical outlier removal: {before} -> {len(pcd.points)} points "
              f"(std_ratio={outlier_std}, neighbors={outlier_neighbors})")

    print(f"Running Poisson reconstruction (depth={depth}, scale={scale}, "
          f"linear_fit={linear_fit})...")
    rec_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=linear_fit
    )
    densities = np.asarray(densities)
    print(f"Reconstructed {len(rec_mesh.triangles)} faces, "
          f"{len(rec_mesh.vertices)} vertices")

    if density_quantile > 0.0:
        threshold = np.quantile(densities, density_quantile)
        print(f"Cropping vertices with density < {threshold:.4f} "
              f"(quantile={density_quantile})...")
        rec_mesh.remove_vertices_by_mask(densities < threshold)
        print(f"After crop: {len(rec_mesh.triangles)} faces, "
              f"{len(rec_mesh.vertices)} vertices")

    if bbox_margin > 0.0:
        margin = bbox_margin * float(np.linalg.norm(mesh.extents))
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=mesh.bounds[0] - margin,
            max_bound=mesh.bounds[1] + margin,
        )
        before = len(rec_mesh.triangles)
        rec_mesh = rec_mesh.crop(bbox)
        print(f"Bbox-cropped (margin={margin:.4f}): {before} -> "
              f"{len(rec_mesh.triangles)} faces")

    return trimesh.Trimesh(
        vertices=np.asarray(rec_mesh.vertices),
        faces=np.asarray(rec_mesh.triangles),
        process=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Clean a CAD mesh by voxelizing or Poisson-reconstructing the outer surface." \
        "Example usage:" \
        "python3 mesh_repairer.py --target-faces 1000 --input-scale 01000 --input-dir ./input_folder --output-dir ./output_folder"
    )
    parser.add_argument(
        "input", nargs="?", default=None,
        help="Input STL file (single-file mode). Omit when using --input-dir."
    )
    parser.add_argument(
        "--input-dir",
        type=str, default=None,
        help="Batch mode: process every .stl file in this folder. Requires --output-dir."
    )
    parser.add_argument(
        "--output-dir",
        type=str, default=None,
        help="Batch mode: write each processed mesh to this folder under the same filename. "
             "Used together with --input-dir; all files get the same processing options."
    )
    parser.add_argument(
        "-m", "--method",
        choices=["voxel", "poisson"], default="poisson",
        help="Reconstruction method. 'poisson' uses raycasting + open3d Poisson "
             "and preserves original face orientations. Default: voxel"
    )
    parser.add_argument(
        "-p", "--pitch",
        type=float, default=0.01,
        help="[voxel] Voxel pitch (smaller = more detail, slower). Default: 0.01"
    )
    parser.add_argument(
        "--poisson-depth",
        type=int, default=9,
        help="[poisson] Octree depth (higher = more detail). Default: 9"
    )
    parser.add_argument(
        "--poisson-views",
        type=int, default=64,
        help="[poisson] Number of viewpoints on a sphere around the mesh. Default: 64"
    )
    parser.add_argument(
        "--poisson-rays-per-view",
        type=int, default=4096,
        help="[poisson] Approx number of parallel rays per viewpoint. Default: 4096"
    )
    parser.add_argument(
        "--poisson-dedup-voxel",
        type=float, default=0.0,
        help="[poisson] Voxel size for point-cloud down-sampling (0 disables). Default: 0"
    )
    parser.add_argument(
        "--poisson-density-quantile",
        type=float, default=0.01,
        help="[poisson] Drop vertices below this density quantile (0 disables). Default: 0.01"
    )
    parser.add_argument(
        "--poisson-scale",
        type=float, default=1.1,
        help="[poisson] Reconstruction domain scale relative to bbox. Default: 1.1"
    )
    parser.add_argument(
        "--poisson-outlier-std",
        type=float, default=2.0,
        help="[poisson] Statistical outlier removal: drop points whose mean "
             "distance to k nearest neighbors exceeds this many std devs. "
             "Removes the points that cause inward-spike artifacts. "
             "Lower = more aggressive. 0 disables. Default: 2.0"
    )
    parser.add_argument(
        "--poisson-outlier-neighbors",
        type=int, default=20,
        help="[poisson] k for the outlier-removal kNN search. Default: 20"
    )
    parser.add_argument(
        "--poisson-bbox-margin",
        type=float, default=0.02,
        help="[poisson] Crop output to input bbox expanded by this fraction of "
             "bbox diagonal (kills inward drift; 0 disables). Default: 0.02"
    )
    parser.add_argument(
        "--poisson-linear-fit",
        action="store_true",
        help="[poisson] Use linear interpolation for iso-vertex positions."
    )
    parser.add_argument(
        "-s", "--smooth-iterations",
        type=int, default=10,
        help="Laplacian smoothing iterations. Default: 10"
    )
    parser.add_argument(
        "-l", "--lamb",
        type=float, default=0.5,
        help="Laplacian smoothing lambda (0.0-1.0). Default: 0.5"
    )
    parser.add_argument(
        "--cleanup-needles",
        type=float, default=0.0,
        help="Drop faces with edge aspect ratio above this and patch holes. "
             "Off by default — too aggressive for dense Poisson output. "
             "Try 50+ if you do enable it. Default: 0 (off)"
    )
    parser.add_argument(
        "--fill-holes",
        type=float, default=1e6,
        help="Final pass: fill holes whose boundary length is below this. "
             "0 disables. Default: 1e6 (effectively unlimited)"
    )
    parser.add_argument(
        "--input-scale",
        type=float, default=1.0,
        help="Uniform scale factor applied immediately after loading (e.g. "
             "0.001 for mm -> m). Affects all unit-dependent params downstream. "
             "Default: 1.0"
    )
    parser.add_argument(
        "--scale",
        type=float, default=1.0,
        help="Uniform scale factor applied as the final step (e.g. 0.001 for "
             "mm -> m). Default: 1.0"
    )
    parser.add_argument(
        "--target-faces",
        type=int, default=0,
        help="Decimate output to ~N triangles (0 disables). Default: 0"
    )
    parser.add_argument(
        "-o", "--output",
        type=str, default=None,
        help="Output filename. If not specified, auto-generated from options."
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Just print mesh info and exit, no processing."
    )

    args = parser.parse_args()

    import os
    import glob

    def process_one(in_path, out_path, args):
        print(f"Loading {in_path}...")
        mesh = trimesh.load(in_path, process=False, force='mesh')
        print(f"Loaded {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")
        print(f"Bounds: {mesh.bounds}")
        print(f"Extents: {mesh.extents}")
        if args.input_scale != 1.0:
            print(f"Input scaling by {args.input_scale}...")
            mesh.apply_scale(args.input_scale)

        if args.info:
            components = mesh.split(only_watertight=False)
            print(f"\nSplit into {len(components)} components:")
            for i, c in enumerate(components):
                print(f"  Component {i}: {len(c.faces)} faces, volume: {c.volume:.4f}")
            return

        if args.method == "voxel":
            outer = reconstruct_voxel(mesh, args.pitch)
        else:
            outer = reconstruct_poisson(
                mesh,
                depth=args.poisson_depth,
                n_views=args.poisson_views,
                rays_per_view=args.poisson_rays_per_view,
                density_quantile=args.poisson_density_quantile,
                scale=args.poisson_scale,
                linear_fit=args.poisson_linear_fit,
                dedup_voxel=args.poisson_dedup_voxel,
                bbox_margin=args.poisson_bbox_margin,
                outlier_std=args.poisson_outlier_std,
                outlier_neighbors=args.poisson_outlier_neighbors,
            )

        if args.cleanup_needles > 0:
            outer = cleanup_needles(outer, args.cleanup_needles)

        if args.smooth_iterations > 0:
            print(f"Smoothing (iterations={args.smooth_iterations}, lambda={args.lamb})...")
            trimesh.smoothing.filter_laplacian(outer, lamb=args.lamb, iterations=args.smooth_iterations)

        if args.target_faces > 0:
            outer = decimate(outer, args.target_faces)

        if args.fill_holes > 0:
            outer = fill_holes(outer, args.fill_holes)

        if args.scale != 1.0:
            print(f"Scaling by {args.scale}...")
            outer.apply_scale(args.scale)

        print(f"Final bounds: {outer.bounds}")

        if out_path is None:
            base = in_path.rsplit('.', 1)[0]
            if args.method == "voxel":
                out_path = f"{base}_voxel_p{args.pitch}_s{args.smooth_iterations}_l{args.lamb}.stl"
            else:
                out_path = (f"{base}_poisson_d{args.poisson_depth}"
                            f"_v{args.poisson_views}_r{args.poisson_rays_per_view}"
                            f"_s{args.smooth_iterations}_l{args.lamb}.stl")

        print(f"\nSaving to {out_path}...")
        outer.export(out_path)
        print(f"Done. Final mesh: {len(outer.faces)} faces, {len(outer.vertices)} vertices")

        edge_counts = np.bincount(outer.edges_unique_inverse)
        boundary_edges = int(np.sum(edge_counts == 1))
        nonmanifold_edges = int(np.sum(edge_counts > 2))

        print("\nWatertightness check:")
        print(f"  is_watertight:         {outer.is_watertight}")
        print(f"  is_winding_consistent: {outer.is_winding_consistent}")
        print(f"  euler_number:          {outer.euler_number}  (closed sphere-topology = 2)")
        print(f"  boundary edges:        {boundary_edges}  (open holes; should be 0)")
        print(f"  non-manifold edges:    {nonmanifold_edges}  (3+ faces share an edge; should be 0)")
        print(f"  components:            {len(outer.split(only_watertight=False))}")
        if not outer.is_watertight:
            broken = trimesh.repair.broken_faces(outer)
            print(f"  broken faces:          {len(broken)} (touching a boundary edge)")

    if args.input_dir is not None:
        if args.output_dir is None:
            parser.error("--output-dir is required when --input-dir is given")
        os.makedirs(args.output_dir, exist_ok=True)
        stl_files = sorted(f for f in glob.glob(os.path.join(args.input_dir, "*"))
                           if f.lower().endswith(".stl"))
        if not stl_files:
            print(f"No .stl files found in {args.input_dir}")
            return
        print(f"Found {len(stl_files)} .stl file(s) in {args.input_dir}")
        for i, in_path in enumerate(stl_files):
            out_path = os.path.join(args.output_dir, os.path.basename(in_path))
            print(f"\n{'='*70}\n[{i+1}/{len(stl_files)}] {os.path.basename(in_path)} -> {out_path}\n{'='*70}")
            try:
                process_one(in_path, out_path, args)
            except Exception as e:
                print(f"ERROR processing {in_path}: {type(e).__name__}: {e}")
    else:
        if args.input is None:
            parser.error("provide an input STL file, or use --input-dir together with --output-dir for batch processing")
        process_one(args.input, args.output, args)


if __name__ == "__main__":
    main()
