import os
import glob
import shutil
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

def _gpu_knn(points: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    N = points.shape[0]
    if N <= 8192:
        diff = points.unsqueeze(1) - points.unsqueeze(0) 
        dist_sq = (diff * diff).sum(-1) 
        dist_sq.fill_diagonal_(float('inf'))
        topk_dists_sq, topk_idxs = dist_sq.topk(k, dim=1, largest=False)
        return topk_dists_sq.sqrt(), topk_idxs
    else:
        chunk_size = 4096
        all_dists = []
        all_idxs = []
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            diff = points[i:end].unsqueeze(1) - points.unsqueeze(0) 
            dist_sq = (diff * diff).sum(-1) 
            self_mask = torch.arange(i, end, device=points.device)
            dist_sq[torch.arange(end - i, device=points.device), self_mask] = float('inf')
            topk_sq, topk_idx = dist_sq.topk(k, dim=1, largest=False)
            all_dists.append(topk_sq.sqrt())
            all_idxs.append(topk_idx)
        return torch.cat(all_dists, 0), torch.cat(all_idxs, 0)

def _gpu_remove_statistical_outlier(
    points: np.ndarray,
    nb_neighbors: int = 16,
    std_ratio: float = 2.5,
    device: torch.device = None,
) -> np.ndarray:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N = len(points)
    if N < nb_neighbors + 1:
        return points

    pts_t = torch.from_numpy(points.astype(np.float32)).to(device)
    dists, _ = _gpu_knn(pts_t, nb_neighbors)
    mean_dists = dists.mean(dim=1)

    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    threshold = global_mean + std_ratio * global_std

    mask = mean_dists <= threshold
    return pts_t[mask].cpu().numpy()

def _gpu_estimate_normals(
    points: np.ndarray,
    k: int = 30,
    device: torch.device = None,
) -> np.ndarray:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N = len(points)
    pts_t = torch.from_numpy(points.astype(np.float32)).to(device)

    k_actual = min(k, N - 1)
    _, knn_idx = _gpu_knn(pts_t, k_actual)

    neighbors = pts_t[knn_idx]

    centroid = neighbors.mean(dim=1, keepdim=True)
    centered = neighbors - centroid

    cov = torch.bmm(centered.transpose(1, 2), centered) / k_actual

    eigenvalues, eigenvectors = torch.linalg.eigh(cov) 
    normals = eigenvectors[:, :, 0] 

    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)

    return normals.cpu().numpy()


def _gpu_estimate_avg_nn_dist(
    points: np.ndarray,
    k: int = 6,
    device: torch.device = None,
) -> float:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pts_t = torch.from_numpy(points.astype(np.float32)).to(device)
    k_actual = min(k, len(points) - 1)
    dists, _ = _gpu_knn(pts_t, k_actual)
    return float(dists.mean().item())

def _points_to_mesh_gpu(
    points_np: np.ndarray,
    fast_mode: bool = False,
    device: torch.device = None,
):
    try:
        import open3d as o3d
    except ImportError:
        return _scipy_convex_hull_to_o3d_like(points_np)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n = len(points_np)
    if n < 4:
        return None

    if n >= 20:
        nb = min(16, n // 3)
        pts_clean = _gpu_remove_statistical_outlier(
            points_np, nb_neighbors=nb, std_ratio=2.5, device=device
        )
    else:
        pts_clean = points_np.copy()

    n2 = len(pts_clean)
    if n2 < 4:
        return None

    k_norm = min(30, max(5, n2 - 1))
    normals = _gpu_estimate_normals(pts_clean, k=k_norm, device=device)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_clean.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))

    if n2 >= 10:
        try:
            pcd.orient_normals_consistent_tangent_plane(k=min(15, n2 - 1))
        except Exception:
            pass

    mesh = None

    if not fast_mode and n2 >= 80:
        try:
            m_p, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=6, width=0, scale=1.1, linear_fit=False,
            )
            dens_arr = np.asarray(densities)
            thresh = np.quantile(dens_arr, 0.05)
            m_p.remove_vertices_by_mask(dens_arr < thresh)
            m_p.remove_degenerate_triangles()
            m_p.remove_unreferenced_vertices()

            tri_clusters, n_tris_per_cluster, _ = m_p.cluster_connected_triangles()
            if len(n_tris_per_cluster) > 1:
                largest = int(np.argmax(n_tris_per_cluster))
                m_p.remove_triangles_by_mask(np.asarray(tri_clusters) != largest)
                m_p.remove_unreferenced_vertices()

            if len(np.asarray(m_p.triangles)) >= 20:
                mesh = m_p
        except Exception:
            pass

    if mesh is None:
        try:
            pts_arr = np.asarray(pcd.points).astype(np.float32)
            k_nn = min(6, n2 - 1)
            avg_nn = _gpu_estimate_avg_nn_dist(pts_arr, k=k_nn, device=device)

            best_mesh_a = None
            best_n_tri = 0
            for alpha_mult in (3.5, 5.0, 2.0):
                alpha = avg_nn * alpha_mult
                m_a = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
                m_a.remove_degenerate_triangles()
                m_a.remove_unreferenced_vertices()
                n_tri = len(np.asarray(m_a.triangles))
                if n_tri > best_n_tri:
                    best_n_tri = n_tri
                    best_mesh_a = m_a

            if best_mesh_a is not None and best_n_tri >= 4:
                tri_clusters, n_tris_per_cluster, _ = best_mesh_a.cluster_connected_triangles()
                if len(n_tris_per_cluster) > 1:
                    largest = int(np.argmax(n_tris_per_cluster))
                    best_mesh_a.remove_triangles_by_mask(
                        np.asarray(tri_clusters) != largest
                    )
                    best_mesh_a.remove_unreferenced_vertices()
                mesh = best_mesh_a
        except Exception:
            pass

    if mesh is None or len(np.asarray(mesh.triangles)) < 4:
        try:
            hull, _ = pcd.compute_convex_hull()
            mesh = hull
        except Exception:
            pass

    if mesh is not None and len(np.asarray(mesh.triangles)) > 0:
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.compute_vertex_normals()

    return mesh


def _read_obj_points(obj_path: str) -> np.ndarray:
    verts = []
    with open(obj_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0] == 'v':
                try:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except (ValueError, IndexError):
                    pass
    return np.array(verts, dtype=np.float32) if verts else np.zeros((0, 3), dtype=np.float32)


def _scipy_convex_hull_to_o3d_like(points_np: np.ndarray):
    from scipy.spatial import ConvexHull

    class _SimpleMesh:
        def __init__(self, verts, faces, normals=None):
            self.vertices = verts
            self.triangles = faces
            self.vertex_normals = normals if normals is not None else np.zeros_like(verts)

    if len(points_np) < 4:
        return None
    try:
        hull = ConvexHull(points_np.astype(np.float64))
        return _SimpleMesh(hull.points, hull.simplices)
    except Exception:
        return None


def _save_mesh_as_obj_generic(mesh, obj_path: str) -> None:
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals) if hasattr(mesh, 'vertex_normals') else np.array([])
    has_normals = (len(normals) == len(verts) and len(normals) > 0)

    with open(obj_path, 'w') as f:
        f.write("# Mesh reconstructed from point cloud\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if has_normals:
            for vn in normals:
                f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
        for tri in faces:
            i0, i1, i2 = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
            if has_normals:
                f.write(f"f {i0}//{i0} {i1}//{i1} {i2}//{i2}\n")
            else:
                f.write(f"f {i0} {i1} {i2}\n")

def _process_single_link(
    obj_path: str,
    fast_mode: bool,
    device: torch.device,
    verbose: bool,
) -> str:
    basename = os.path.basename(obj_path)
    stem = basename[:-4]
    obj_dir = os.path.dirname(obj_path)

    if stem.endswith("_ply"):
        return "skip"

    ply_obj_path = os.path.join(obj_dir, f"{stem}_ply.obj")

    shutil.move(obj_path, ply_obj_path)

    points = _read_obj_points(ply_obj_path)

    if len(points) < 4:
        shutil.copy(ply_obj_path, obj_path)
        if verbose:
            print(f"    [mesh_convert_gpu] {stem}: only {len(points)} pts → copied ply")
        return "fallback"

    mesh = None
    try:
        mesh = _points_to_mesh_gpu(points, fast_mode=fast_mode, device=device)
    except Exception as e:
        if verbose:
            print(f"    [mesh_convert_gpu] {stem}: reconstruction error: {e}")

    n_tri = len(np.asarray(mesh.triangles)) if mesh is not None else 0

    if mesh is not None and n_tri >= 4:
        try:
            _save_mesh_as_obj_generic(mesh, obj_path)
            n_v = len(np.asarray(mesh.vertices))
            if verbose:
                print(f"    [mesh_convert_gpu] {stem}: {len(points)} pts → mesh {n_v}v/{n_tri}f ✓")
            return "ok"
        except Exception as e:
            shutil.copy(ply_obj_path, obj_path)
            if verbose:
                print(f"    [mesh_convert_gpu] {stem}: save failed ({e}) → fallback")
            return "fallback"
    else:
        shutil.copy(ply_obj_path, obj_path)
        if verbose:
            print(f"    [mesh_convert_gpu] {stem}: degenerate → fallback")
        return "fallback"

def convert_link_objs_to_mesh_gpu(
    out_dir: str,
    verbose: bool = True,
    fast_mode: bool = False,
    max_workers: int = 4,
    device: torch.device = None,
) -> None:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    patterns = [
        os.path.join(out_dir, "link_*.obj"),
        os.path.join(out_dir, "*", "link_*.obj"),
        os.path.join(out_dir, "*", "*", "link_*.obj"),
    ]
    all_obj_files = []
    for pat in patterns:
        all_obj_files.extend(glob.glob(pat))
    all_obj_files = sorted(set(all_obj_files))

    if not all_obj_files:
        if verbose:
            print(f"    [mesh_convert_gpu] No link_*.obj found in {out_dir}")
        return

    ok_count = 0
    fallback_count = 0
    skip_count = 0

    if max_workers > 1 and len(all_obj_files) > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(all_obj_files))) as pool:
            futures = {
                pool.submit(
                    _process_single_link, obj_path, fast_mode, device, verbose
                ): obj_path
                for obj_path in all_obj_files
            }
            for fut in as_completed(futures):
                result = fut.result()
                if result == "ok":
                    ok_count += 1
                elif result == "fallback":
                    fallback_count += 1
                else:
                    skip_count += 1
    else:
        for obj_path in all_obj_files:
            result = _process_single_link(obj_path, fast_mode, device, verbose)
            if result == "ok":
                ok_count += 1
            elif result == "fallback":
                fallback_count += 1
            else:
                skip_count += 1

    if verbose:
        total = ok_count + fallback_count + skip_count
        mode_str = "FAST" if fast_mode else "FULL"
        print(f"    [mesh_convert_gpu] [{mode_str}] Summary: {ok_count} mesh / "
              f"{fallback_count} fallback / {skip_count} skipped  (total: {total})")

convert_link_objs_to_mesh = convert_link_objs_to_mesh_gpu