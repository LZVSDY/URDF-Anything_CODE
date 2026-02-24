import os
import re
import json
import struct
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Any, Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation
import trimesh

def debug_palette_10():
    return np.array([
        [255,   0,   0],
        [  0, 255,   0],
        [  0,   0, 255],
        [255, 255,   0],
        [255,   0, 255],
        [  0, 255, 255],
        [255, 128,   0],
        [128,   0, 255],
        [  0, 255, 128],
        [255,   0, 128],
    ], dtype=np.uint8)


def split_points_by_palette(vertices: np.ndarray,
                            colors: np.ndarray,
                            n_parts: int,
                            palette: np.ndarray,
                            bg_colors: List[Tuple[int,int,int]] = [(50,50,50),(100,100,100)],
                            tol: int = 0):
    parts = []
    if bg_colors:
        bg = np.zeros((colors.shape[0],), dtype=bool)
        for c in bg_colors:
            c = np.array(c, dtype=np.uint8)
            if tol == 0:
                bg |= np.all(colors == c[None, :], axis=1)
            else:
                bg |= (np.abs(colors.astype(int) - c[None, :].astype(int)).sum(axis=1) <= tol)
    else:
        bg = np.zeros((colors.shape[0],), dtype=bool)

    for k in range(n_parts):
        c = palette[k]
        if tol == 0:
            mask = np.all(colors == c[None, :], axis=1) & (~bg)
        else:
            mask = (np.abs(colors.astype(int) - c[None, :].astype(int)).sum(axis=1) <= tol) & (~bg)
        parts.append(vertices[mask])
    return parts

def read_ply_xyzrgb(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header.append(line)
            if line == "end_header":
                break

        fmt = "ascii"
        n_vertices = 0
        props: List[Tuple[str, str]] = []
        in_vertex = False

        for line in header:
            if line.startswith("format"):
                if "binary_little_endian" in line:
                    fmt = "binary_le"
                elif "binary_big_endian" in line:
                    fmt = "binary_be"
                else:
                    fmt = "ascii"
            elif line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element") and not line.startswith("element vertex"):
                in_vertex = False
            elif in_vertex and line.startswith("property"):
                parts = line.split()
                if len(parts) >= 3:
                    props.append((parts[1], parts[2]))

        name2idx = {name: i for i, (_, name) in enumerate(props)}

        def idx_or_none(names):
            for nm in names:
                if nm in name2idx:
                    return name2idx[nm]
            return None

        ix = idx_or_none(["x"])
        iy = idx_or_none(["y"])
        iz = idx_or_none(["z"])
        ir = idx_or_none(["red", "r"])
        ig = idx_or_none(["green", "g"])
        ib = idx_or_none(["blue", "b"])

        if ix is None or iy is None or iz is None:
            raise ValueError(f"PLY missing xyz fields: {path}")
        if ir is None or ig is None or ib is None:
            raise ValueError(f"PLY missing rgb fields: {path}")

        vertices = np.zeros((n_vertices, 3), dtype=np.float64)
        colors = np.zeros((n_vertices, 3), dtype=np.uint8)

        if n_vertices == 0:
            return vertices, colors

        if fmt == "ascii":
            for i in range(n_vertices):
                line = f.readline().decode("ascii", errors="replace").strip()
                vals = line.split()
                vertices[i, 0] = float(vals[ix])
                vertices[i, 1] = float(vals[iy])
                vertices[i, 2] = float(vals[iz])
                r = float(vals[ir]); g = float(vals[ig]); b = float(vals[ib])

                if max(r, g, b) <= 1.0:
                    r, g, b = r * 255.0, g * 255.0, b * 255.0
                colors[i] = [int(round(r)), int(round(g)), int(round(b))]

        else:
            endian = "<" if fmt == "binary_le" else ">"
            type_map = {
                "float": "f", "float32": "f",
                "double": "d", "float64": "d",
                "uchar": "B", "uint8": "B",
                "char": "b", "int8": "b",
                "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
                "int": "i", "int32": "i", "uint": "I", "uint32": "I",
            }
            struct_chars = "".join(type_map.get(dt, "f") for dt, _ in props)
            st = struct.Struct(endian + struct_chars)

            for i in range(n_vertices):
                data = f.read(st.size)
                if len(data) < st.size:
                    break
                vals = st.unpack(data)

                vertices[i] = [float(vals[ix]), float(vals[iy]), float(vals[iz])]

                r = float(vals[ir]); g = float(vals[ig]); b = float(vals[ib])
                if max(r, g, b) <= 1.0:
                    r, g, b = r * 255.0, g * 255.0, b * 255.0
                colors[i] = [int(round(r)), int(round(g)), int(round(b))]

    return vertices, colors


def write_ply_xyzrgb(path: str, vertices: np.ndarray, colors: np.ndarray):
    """Write ASCII PLY with vertex colors."""
    N = vertices.shape[0]
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = vertices[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

def write_obj_points(path: str, vertices: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    pc = trimesh.points.PointCloud(vertices)
    pc.export(path)


def extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    brace = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0:
                return text[start : i + 1]
    return None


def parse_articulation(text_or_dict) -> Optional[dict]:
    if isinstance(text_or_dict, dict):
        return text_or_dict
    if isinstance(text_or_dict, str):
        js = extract_first_json_object(text_or_dict)
        if js:
            try:
                return json.loads(js)
            except Exception:
                js2 = re.sub(r",\s*([}\]])", r"\1", js)
                try:
                    return json.loads(js2)
                except Exception:
                    pass
    return None


def safe_list3(x, default=(0.0, 0.0, 0.0)):
    if isinstance(x, (list, tuple)) and len(x) >= 3:
        try:
            return [float(x[0]), float(x[1]), float(x[2])]
        except Exception:
            return list(default)
    return list(default)

def build_transform(xyz, rpy):
    R = Rotation.from_euler('xyz', safe_list3(rpy)).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = safe_list3(xyz)
    return T


def transform_points(T_inv: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones]) 
    pts_local = (T_inv @ pts_h.T).T[:, :3]
    return pts_local

def generate_urdf_xml(
    link_meshes: Dict[str, str],
    joints: List[dict],
    robot_name: str = "reconstructed",
) -> str:
    robot = ET.Element('robot', name=robot_name)

    ET.SubElement(robot, 'link', name='base')

    def link_sort_key(name):
        if name.startswith('link_') and name[5:].isdigit():
            return (0, int(name[5:]))
        return (1, name)

    sorted_link_names = sorted(link_meshes.keys(), key=link_sort_key)

    joint_by_child = {}
    for j in joints:
        if isinstance(j, dict) and "child" in j:
            joint_by_child[j["child"]] = j

    for link_name in sorted_link_names:
        link_elem = ET.SubElement(robot, 'link', name=link_name)
        mesh_path = link_meshes[link_name]
        if mesh_path:
            visual = ET.SubElement(link_elem, 'visual')
            ET.SubElement(visual, 'origin', xyz="0 0 0", rpy="0 0 0")
            geom = ET.SubElement(visual, 'geometry')
            ET.SubElement(geom, 'mesh', filename=mesh_path)

        ji = joint_by_child.get(link_name)
        if ji:
            jtype = ji.get("type", "fixed")
            jid = ji.get("id", f"joint_{link_name}")
            joint_elem = ET.SubElement(robot, 'joint', name=jid, type=jtype)

            parent = ji.get("parent", "base")
            ET.SubElement(joint_elem, 'parent', link=parent)
            ET.SubElement(joint_elem, 'child', link=link_name)

            origin = ji.get("origin", {})
            xyz_str = " ".join(f"{v:.5f}" for v in safe_list3(origin.get("xyz")))
            rpy_str = " ".join(f"{v:.5f}" for v in safe_list3(origin.get("rpy")))
            ET.SubElement(joint_elem, 'origin', xyz=xyz_str, rpy=rpy_str)

            axis = ji.get("axis", [1, 0, 0])
            axis_str = " ".join(f"{v:.4f}" for v in safe_list3(axis, default=(1, 0, 0)))
            ET.SubElement(joint_elem, 'axis', xyz=axis_str)

            if "limit" in ji:
                lim = ji["limit"]
                ET.SubElement(joint_elem, 'limit',
                              lower=str(lim.get("lower", 0.0)),
                              upper=str(lim.get("upper", 0.0)),
                              effort="100", velocity="1")
            elif jtype in ("revolute", "prismatic"):
                ET.SubElement(joint_elem, 'limit',
                              lower="-3.14", upper="3.14",
                              effort="100", velocity="1")
        else:
            joint_elem = ET.SubElement(robot, 'joint',
                                       name=f"joint_{link_name}_fixed",
                                       type="fixed")
            ET.SubElement(joint_elem, 'parent', link='base')
            ET.SubElement(joint_elem, 'child', link=link_name)
            ET.SubElement(joint_elem, 'origin', xyz="0 0 0", rpy="0 0 0")

    xml_str = minidom.parseString(ET.tostring(robot)).toprettyxml(indent="  ")
    lines = xml_str.split('\n')
    if lines[0].startswith('<?xml'):
        xml_str = '\n'.join(lines[1:])
    return xml_str

def reconstruct_single(
    content: dict,
    sample_id: str,
    out_dir: str,
    mode: str = "both", 
    denormalize: bool = False,
):
    norm_info = content.get("normalize", {})
    centroid = np.array(safe_list3(norm_info.get("centroid", [0, 0, 0])))
    scale = float(norm_info.get("scale", 1.0))

    seg_viz = content.get("seg_visualization", {})
    pred_fused_path = seg_viz.get("pred_fused_ply", "")
    gt_fused_path   = seg_viz.get("gt_fused_ply", "")


    modes_to_run = []
    if mode in ("pred", "both"):
        modes_to_run.append("pred")
    if mode in ("gt", "both"):
        modes_to_run.append("gt")

    for m in modes_to_run:
        if m == "pred":
            art = parse_articulation(content.get("pred_answers", ""))
            fused_ply = pred_fused_path
        else:
            art = content.get("answer")
            if art is None:
                art = parse_articulation(content.get("gt_answers", ""))
            fused_ply = gt_fused_path

        if (not fused_ply) or (not os.path.isfile(fused_ply)):
            print(f"    [{m}] fused ply not found: {fused_ply}")
            continue

        vertices_all, colors_all = read_ply_xyzrgb(fused_ply)
        uniq = np.unique(colors_all, axis=0)
        print(f"    [{m}] fused uniq colors={len(uniq)}, first20={uniq[:20].tolist()}")
        print(f"    [{m}] color min={colors_all.min(axis=0).tolist()} max={colors_all.max(axis=0).tolist()}")


        if art is None:
            print(f"    [{m}] Cannot parse articulation params, skipping")
            continue

        joints = art.get("joints", [])
        links_dict = art.get("links", {})

        def lsort(name):
            if name.startswith('link_') and name[5:].isdigit():
                return (0, int(name[5:]))
            return (1, name)

        link_names = sorted(
            [k for k in links_dict.keys() if k != 'base'],
            key=lsort
        )
        n_links = len(link_names)

        joint_by_child = {}
        transform_map = {} 
        for j in joints:
            if not isinstance(j, dict):
                continue
            child = j.get("child", "")
            joint_by_child[child] = j
            origin = j.get("origin", {})
            T = build_transform(origin.get("xyz", [0, 0, 0]),
                                origin.get("rpy", [0, 0, 0]))
            transform_map[child] = T
            
        palette = debug_palette_10()
        n_parts = len(link_names)
        if n_parts > len(palette):
            raise ValueError(f"n_parts={n_parts} > palette size={len(palette)}; extend palette")

        parts_world = split_points_by_palette(
            vertices_all, colors_all,
            n_parts=n_parts,
            palette=palette,
            bg_colors=[(30,30,30)],
            tol=0
        )


        m_dir = os.path.join(out_dir, m)
        mesh_dir = os.path.join(m_dir, "meshes")
        os.makedirs(mesh_dir, exist_ok=True)

        link_mesh_map = {}

        for link_idx, link_name in enumerate(link_names):
            part_pts = parts_world[link_idx]
            if part_pts.shape[0] == 0:
                print(f"    [{m}] No points for {link_name} from fused ply (palette idx={link_idx})")
                link_mesh_map[link_name] = ""
                continue

            if denormalize and scale > 0:
                part_pts = part_pts * scale + centroid

            T = transform_map.get(link_name, np.eye(4))
            if denormalize and scale > 0:
                T_denorm = T.copy()
                T_denorm[:3, 3] = T[:3, 3] * scale + centroid
                T_inv = np.linalg.inv(T_denorm)
            else:
                T_inv = np.linalg.inv(T)

            pts_local = transform_points(T_inv, part_pts)

            default_palette = [
                [31, 119, 180], [255, 127, 14], [44, 160, 44],
                [214, 39, 40], [148, 103, 189], [140, 86, 75],
                [227, 119, 194], [127, 127, 127], [188, 189, 34],
                [23, 190, 207],
            ]
            palette_color = default_palette[link_idx % len(default_palette)]
            export_colors = np.tile(palette_color, (pts_local.shape[0], 1)).astype(np.uint8)

            ply_name = f"{link_name}.ply"
            ply_path = os.path.join(mesh_dir, ply_name)
            write_ply_xyzrgb(ply_path, pts_local, export_colors)

            obj_name = f"{link_name}.obj"
            obj_path = os.path.join(mesh_dir, obj_name)
            write_obj_points(obj_path, pts_local)

            link_mesh_map[link_name] = f"meshes/{obj_name}"


        urdf_joints = joints
        if denormalize and scale > 0:
            urdf_joints = []
            for j in joints:
                jc = dict(j)
                if "origin" in jc:
                    oc = dict(jc["origin"])
                    xyz = safe_list3(oc.get("xyz"))
                    oc["xyz"] = [v * scale + c for v, c in zip(xyz, centroid)]
                    jc["origin"] = oc
                urdf_joints.append(jc)

        urdf_xml = generate_urdf_xml(link_mesh_map, urdf_joints,
                                      robot_name=f"{sample_id}_{m}")
        urdf_path = os.path.join(m_dir, "mobility.urdf")
        with open(urdf_path, "w") as f:
            f.write(urdf_xml)

        n_pts_total = sum(
            1 for ln in link_names if link_mesh_map.get(ln)
        )

def collect_json_files(root: str) -> List[str]:
    skip_dirs = {"seg_result", "seg_visualization", "seg_per_part",
                 "articulation_params", "reconstructed"}
    files = []
    for r, dirs, fns in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fn in fns:
            if fn.lower().endswith(".json"):
                files.append(os.path.join(r, fn))
    files.sort()
    return files


def main():
    ap = argparse.ArgumentParser(description="Reconstruct URDF from inference output")
    ap.add_argument("--root", type=str, required=True,
                    help="Root directory of inference output")
    ap.add_argument("--out_dir", type=str, default="",
                    help="Output dir (default: {root}/reconstructed)")
    ap.add_argument("--mode", type=str, default="both",
                    choices=["pred", "gt", "both"],
                    help="Reconstruct pred, gt, or both URDFs")
    ap.add_argument("--denormalize", action="store_true",
                    help="Denormalize coordinates to original scale")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="Max samples to process (0 = all)")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.root, "reconstructed")
    os.makedirs(out_dir, exist_ok=True)

    json_files = collect_json_files(args.root)
    if args.max_samples > 0:
        json_files = json_files[:args.max_samples]

    print(f"Found {len(json_files)} JSON files")
    print(f"Mode: {args.mode}, Denormalize: {args.denormalize}")
    print(f"Output: {out_dir}")
    print()

    n_ok = 0
    n_fail = 0

    for i, fp in enumerate(json_files):
        try:
            content = json.load(open(fp, "r"))
        except Exception as e:
            print(f"[{i}] SKIP {fp}: {e}")
            n_fail += 1
            continue

        sample_id = os.path.splitext(os.path.basename(fp))[0]
        sample_out = os.path.join(out_dir, sample_id)

        try:
            reconstruct_single(
                content=content,
                sample_id=sample_id,
                out_dir=sample_out,
                mode=args.mode,
                denormalize=args.denormalize,
            )
            n_ok += 1
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(json_files)}...")
        except Exception as e:
            print(f"[{i}] FAIL {sample_id}: {e}")
            n_fail += 1

    print(f"\nDone: {n_ok} OK, {n_fail} failed")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()