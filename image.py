"""
Photogrammetry Pipeline App
Upload images → COLMAP reconstruction → Open3D mesh → Scale → Download OBJ/STL
"""

import streamlit as st
import subprocess
import shutil
import os
import tempfile
import zipfile
from pathlib import Path

# ── Optional heavy imports ──────────────────────────────────────────────────
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

COLMAP_BIN = shutil.which("colmap") or "colmap"


# ── Helpers ─────────────────────────────────────────────────────────────────

def run_colmap(image_dir: Path, workspace: Path) -> bool:
    """Run full COLMAP automatic reconstruction pipeline."""
    db = workspace / "database.db"
    sparse = workspace / "sparse"
    dense = workspace / "dense"
    sparse.mkdir(exist_ok=True)
    dense.mkdir(exist_ok=True)

    steps = [
        # Feature extraction
        [COLMAP_BIN, "feature_extractor",
         "--database_path", str(db),
         "--image_path", str(image_dir),
         "--ImageReader.single_camera", "1"],
        # Feature matching
        [COLMAP_BIN, "exhaustive_matcher",
         "--database_path", str(db)],
        # Sparse reconstruction (mapper)
        [COLMAP_BIN, "mapper",
         "--database_path", str(db),
         "--image_path", str(image_dir),
         "--output_path", str(sparse)],
        # Image undistortion for dense
        [COLMAP_BIN, "image_undistorter",
         "--image_path", str(image_dir),
         "--input_path", str(sparse / "0"),
         "--output_path", str(dense),
         "--output_type", "COLMAP"],
        # Dense stereo
        [COLMAP_BIN, "patch_match_stereo",
         "--workspace_path", str(dense),
         "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "true"],
        # Stereo fusion
        [COLMAP_BIN, "stereo_fusion",
         "--workspace_path", str(dense),
         "--workspace_format", "COLMAP",
         "--input_type", "geometric",
         "--output_path", str(dense / "fused.ply")],
    ]

    logs = []
    for cmd in steps:
        st.write(f"▶ `{' '.join(cmd[:2])}`")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        logs.append(result.stdout + result.stderr)
        if result.returncode != 0:
            with st.expander("COLMAP error log"):
                st.code(result.stderr)
            return False

    with st.expander("Full COLMAP logs"):
        st.code("\n".join(logs))
    return True


def build_mesh_from_pointcloud(ply_path: Path, workspace: Path) -> Path | None:
    """Convert fused point cloud to mesh via Poisson reconstruction."""
    if not HAS_OPEN3D:
        st.error("open3d is not installed. Cannot build mesh.")
        return None

    st.write("🔷 Loading point cloud…")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    st.write(f"   Points: {len(pcd.points):,}")

    st.write("🔷 Estimating normals…")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    st.write("🔷 Poisson surface reconstruction…")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    # Remove low-density vertices (trim artifacts)
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    out = workspace / "mesh.ply"
    o3d.io.write_triangle_mesh(str(out), mesh)
    st.write(f"   Triangles: {len(mesh.triangles):,}")
    return out


def scale_mesh(mesh_path: Path, measured_px: float, real_length: float, unit: str) -> Path:
    """Scale mesh so that `measured_px` model-units equal `real_length` in chosen unit."""
    if not HAS_OPEN3D:
        return mesh_path

    unit_factors = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "ft": 0.3048}
    factor_m = unit_factors.get(unit, 1.0)
    real_m = real_length * factor_m
    scale = real_m / measured_px

    st.write(f"📐 Scale factor: {scale:.6f} (model units → metres)")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.scale(scale, center=mesh.get_center())

    scaled = mesh_path.parent / "mesh_scaled.ply"
    o3d.io.write_triangle_mesh(str(scaled), mesh)
    return scaled


def export_mesh(mesh_path: Path, fmt: str) -> Path:
    """Export mesh to OBJ or STL."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    out = mesh_path.parent / f"model.{fmt.lower()}"
    o3d.io.write_triangle_mesh(str(out), mesh)
    return out


def demo_sphere_mesh(workspace: Path) -> Path:
    """Generate a demo mesh when COLMAP is unavailable (for UI testing)."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    mesh.compute_vertex_normals()
    out = workspace / "mesh.ply"
    o3d.io.write_triangle_mesh(str(out), mesh)
    return out


# ── UI ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Photogrammetry Pipeline", page_icon="🧊", layout="wide")

st.title("🧊 Photogrammetry Pipeline")
st.caption("Upload images → COLMAP SfM + MVS → Open3D mesh → Scale → Download OBJ/STL")

# ── Sidebar: configuration ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    colmap_available = bool(shutil.which("colmap"))
    open3d_available = HAS_OPEN3D

    st.markdown("**Dependency status**")
    st.write("COLMAP:", "✅ found" if colmap_available else "❌ not found")
    st.write("Open3D:", "✅ found" if open3d_available else "❌ not found")

    st.divider()
    demo_mode = st.checkbox(
        "🎭 Demo mode (skip COLMAP, use sphere)",
        value=not colmap_available or not open3d_available,
        help="Runs without COLMAP/Open3D for UI preview."
    )

    st.divider()
    st.subheader("📐 Scale reference")
    ref_known = st.number_input("Known real-world length", min_value=0.001, value=10.0, step=0.1)
    ref_unit = st.selectbox("Unit", ["mm", "cm", "m", "in", "ft"], index=1)
    ref_model = st.number_input(
        "Corresponding length in model (units)",
        min_value=0.0001, value=1.0, step=0.001,
        help="Measure this distance in the point cloud / mesh using a 3D viewer."
    )
    apply_scale = st.checkbox("Apply scale", value=True)

    st.divider()
    export_format = st.selectbox("Export format", ["OBJ", "STL"], index=0)

# ── Main area ────────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "📁 Upload images (JPG / PNG / TIFF — at least 3 overlapping photos)",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=True,
)

if uploaded:
    st.success(f"{len(uploaded)} image(s) uploaded.")
    cols = st.columns(min(len(uploaded), 6))
    for i, f in enumerate(uploaded[:6]):
        cols[i].image(f, use_container_width=True, caption=f.name)
    if len(uploaded) > 6:
        st.caption(f"… and {len(uploaded)-6} more.")

st.divider()

run_btn = st.button("🚀 Run Pipeline", type="primary", disabled=(not uploaded and not demo_mode))

if run_btn or (demo_mode and st.session_state.get("_demo_triggered")):
    st.session_state["_demo_triggered"] = True

    workspace = Path(tempfile.mkdtemp(prefix="photogrammetry_"))
    image_dir = workspace / "images"
    image_dir.mkdir()

    # Save uploaded images
    if uploaded:
        for f in uploaded:
            (image_dir / f.name).write_bytes(f.read())

    mesh_ply = None

    # ── Step 1: COLMAP ──────────────────────────────────────────────────────
    with st.status("Step 1 / 3 — COLMAP reconstruction", expanded=True) as status:
        if demo_mode:
            st.info("Demo mode: skipping COLMAP, generating synthetic mesh.")
            if open3d_available:
                mesh_ply = demo_sphere_mesh(workspace)
                status.update(label="Step 1 / 3 — COLMAP reconstruction ✅ (demo)", state="complete")
            else:
                st.warning("Open3D not installed; cannot generate demo mesh either.")
                status.update(label="Step 1 / 3 — skipped (no Open3D)", state="error")
        else:
            if not colmap_available:
                st.error("COLMAP not found on PATH. Install it or enable Demo mode.")
                status.update(state="error")
            elif len(uploaded) < 3:
                st.warning("Please upload at least 3 images.")
                status.update(state="error")
            else:
                ok = run_colmap(image_dir, workspace)
                fused = workspace / "dense" / "fused.ply"
                if ok and fused.exists():
                    mesh_ply = fused
                    status.update(label="Step 1 / 3 — COLMAP reconstruction ✅", state="complete")
                else:
                    st.error("COLMAP reconstruction failed. Check logs above.")
                    status.update(state="error")

    # ── Step 2: Mesh generation ─────────────────────────────────────────────
    if mesh_ply and not demo_mode:
        with st.status("Step 2 / 3 — Building mesh (Poisson)", expanded=True) as status:
            if open3d_available:
                mesh_ply = build_mesh_from_pointcloud(mesh_ply, workspace)
                if mesh_ply:
                    status.update(label="Step 2 / 3 — Mesh built ✅", state="complete")
                else:
                    status.update(state="error")
            else:
                st.warning("Open3D not available; skipping meshing. Raw point cloud will be exported.")
                status.update(state="error")
    elif mesh_ply and demo_mode:
        st.info("Demo: synthetic sphere mesh ready — skipping Poisson step.")

    # ── Step 3: Scale & export ──────────────────────────────────────────────
    if mesh_ply and open3d_available:
        with st.status("Step 3 / 3 — Scale & export", expanded=True) as status:
            try:
                if apply_scale and ref_model > 0:
                    mesh_ply = scale_mesh(mesh_ply, ref_model, ref_known, ref_unit)

                export_path = export_mesh(mesh_ply, export_format)
                st.write(f"✅ Exported: `{export_path.name}`  ({export_path.stat().st_size/1024:.1f} KB)")
                status.update(label="Step 3 / 3 — Done ✅", state="complete")

                # Download button
                st.divider()
                with open(export_path, "rb") as fp:
                    st.download_button(
                        label=f"⬇️ Download {export_format} model",
                        data=fp,
                        file_name=export_path.name,
                        mime="application/octet-stream",
                        type="primary",
                    )

                # Also offer a zip with PLY + export
                zip_path = workspace / "model_package.zip"
                with zipfile.ZipFile(zip_path, "w") as zf:
                    zf.write(mesh_ply, mesh_ply.name)
                    zf.write(export_path, export_path.name)

                with open(zip_path, "rb") as fp:
                    st.download_button(
                        label="⬇️ Download full package (PLY + export)",
                        data=fp,
                        file_name="model_package.zip",
                        mime="application/zip",
                    )

            except Exception as e:
                st.error(f"Export failed: {e}")
                status.update(state="error")

# ── Install instructions ─────────────────────────────────────────────────────
with st.expander("📦 Installation & usage guide"):
    st.markdown("""
### Installation

```bash
# 1. Python dependencies
pip install streamlit open3d numpy

# 2. COLMAP  (choose your platform)
# Ubuntu/Debian:
sudo apt install colmap

# macOS (Homebrew):
brew install colmap

# Windows: download installer from https://colmap.github.io/install.html
```

### Running the app

```bash
streamlit run photogrammetry_app.py
```

### Tips for good reconstructions

- **Overlap**: each area should appear in at least 3 images (aim for 60–80 % overlap).
- **Lighting**: even, diffuse lighting; avoid reflective / transparent surfaces.
- **Camera**: disable auto-focus / auto-exposure if possible; use fixed ISO.
- **Reference object**: place a ruler or object of known size in the scene and enter its
  real-world length in the sidebar to get an accurate scale.

### Pipeline overview

```
Images → COLMAP feature extraction
       → COLMAP exhaustive matching
       → COLMAP sparse mapper (SfM)
       → COLMAP dense stereo (MVS)
       → COLMAP stereo fusion  → fused.ply
       → Open3D normal estimation
       → Open3D Poisson meshing
       → Scale by reference length
       → Export OBJ / STL
```
""")
