import streamlit as st
import os
import subprocess
import shutil
from pathlib import Path
import open3d as o3d
import numpy as np
from PIL import Image

# --- Configuration & Paths ---
WORKSPACE_DIR = Path("photogrammetry_workspace")
IMAGE_DIR = WORKSPACE_DIR / "images"
DATABASE_PATH = WORKSPACE_DIR / "database.db"
SPARSE_DIR = WORKSPACE_DIR / "sparse"
DENSE_DIR = WORKSPACE_DIR / "dense"
MODEL_PATH = DENSE_DIR / "fused.ply"

# --- Helper Functions ---

def run_command(command):
    """Executes a shell command and displays output in Streamlit."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True)
    for line in process.stdout:
        st.text(line.strip())
    process.wait()
    return process.returncode

def run_colmap_pipeline():
    """Manual execution of the COLMAP SfM and MVS pipeline."""
    st.info("Step 1: Extracting Features...")
    run_command(f"colmap feature_extractor --database_path {DATABASE_PATH} --image_path {IMAGE_DIR}")
    
    st.info("Step 2: Matching Features...")
    run_command(f"colmap exhaustive_matcher --database_path {DATABASE_PATH}")
    
    st.info("Step 3: Sparse Reconstruction (Mapper)...")
    SPARSE_DIR.mkdir(exist_ok=True)
    run_command(f"colmap mapper --database_path {DATABASE_PATH} --image_path {IMAGE_DIR} --output_path {SPARSE_DIR}")
    
    st.info("Step 4: Image Undistortion & Dense Preparation...")
    DENSE_DIR.mkdir(exist_ok=True)
    # Note: Using '0' assumes the first generated model is the main one
    run_command(f"colmap image_undistorter --image_path {IMAGE_DIR} --input_path {SPARSE_DIR}/0 --output_path {DENSE_DIR} --output_type COLMAP")
    
    st.info("Step 5: Patch Match Stereo (Dense Depth Estimation)...")
    run_command(f"colmap patch_match_stereo --workspace_path {DENSE_DIR} --workspace_format COLMAP --PatchMatchStereo.geom_consistency true")
    
    st.info("Step 6: Stereo Fusion (Point Cloud Generation)...")
    run_command(f"colmap stereo_fusion --workspace_path {DENSE_DIR} --workspace_format COLMAP --input_type geometric --output_path {MODEL_PATH}")

def create_mesh(point_cloud_path, poisson_depth=9):
    """Converts point cloud to a watertight mesh using Poisson Reconstruction."""
    pcd = o3d.io.read_point_cloud(str(point_cloud_path))
    
    # Ensure normals exist (required for Poisson)
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Poisson Surface Reconstruction
    with st.spinner("Running Poisson Reconstruction..."):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
    
    # Remove low-density artifacts (Poisson creates a 'bubble' around the object)
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    return mesh

# --- UI Layout ---

st.title("📸 3D Photogrammetry & Scaler")
st.markdown("Upload 20–50 overlapping images to generate a dimensionally accurate 3D model.")

# 1. Image Upload
uploaded_files = st.file_uploader("Upload Images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    if st.button("Start 3D Reconstruction"):
        # Reset workspace
        if WORKSPACE_DIR.exists(): shutil.rmtree(WORKSPACE_DIR)
        IMAGE_DIR.mkdir(parents=True)
        
        # Save images
        for idx, file in enumerate(uploaded_files):
            img = Image.open(file)
            img.save(IMAGE_DIR / f"img_{idx}.jpg")
        
        # Run COLMAP
        run_colmap_pipeline()
        st.success("Point cloud generated successfully!")

# 2. Scaling & Meshing
if MODEL_PATH.exists():
    st.divider()
    st.header("📏 Scaling & Mesh Export")
    
    col1, col2 = st.columns(2)
    with col1:
        ref_real_mm = st.number_input("Real-world reference length (mm)", value=297.0, help="Length of your reference object (e.g., A4 height)")
    with col2:
        ref_model_units = st.number_input("Measured length in model units", value=1.0, step=0.001, format="%.4f")
    
    if st.button("Generate Scaled Mesh"):
        # Calculate scale factor
        scale_factor = ref_real_mm / ref_model_units
        
        # Generate mesh
        mesh = create_mesh(MODEL_PATH)
        
        # Apply scaling
        # center=False ensures we don't move the mesh, just scale coordinates
        mesh.scale(scale_factor, center=mesh.get_center())
        
        # Save outputs
        obj_file = DENSE_DIR / "final_model.obj"
        stl_file = DENSE_DIR / "final_model.stl"
        
        o3d.io.write_triangle_mesh(str(obj_file), mesh)
        o3d.io.write_triangle_mesh(str(stl_file), mesh)
        
        st.success(f"Model scaled by {scale_factor:.2f}x and saved.")
        
        # Download Buttons
        with open(obj_file, "rb") as f:
            st.download_button("Download OBJ", f, file_name="model_scaled.obj")
        with open(stl_file, "rb") as f:
            st.download_button("Download STL", f, file_name="model_scaled.stl")

else:
    st.info("Upload images and run reconstruction to see scaling options.")
