import streamlit as st
import brainrender
from brainrender.scene import Scene
import pyvista as pv

st.set_page_config(page_title="Glass Brain", layout="wide")
st.title("Glass Brain Viewer")

# Fetch the whole brain mesh from brainrender
scene = Scene()
root_mesh = scene.add_brain_region("root")

# Convert to PyVista PolyData
if root_mesh is not None:
    # Extract vertices and faces from the mesh
    vertices = root_mesh.points
    faces = root_mesh.faces
    
    # Create PyVista PolyData
    poly_data = pv.PolyData(vertices, faces)
    
    # Create plotter
    plotter = pv.Plotter()
    plotter.add_mesh(poly_data, color="grey", opacity=0.8)
    plotter.camera_position = "iso"
    
    # Display in Streamlit
    st.pyplot(plotter.screenshot())
else:
    st.error("Failed to load brain mesh")