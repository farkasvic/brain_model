from __future__ import annotations

import streamlit as st
import pyvista as pv
import plotly.graph_objects as go

from src.brain_data import get_region_acronyms, load_region_mesh
from src.voxelize import voxelize_mesh


def get_region_mesh(region_name: str) -> pv.PolyData:
    """Load (or retrieve from cache) a region mesh."""
    key = f"region_mesh_{region_name}"
    if key not in st.session_state:
        with st.spinner(f"Loading {region_name}..."):
            try:
                st.session_state[key] = load_region_mesh(region_name)
            except Exception as exc:
                st.error(f"Failed to load region '{region_name}': {exc}")
                raise exc
    return st.session_state[key]


def main() -> None:
    st.set_page_config(page_title="Neuro-Canvas: Brain Regions", layout="wide")

    st.title("Neuro-Canvas")
    st.subheader("Voxelized brain regions")

    with st.sidebar:
        st.header("Controls")
        if "region_acronyms" not in st.session_state:
            with st.spinner("Loading atlas..."):
                try:
                    st.session_state["region_acronyms"] = get_region_acronyms()
                except Exception as exc:
                    st.error(f"Failed to load atlas: {exc}")
                    st.stop()
        region_options = st.session_state["region_acronyms"]
        selected_regions = st.multiselect(
            "Brain Regions",
            options=region_options,
            default=[],
            help="Select regions to visualize as voxel grids.",
        )
        density = st.slider(
            "Voxel Density",
            min_value=0.02,
            max_value=0.3,
            value=0.1,
            step=0.01,
            help="Grid spacing and voxel size. Smaller = finer grid.",
        )

    if not selected_regions:
        st.info("Select one or more brain regions in the sidebar to visualize.")
        return

    traces = []
    for region in selected_regions:
        try:
            mesh = get_region_mesh(region)
            vox_mesh = voxelize_mesh(mesh, density=density)
            vox_mesh = vox_mesh.triangulate()
            points = vox_mesh.points
            faces = vox_mesh.faces.reshape(-1, 4)[:, 1:]
            color = "lightgrey"
            traces.append(
                go.Mesh3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=color,
                    opacity=0.5,
                    name=region,
                )
            )
        except Exception as exc:
            st.error(f"Error processing {region}: {exc}")

    if not traces:
        st.warning("Could not load any selected regions.")
        return

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
