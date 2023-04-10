import os
import sys
import pandas as pd
import numpy as np 
import requests

from io import BytesIO
from glob import glob
from PIL import Image, ImageEnhance

import matplotlib.pyplot as plt
from PIL import ImageColor
import colorsys
import streamlit as st
import plotly.express as px

from sklearn.cluster import KMeans, BisectingKMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE

import streamlit as st



model_dict = {
    "KMeans": KMeans,
    "BisectingKMeans" : BisectingKMeans,
    "GaussianMixture": GaussianMixture,
    "MiniBatchKMeans": MiniBatchKMeans,
}

center_method = {
    "KMeans": "cluster_centers_",
    "BisectingKMeans" : "cluster_centers_",
    "GaussianMixture": "means_",
    "MiniBatchKMeans": "cluster_centers_",
}

n_cluster_arg = {
    "KMeans": "n_clusters",
    "BisectingKMeans" : "n_clusters",
    "GaussianMixture": "n_components",
    "MiniBatchKMeans": "n_clusters",

}

enhancement_range = {
    "Color": [0., 5., 0.2], 
    "Sharpness": [0., 3., 0.2], 
    "Contrast": [0.5, 1.5, 0.1], 
    "Brightness": [0.5, 1.5, 0.1]
}

sort_func_dict = {
    "rgb": (lambda r,g,b: (r, g, b)),
    "sum_rgb": (lambda r,g,b: r+g+b),
    "sqr_rgb": (lambda r,g,b: r**2+g**2+b**2),
    "hsv": (lambda r, g, b : colorsys.rgb_to_hsv(r, g, b)),
    "random": (lambda r, g, b: np.random.random())
}

def get_df_rgb(img, sample_size):
    """construct a sample RGB dataframe from image"""

    n_dims = np.array(img).shape[-1]
    r,g,b = np.array(img).reshape(-1,n_dims).T
    df = pd.DataFrame({"R": r, "G": g, "B": b}).sample(n=sample_size)
    return df

def get_palette(df_rgb, model_name, palette_size, sort_func="random"):
    """cluster pixels together and return a sorted color palette."""
    params = {n_cluster_arg[model_name]: palette_size}
    model = model_dict[model_name](**params)

    clusters = model.fit_predict(df_rgb)
        
    palette = getattr(model, center_method[model_name]).astype(int).tolist()
    
    palette.sort(key=lambda rgb : sort_func_dict[sort_func.rstrip("_r")](*rgb), 
                reverse=bool(sort_func.endswith("_r")))

    return palette

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

def show_palette(palette_hex):
    """show palette strip"""
    palette = np.array([ImageColor.getcolor(color, "RGB") for color in  palette_hex])
    fig, ax = plt.subplots(dpi=100)
    ax.imshow(palette[np.newaxis, :, :])
    ax.axis('off')
    return fig


def store_palette(palette):
    """store palette colors in session state"""
    palette_size = len(palette)
    columns = st.columns(palette_size)
    for i, col in enumerate(columns):
        with col:        
            st.session_state[f"col_{i}"]= st.color_picker(label=str(i), value=rgb_to_hex(palette[i]), key=f"pal_{i}")

def display_matplotlib_code(palette_hex):

    st.write('Use this snippet in your code to make your color palette more sophisticated!')
    code = st.code(f"""
import matplotlib as mpl
from cycler import cycler
palette = {palette_hex}
mpl.rcParams["axes.prop_cycle"] = cycler(color=palette)
    """
    )   

def display_plotly_code(palette_hex):
    st.write('Use this snippet in your code to make your color palette more sophisticated!')
    st.code(f"""
import plotly.io as pio
import plotly.graph_objects as go
pio.templates["sophisticated"] = go.layout.Template(
    layout=go.Layout(
    colorway={palette_hex}
    )
)
pio.templates.default = 'sophisticated'
            """)

def plot_rgb_3d(df_rgb):
    """plot the sampled pixels in 3D RGB space"""

    if df_rgb.shape[0] > 2000:
        st.error("RGB plot can only be used for less than 2000 sample pixels.")
    else:
        colors = df_rgb.apply(rgb_to_hex, axis=1)
        fig = px.scatter_3d(df_rgb, x='R', y='G', z='B',
                color=colors, size=[1]*df_rgb.shape[0],
                opacity=0.7)

        st.plotly_chart(fig)


def plot_hsv_3d(df):
    """plot the sampled pixels in 3D RGB space"""
    df_rgb = df.copy()
    if df_rgb.shape[0] > 2000:
        st.error("RGB plot can only be used for less than 2000 sample pixels.")

    else:
        df_rgb[["H","S",'V']]= df_rgb.apply(lambda x: pd.Series(colorsys.rgb_to_hsv(x.R/255.,x.G/255.,x.B/255.)).T, axis=1)
        st.dataframe(df_rgb[["H","S",'V']])
        colors = df_rgb[["R","G","B"]].apply(rgb_to_hex, axis=1)
        fig = px.scatter_3d(df_rgb, x='H', y='S', z='V',
                color=colors, size=[1]*df_rgb.shape[0],
                opacity=0.7)

        st.plotly_chart(fig)

    
sys.path.insert(0, ".")


gallery_files = glob(os.path.join(".", "images", "*"))
gallery_dict = {image_path.split("/")[-1].split(".")[-2].replace("-", " "): image_path
    for image_path in gallery_files}


toggle = st.sidebar.checkbox("Toggle Update", value=True, help="Continuously update the pallete with every change in the app.")
click = st.sidebar.button("Find Palette", disabled=bool(toggle))

st.sidebar.markdown("---")
st.sidebar.header("Settings")
palette_size = int(st.sidebar.number_input("palette size", min_value=1, max_value=20, value=5, step=1, help="Number of colors to infer from the image."))
sample_size = int(st.sidebar.number_input("sample size", min_value=5, max_value=3000, value=500, step=500, help="Number of sample pixels to pick from the image."))

# Image Enhancement
enhancement_categories = enhancement_range.keys()
enh_expander = st.sidebar.expander("Image Enhancements", expanded=False)
with enh_expander:
    
    if st.button("reset"):
        for cat in enhancement_categories:
            if f"{cat}_enhancement" in st.session_state:
                st.session_state[f"{cat}_enhancement"] = 1.0
enhancement_factor_dict = {
    cat: enh_expander.slider(f"{cat} Enhancement", 
                            value=1., 
                            min_value=enhancement_range[cat][0], 
                            max_value=enhancement_range[cat][1], 
                            step=enhancement_range[cat][2],
                            key=f"{cat}_enhancement")
    for cat in enhancement_categories
}
enh_expander.info("**Try the following**\n\nColor Enhancements = 2.6\n\nContrast Enhancements = 1.1\n\nBrightness Enhancements = 1.1")

# Clustering Model 
model_name = st.sidebar.selectbox("machine learning model", model_dict.keys(), help="Machine Learning model to use for clustering pixels and colors together.")
sklearn_info = st.sidebar.empty()

sort_options = sorted(list(sort_func_dict.keys()) + [key + "_r" for key in sort_func_dict.keys() if key!="random"])
sort_func = st.sidebar.selectbox("palette sort function", options=sort_options, index=5)

# Random Number Seed
seed = int(st.sidebar.number_input("random seed", value=42, help="Seed used for all random samplings."))
np.random.seed(seed)
st.sidebar.markdown("---")


# =======
#   App
# =======

# provide options to either select an image form the gallery, upload one, or fetch from URL
gallery_tab, upload_tab  = st.tabs(["Gallery", "Upload"])
with gallery_tab:
    options = list(gallery_dict.keys())
    file_name = st.selectbox("Select Art", 
                            options=options, index=0)
    file = gallery_dict[file_name]

    if st.session_state.get("file_uploader") is not None:
        st.warning("To use the Gallery, remove the uploaded image first.")
    if st.session_state.get("image_url") not in ["", None]:
        st.warning("To use the Gallery, remove the image URL first.")

    img = Image.open(file)

with upload_tab:
    file = st.file_uploader("Upload Art", key="file_uploader")
    if file is not None:
        try:
            img = Image.open(file)
        except:
            st.error("The file you uploaded does not seem to be a valid image. Try uploading a png or jpg file.")
    if st.session_state.get("image_url") not in ["", None]:
        st.warning("To use the file uploader, remove the image URL first.")


# convert RGBA to RGB if necessary
n_dims = np.array(img).shape[-1]
if n_dims == 4:
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
    img = background

# apply image enhancements
for cat in enhancement_categories:
    img = getattr(ImageEnhance, cat)(img)
    img = img.enhance(enhancement_factor_dict[cat])

# show the image
with st.expander("", expanded=True):
    st.image(img, use_column_width=True)


if click or toggle:
    
    df_rgb = get_df_rgb(img, sample_size)

    # (optional for later)
    # plot_rgb_3d(df_rgb) 
    # plot_hsv_3d(df_rgb) 

    # calculate the RGB palette and cache it to session_state
    st.session_state["palette_rgb"] = get_palette(df_rgb, model_name, palette_size, sort_func=sort_func)

    if "palette_rgb" in st.session_state:
        
        # store individual colors in session state
        store_palette(st.session_state["palette_rgb"])

        st.write("---")

        # sort the colors based on the selected option
        colors = {k: v for k, v in st.session_state.items() if k.startswith("col_")}
        sorted_colors = {k: colors[k] for k in sorted(colors, key=lambda k: int(k.split("_")[-1]))}
        
        # find the hex representation for matplotlib and plotly settings
        palette_hex = [color for color in sorted_colors.values()][:palette_size]
        with st.expander("Adopt this Palette", expanded=False):
            st.pyplot(show_palette(palette_hex))

            matplotlib_tab, plotly_tab = st.tabs(["matplotlib", "plotly"])

            with matplotlib_tab:
                display_matplotlib_code(palette_hex)

                import matplotlib as mpl
                from cycler import cycler

                mpl.rcParams["axes.prop_cycle"] = cycler(color=palette_hex)
                import matplotlib.pyplot as plt

                x = np.arange(5)
                y_list = np.random.random((len(palette_hex), 5))+2
                df = pd.DataFrame(y_list).T

                area_tab, bar_tab = st.tabs(["area chart", "bar chart"])

                with area_tab:
                    fig_area , ax_area = plt.subplots()
                    df.plot(kind="area", ax=ax_area, backend="matplotlib", )  
                    st.header("Example Area Chart")
                    st.pyplot(fig_area)
    
                with bar_tab:
                    fig_bar , ax_bar = plt.subplots()
                    df.plot(kind="bar", ax=ax_bar, stacked=True, backend="matplotlib", )
                    st.header("Example Bar Chart")
                    st.pyplot(fig_bar)

                
            with plotly_tab:
                display_plotly_code(palette_hex)

                import plotly.io as pio
                import plotly.graph_objects as go
                pio.templates["sophisticated"] = go.layout.Template(
                    layout=go.Layout(
                    colorway=palette_hex
                    )
                )
                pio.templates.default = 'sophisticated'

                area_tab, bar_tab = st.tabs(["area chart", "bar chart"])

                with area_tab:
                    fig_area = df.plot(kind="area", backend="plotly", )
                    st.header("Example Area Chart")
                    st.plotly_chart(fig_area, use_container_width=True)
    
                with bar_tab:
                    fig_bar = df.plot(kind="bar", backend="plotly", barmode="stack")
                    st.header("Example Bar Chart")
                    st.plotly_chart(fig_bar, use_container_width=True)

       
else:
    st.info("Click on 'Find Palette' ot turn on 'Toggle Update' to see the color palette.")

   