import sys

sys.path.append("..")
from ProbabilisticParcellation.util import *
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.similarity_colormap as cm
from Functional_Fusion.dataset import *
import matplotlib.pyplot as plt
import string
from ProbabilisticParcellation.scripts.parcel_hierarchy import analyze_parcel
import ProbabilisticParcellation.functional_profiles as fp
from ProbabilisticParcellation.scripts.explorer_layout import make_layout
from copy import deepcopy
import base64


# Import Dash dependencies
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output
from io import BytesIO
from jupyter_dash import JupyterDash
from wordcloud import WordCloud

base_dir = "/Volumes/diedrichsen_data$/data/FunctionalFusion"
if not Path(base_dir).exists():
    base_dir = "/srv/diedrichsen/data/FunctionalFusion"
if not Path(base_dir).exists():
    base_dir = "/Users/callithrix/Documents/Projects/Functional_Fusion/"
if not Path(base_dir).exists():
    raise (NameError("Could not find base_dir"))

atlas = "MNISymC2"

fine_model = f"/Models_03/sym_MdPoNiIbWmDeSo_space-{atlas}_K-68"
fileparts = fine_model.split("/")
split_mn = fileparts[-1].split("_")

# Get model, model info, Probabilities, parcellation and colour map
info_68, model_68 = load_batch_best(fine_model)
Prob_68 = np.array(model_68.marginal_prob())
parcel_68 = Prob_68.argmax(axis=0) + 1
cmap_68 = cm.read_cmap(
    f"{model_dir}/Atlases/sym_MdPoNiIbWmDeSo_space-{atlas}_K-68.cmap"
)
info = fp.recover_info(info_68, model_68, fine_model)

# Get labels
w_cos_sym, _, _ = cl.parcel_similarity(model_68, plot=True, sym=True)
labels_68, clusters, _ = cl.agglomative_clustering(
    w_cos_sym, sym=True, num_clusters=5, plot=False
)
info["labels"] = labels_68


# Get task profiles
profile = pd.read_csv(
    f'{model_dir}/Atlases/{fine_model.split("/")[-1]}_task_profile_data.tsv', sep="\t"
)

# Make flatmap plot
cerebellum = plot_data_flat(
    parcel_68, atlas, cmap=cmap_68, dtype="label", labels=info.labels, render="plotly"
)

# ----- Function Explorer -----
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
application = app.server
app.layout = make_layout(cerebellum)


@app.callback(
    Output("image_wc", "src"),
    Input(component_id="image_wc", component_property="src"),
    Input(component_id="figure-cerebellum", component_property="clickData"),
)
def plot_wordcloud(b, region):
    """Makes word cloud image to be loaded into app
    Args:
        b: parcel scores for each condition in each dataset
        region: selected region for which to show functional profile

    Returns:
        word_cloud: word cloud image png object

    """
    img = BytesIO()
    wc = fp.get_wordcloud(profile, selected_region=region)

    # # Colour conditions by source dataset (takes a long time to load)
    # wc.recolor(color_func=fp.dataset_colours)
    wc.to_image().save(img, format="PNG")
    word_cloud = "data:image/png;base64,{}".format(
        base64.b64encode(img.getvalue()).decode()
    )
    return word_cloud


if __name__ == "__main__":
    app.run_server(debug=True)
