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
parcel_profiles, profile_data = fp.get_profiles(model=model_68, info=info)

# Make flatmap plot
cerebellum = plot_data_flat(
    parcel_68, atlas, cmap=cmap_68, dtype="label", labels=info.labels, render="plotly"
)

# start of app
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
application = app.server


click_region_labels = dcc.Markdown(id="clicked-region")


app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Functional Atlas Explorer"),
                html.Div(
                    [
                        dcc.Graph(
                            id="figure-cerebellum",
                            figure=cerebellum,
                            clear_on_unhover=False,
                        ),
                        dcc.Tooltip(id="graph-tooltip"),
                    ]
                ),
            ],
            style={"width": "49%", "display": "inline-block"},
        ),
        html.Div(
            [
                html.P("Display functions for a selected region and dataset."),
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        # html.H5('Word Cloud',className='text-center'),
                                                        html.Img(id="image_wc"),
                                                    ]
                                                )
                                            ]
                                        )
                                    ],
                                    width={"size": 12, "offset": 0, "order": 1},
                                    style={"padding-left": 25, "padding-right": 25},
                                    className="text-center",
                                ),
                            ]
                        )
                    ]
                ),
                html.Div(
                    [
                        html.H4(id="clicked-region"),
                    ]
                ),
            ],
            style={"width": "49%", "display": "inline-block"},
        ),
    ],
    style={"display": "flex", "flex-direction": "row"},
)


@app.callback(
    Output("image_wc", "src"),
    Input(component_id="image_wc", component_property="src"),
    Input(component_id="figure-cerebellum", component_property="clickData"),
)
def make_image(b, region):
    img = BytesIO()
    fp.plot_wordcloud(parcel_profiles, profile_data, labels_68, region).save(
        img, format="PNG"
    )
    word_cloud = "data:image/png;base64,{}".format(
        base64.b64encode(img.getvalue()).decode()
    )
    return word_cloud


if __name__ == "__main__":
    app.run_server(debug=True)