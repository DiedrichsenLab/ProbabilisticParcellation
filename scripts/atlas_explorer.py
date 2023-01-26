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

# start of app
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])


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
                    children=[
                        html.Label("Dataset"),
                        dcc.Dropdown(
                            datasets,
                            id="chosen_dataset",
                            value=datasets[0],
                            clearable=False,
                        ),
                    ],
                    style={"padding": 10, "flex": 1},
                ),
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


def plot_wordcloud(df, dset, region):
    reg = "A1L"
    # When initiliazing the website and if clickin on a null region, show no conditions
    if region is not None and region["points"][0]["text"] != "0":
        # get the region name
        reg = region["points"][0]["text"]
    d = df.conditions[(df.dataset == dset) & (df.label == reg)]
    wc = WordCloud(background_color="white", width=512, height=384).generate(
        " ".join(d)
    )
    return wc.to_image()


@app.callback(
    Output("image_wc", "src"),
    # Input(component_id='figure-cerebellum', component_property='clickData'),
    Input(component_id="image_wc", component_property="src"),
    Input(component_id="chosen_dataset", component_property="value"),
    Input(component_id="figure-cerebellum", component_property="clickData"),
)
def make_image(b, dset, region):
    img = BytesIO()
    plot_wordcloud(df, dset, region).save(img, format="PNG")
    return "data:image/png;base64,{}".format(base64.b64encode(img.getvalue()).decode())


if __name__ == "__main__":
    app.run_server(debug=True)
