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


def make_layout(cerebellum):
    layout = html.Div(
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
                ],
                style={"width": "49%", "display": "inline-block"},
            ),
        ],
        style={"display": "flex", "flex-direction": "row"},
    )
    return layout
