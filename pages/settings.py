import dash
from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/settings")

layout = dbc.Container([
      dcc.Store(id="selected-data-type", data="demand"), 

    dbc.Row([
    # Product Information Card 
        dbc.Col([
            dbc.Row(
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id='selected-product-output',className="text-primary fw-bold"),
                        html.P(id={"type": "i18n", "key": "settings.user_settings_description"}),
                        html.P(id={"type": "i18n", "key": "settings.settings_list_instruction"}),
                    ])
            ])),
            # dbc.Row([
            #         dbc.Card([
            #             dbc.CardBody(
            #                 dbc.Row([
            #                     dbc.Col([
            #                         dbc.Card(
            #                             # the green box the contains the text "De-Anonymization Settings"
            #                             dbc.CardBody([
            #                                 html.Div(id={"type": "i18n", "key": "settings.de_anonymization_title"}),
            #                             ]),
            #                             style={
            #                                 "backgroundColor": "#006400",
            #                                 "color": "white",
            #                                 "border": "1px solid #004b23"
            #                             }
            #                         )], width="auto"),
            #                     # the switch for de-anonymization
            #                     dbc.Col([
            #                         dbc.Switch(id="De-Anonymization-Switch", value = False, persistence = True, persistence_type = "session",
            #                                    style={"transform": "scale(1.5)"}
            #                         )], width="auto", align = "center"),
            #                 ], justify="between"),
            #             )
            #         ])
            # ]),
            dbc.Row([
                dbc.Card([
                    dbc.CardBody(
                        dbc.Row([
                            dbc.Col([
                                dbc.Card(
                                    # the green box the contains the text "Language Settings"
                                    dbc.CardBody([
                                        html.Div(id={"type": "i18n", "key": "settings.language_title"}),
                                    ]),
                                    style={
                                        "backgroundColor": "#006400",
                                        "color": "white",
                                        "border": "1px solid #004b23"
                                    }
                                )], width="auto"),
                            # the switch for changing language
                            dbc.Col(
                                html.Div([
                                    html.Span("de",style={"marginRight": "20px", "fontWeight": "bold", "fontSize": "1.2em"}),
                                    dbc.Switch(
                                        id="language-Switch",
                                        value=False,
                                        persistence=True,
                                        persistence_type="session",
                                        style={"transform": "scale(1.5)"}
                                    ),
                                    html.Span("en", style={"marginLeft": "10px", "fontWeight": "bold", "fontSize": "1.2em"})
                                ],
                                style={"display": "flex", "alignItems": "center"}
                                ),
                                width="auto",
                                align="center"),
                        ], justify="between"),
                    )
                ])
            ]),
        ], width=6)
    ]),
])