import dash
from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import logging

from . import functions as fn

## Dash Page for historical article Forecasts
dash.register_page(__name__, path="/article")

dropdown_options = fn.get_article_dropdown_options()
layout = dbc.Container([
    #dcc.Store(id="selected-data-type", data="demand"),

    dbc.Row([
        # Product Information Card
        # dbc.Col([
        #     dbc.Card([
        #         dbc.CardBody([
        #             html.H4(id='selected-product-output',className="text-primary fw-bold"),
        #             html.P(id={"type": "i18n", "key": "article.profile"}),
        #             html.P(id={"type": "i18n", "key": "article.more_details"}),
        #         ])
        #     ])
        # ], width=6),
        # Dropdown and Selected Product Output This can be done later with a dict with all items
        dbc.Col([

            dcc.Dropdown(
                    id='end-product-dropdown-article',
                    options=dropdown_options,
                    placeholder="",
                    className="mb-3"
                ),
            html.Div(id='selected-product-output3', className="text-primary fw-bold"),

        ], width=6),

    ], className="mt-3"),
    # type Buttons to select which data type should be displayed added later
    dbc.Row(

    ),

    # graph place holder
    dbc.Row([
        dbc.Col([
        dcc.Graph(id='fc-graph')
        ], width=12)
    ], className="mt-4"),

    #Placeholder for Analytical Stats
    dbc.Row([
        #   Analytical Metrics & Prediction Values
        # Analytical Metrics Table
        dbc.Card([
            dbc.CardHeader(id={"type": "i18n", "key": "article.metric_title"}),
            dbc.CardBody([
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th(id={"type": "i18n", "key": "article.metric"}), html.Th(id={"type": "i18n", "key": "article.train_size"}), html.Th(id={"type": "i18n", "key": "article.test_size"}),
                        html.Th("RMSE"), html.Th("MSE"), html.Th("MPIW"), html.Th("PICP")
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(id={"type": "i18n", "key": "article.value"}),  # First column for labeling
                            html.Td(id="train-size-output"),
                            html.Td(id="test-size-output"),
                            html.Td(id="rmse-output"),
                            html.Td(id="mse-output"),
                            html.Td(id="mpiw-output"),
                            html.Td(id="picp-output")
                        ])
                    ])
                ], bordered=True, hover=True, responsive=True)
            ])
        ], className="mb-4"),

        # # Prediction Values Table Comparision
        # dbc.Card([
        #     dbc.CardHeader(id={"type": "i18n", "key": "article.pred_value"}),
        #     dbc.CardBody([
        #         dbc.Table([
        #             html.Thead(html.Tr([
        #                 html.Th(id={"type": "i18n", "key": "article.month"}), html.Th(id={"type": "i18n", "key": "article.jan"}), html.Th(id={"type": "i18n", "key": "article.feb"}), html.Th(id={"type": "i18n", "key": "article.mar"}),
        #                 html.Th(id={"type": "i18n", "key": "article.apr"}), html.Th(id={"type": "i18n", "key": "article.may"}), html.Th(id={"type": "i18n", "key": "article.jun"}), html.Th(id={"type": "i18n", "key": "article.jul"}),
        #                 html.Th(id={"type": "i18n", "key": "article.aug"}), html.Th(id={"type": "i18n", "key": "article.sep"}), html.Th(id={"type": "i18n", "key": "article.okt"}), html.Th(id={"type": "i18n", "key": "article.nov"}), html.Th(id={"type": "i18n", "key": "article.dec"})
        #             ])),
        #             html.Tbody([
        #                 html.Tr([
        #                     html.Td(id={"type": "i18n", "key": "article.pred"}),
        #                     html.Td(id="jan-pred-output"), html.Td(id="feb-pred-output"), html.Td(id="mar-pred-output"),
        #                     html.Td(id="apr-pred-output"), html.Td(id="may-pred-output"), html.Td(id="jun-pred-output"),
        #                     html.Td(id="jul-pred-output"), html.Td(id="aug-pred-output"), html.Td(id="sep-pred-output"),
        #                     html.Td(id="oct-pred-output"), html.Td(id="nov-pred-output"), html.Td(id="dec-pred-output")
        #                 ]),
        #                 html.Tr([
        #                     html.Td(id={"type": "i18n", "key": "article.actual_val"}),
        #                     html.Td(id="jan-actual-output"), html.Td(id="feb-actual-output"), html.Td(id="mar-actual-output"),
        #                     html.Td(id="apr-actual-output"), html.Td(id="may-actual-output"), html.Td(id="jun-actual-output"),
        #                     html.Td(id="jul-actual-output"), html.Td(id="aug-actual-output"), html.Td(id="sep-actual-output"),
        #                     html.Td(id="oct-actual-output"), html.Td(id="nov-actual-output"), html.Td(id="dec-actual-output")
        #                 ]),
        #                 html.Tr([
        #                     html.Td(id={"type": "i18n", "key": "article.sel_display"}),
        #                     *[html.Td(dcc.Checklist(options=[{"label": "", "value": "show"}])) for _ in range(12)]
        #                 ])
        #             ])
        #         ], bordered=True, hover=True, responsive=True)
        #     ])
        # ])
        # Parameters & Inputs
    ]),
], fluid=True)

@callback(
    Output('end-product-dropdown-article', 'placeholder'),
    Input('language-store', 'data')
)

def update_dropdown_placeholders(language):
    first = fn.outside_translate("article.dropdown_placeholder")
    return first

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

@callback(
    [Output("fc-graph", "figure"),
            Output("train-size-output", "children"),
            Output("test-size-output", "children"),
            Output("rmse-output", "children"),
            Output("mse-output", "children"),
            Output("mpiw-output", "children"),
            Output("picp-output", "children"),
     ],
    [Input("end-product-dropdown-article", "value"),]
)
def update_forecast_graph(selected_item):
    logging.info(f"Selected item: {selected_item}. Processing data and forecasts")
    if not selected_item:
        return  go.Figure(), "-", "-", "-", "-", "-", "-"
    X_train, future_dates, y_true = fn.get_data_with_true(selected_item)
    predictions, lower_bounds, upper_bounds, _ = fn.recursive_prediction_darts_with_split(
        selected_item,
        X_train,
        future_dates,
        y_true,
        quantile=[0.025, 0.5, 0.975],
        sample=5000,
        lag=15
    )
    results_df: pd.DataFrame = pd.DataFrame(predictions,columns=[
        fn.PREDICTED_DATE_COLUMN, fn.PREDICTED_COLUMN, fn.PREDICTED_TRUE_COLUMN])
    results_df[fn.LOWER_BOUND_COLUMN] = lower_bounds
    results_df[fn.UPPER_BOUND_COLUMN] = upper_bounds

    results_df[fn.PREDICTED_DATE_COLUMN] = pd.to_datetime(
        results_df[fn.PREDICTED_DATE_COLUMN])  # Ensure the 'Date' column is in datetime format.
    results_df = results_df.sort_values(by=fn.PREDICTED_DATE_COLUMN)  # Sort the DataFrame by date for proper plotting.
    last_day: pd.Timestamp = results_df[fn.PREDICTED_DATE_COLUMN].min()

    # Metrics
    rmse,mse,mae,mpiw,picp = fn.calculate_metrics(results_df[fn.PREDICTED_TRUE_COLUMN],results_df[fn.PREDICTED_COLUMN],results_df[fn.LOWER_BOUND_COLUMN],results_df[fn.UPPER_BOUND_COLUMN])
    train_size = len(X_train)
    test_size = len(y_true)

    true_demand = fn.outside_translate("article.true_demand")
    pred_demand = fn.outside_translate("article.pred_demand")
    upper_bound = fn.outside_translate("article.upper_bound")
    lower_bound = fn.outside_translate("article.lower_bound")
    graph_title = fn.outside_translate("article.graph_title").format(y_true=len(y_true))
    xaxis_title = fn.outside_translate("article.xaxis_title")
    yaxis_title = fn.outside_translate("article.yaxis_title")

    fig = go.Figure()
    # Add trace for demand
    fig.add_trace(
        go.Scatter(
            x=results_df[fn.PREDICTED_DATE_COLUMN],
            y=results_df[fn.PREDICTED_TRUE_COLUMN],
            mode='lines+markers',
            name=true_demand,
            line=dict(color='blue')
        )
    )
    # trace for predicted demand
    fig.add_trace(
        go.Scatter(
            x=results_df[fn.PREDICTED_DATE_COLUMN],
            y=results_df[fn.PREDICTED_COLUMN],
            mode='lines+markers',
            name=pred_demand,
            line=dict(color='orange')
        )
    )
    # trace for upper bound
    fig.add_trace(
        go.Scatter(
            x=results_df[fn.PREDICTED_DATE_COLUMN],
            y=results_df[fn.UPPER_BOUND_COLUMN],
            mode='lines',
            name=upper_bound,
            line=dict(color='grey')
        )
    )
    # trace for lower bound
    fig.add_trace(
        go.Scatter(
            x=results_df[fn.PREDICTED_DATE_COLUMN],
            y=results_df[fn.LOWER_BOUND_COLUMN],
            mode='lines',
            name=lower_bound,
            line=dict(color='grey'),
            fill='tonexty'
        )
    )
    fig.update_layout(
        title=graph_title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode="x unified"
    )
    logging.info(f"Returned Graph")
    return fig,train_size,test_size,rmse,mse,mpiw,picp