import logging

import dash
from dash import callback, Input, Output, State, callback_context, dash_table
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional, Any
from . import functions as fn
from . import dash_functions as dfn

# Suppress warnings if needed (as in the notebook)
warnings.filterwarnings('ignore')

# Register the page with Dash
dash.register_page(__name__, path="/")


dates = {0: "2020", 1: "2021", 2: "2022", 3: "2023", 4: "2024"}
slider_store = dcc.Store(id='slider_store', data={'year_range': ("2020", "2024")})
mode_store   = dcc.Store(id='mode_store',   data={'mode': 'treemap', 'item': None})

layout = dbc.Container([
    dbc.Row(html.H1(id={"type": "i18n", "key": "home.overall_item_usage"}, className="text-center my-4")),
    dbc.Row([
        html.H4(id={"type": "i18n", "key": "home.select_date_range"}),
        dcc.RangeSlider(
            id="year-range-slider",
            min=0, max=len(dates)-1,
            value=[0, len(dates)-1],
            marks=dates, step=None, allowCross=False
        )
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(
            html.Button(
                children=html.Span(id={"type": "i18n", "key": "home.back_to_treemap"}),
                id="back-button",
                n_clicks=0,
                style={"display": "none"}  # start hidden
            ),
            width=12,
            className="mb-2"
        )
    ]),
    dbc.Row(dbc.Col(
        dcc.Graph(id="main-graph"),
        width=12
    )),
    # Sunburst and table beneath main graph
    dbc.Row([
        dbc.Col(
            dcc.Graph(id="sunburst-graph"),
            width=8
        ),
        dbc.Col(
            dash_table.DataTable(
                id='sunburst-table',
                columns=[{'name': "", 'id': 'label'}, {'name': "", 'id': 'value'}],
                data=[],
                style_table={'overflowY': 'auto', 'maxHeight': '400px'},
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'fontWeight': 'bold'}
            ),
            width=4
        )
    ], className="mt-4"),
    # hidden stores
    slider_store,
    mode_store
], fluid=True)

# Callback 1: update main-graph (treemap -> line)
@callback(
    Output("main-graph", "figure"),
    Output("slider_store", "data"),
    Output("mode_store",   "data"),
    Output("back-button",  "style"),
    Input("year-range-slider", "value"),
    Input("main-graph",       "clickData"),
    Input("back-button",      "n_clicks"),
    State("slider_store",     "data"),
    State("mode_store",       "data"),
)
def update_main_graph(year_range, clickData, back_clicks, slider_data, mode_data):
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]
    start_idx, end_idx = year_range
    start_year = dates[start_idx]
    end_year   = dates[end_idx]
    # Slider moved: treemap
    if triggered == "year-range-slider":
        fig = fn.create_treemap_figure(start_year, end_year)
        return fig, {'year_range': (start_year, end_year)}, {'mode': 'treemap','item':None}, {"display":"none"}
    # Back button: treemap
    if triggered == "back-button":
        fig = fn.create_treemap_figure(start_year, end_year)
        return fig, slider_data, {'mode':'treemap','item':None}, {"display":"none"}
    # Click on main-graph
    if triggered == "main-graph" and clickData:
        mode = mode_data['mode']
        if mode == 'treemap':
            item = clickData['points'][0]['label']
            if not item.startswith("All"):
                fig = fn.create_demand_over_time_figure(item, start_year, end_year)

                return fig, slider_data, {'mode':'line','item':item}, {"display":"inline-block"}
        if mode == 'line':
            # preserve the current line chart and state
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    # initial
    fig = fn.create_treemap_figure(start_year, end_year)
    return fig, {'year_range':(start_year,end_year)}, {'mode':'treemap','item':None}, {"display":"none"}

# Callback 2: update sunburst_graph & table when clicking in line chart or on back-button
@callback(
    Output('sunburst-graph','figure'),
    Output('sunburst-table','data'),
    Input('main-graph','clickData'),
    Input('back-button','n_clicks'),
    State('mode_store','data'),
    State('slider_store','data')
)
def update_sunburst(clickData, back_clicks, mode_data, slider_data):
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"].split(".")[0]

    # If back button clicked, clear sunburst
    if triggered == 'back-button':
        title = fn.outside_translate("functions.empty_graph")
        empty_fig = go.Figure().update_layout(title_text=title)
        return empty_fig, []

    # Only update when a date is clicked in line mode
    if triggered == 'main-graph' and mode_data.get('mode') == 'line' and clickData:
        item = mode_data['item']
        date_clicked = clickData['points'][0]['x']
        start_year,end_year = slider_data['year_range']
        fig = fn.create_sunburst_figure(item, date_clicked, start_year, end_year)
        trace = fig.data[0]
        rows = []
        for _id, par, val in zip(trace.ids, trace.parents, trace.values):
            if par == item:
                after = _id.split('/', 1)[-1]
                rows.append({'label': after, 'value': val})
        return fig, rows

    # Default clear
    title = fn.outside_translate("functions.empty_graph")
    empty_fig = go.Figure().update_layout(title_text=title)
    return empty_fig, []

@callback(
    Output('sunburst-table', 'columns'),
    Input('language-store', 'data'),
    prevent_initial_call=False
)

def translate_sunburst_table_columns(language):
    """
    Whenever the language-store changes, update the table headers
    for the sunburst table on the home page.
    """
    return [
        {'name': fn.outside_translate("functions.category"),   'id': 'label'},
        {'name': fn.outside_translate("functions.line_percentage"), 'id': 'value'}
    ]