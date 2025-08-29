import dash
from dash import html, dcc, callback, Output, Input, State # Import State for slider value
import dash_bootstrap_components as dbc
from . import functions as fn
from .import dash_functions as dfn
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Any, Tuple, Optional

# Register the page
dash.register_page(__name__, path="/product_usage")
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load product list
try:
    products = fn.get_product_list()
    dropdown_options: List[Dict[str, str]] = [{'label': source_no, 'value': source_no} for source_no in products]
except Exception as e:
    logging.error(f"Failed to load product list: {e}", exc_info=True)
    products = []
    dropdown_options = []

# Define the layout
layout = dbc.Container([
    # Hidden Store for table data and Download component
    dcc.Store(id='table-pivot-store'),
    dcc.Download(id="download-table-data"),

    dbc.Row([
        dbc.Col([
            html.H2(id={"type": "i18n", "key": "product_usage.title"}, className="text-center text-success mb-4"),
            html.H3(id={"type": "i18n", "key": "product_usage.subtitle"}, className="text-center text-secondary"),
        ])
    ]),
    dbc.Row([
        # Column for Product Dropdowns (md=6)
        dbc.Col(md=6, children=[
            dbc.Card([
                dbc.CardBody([
                    dcc.Dropdown(
                        id='source-no-dropdown',
                        options=dropdown_options,
                        placeholder="",
                        className="mb-3"
                    ),
                    html.Div(id='selected-source-no-output1', className="text-primary fw-bold"),
                    dcc.Dropdown(
                        id='source-no-dropdown2',
                        options=dropdown_options,
                        placeholder="",
                        className="mb-3"
                    ),
                    html.Div(id='selected-source-no-output2', className="text-primary fw-bold"),
                ])
            ])
        ]),
        # Column for Chart Type (md=3)
        dbc.Col(md=3, children=[
            dbc.Card([
                dbc.CardBody([
                    html.H5(id={"type": "i18n", "key": "product_usage.chart_type"}, className="card-title"),
                    dcc.RadioItems(
                        id='chart-type-selector',
                        options=[],
                        value='bar',
                        labelStyle={'display': 'block', 'marginBottom': '5px'},
                        inputStyle={'marginRight': '5px'}
                    )
                ])
            ])
        ]),
        # Column for Occurrence Threshold Slider (md=3)
        dbc.Col(md=3, children=[
            dbc.Card([
                dbc.CardBody([
                    html.H5(id={"type": "i18n", "key": "product_usage.occurrence_threshold"}, className="card-title"),
                    dcc.Slider(
                        id='threshold-slider',
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        value=0.8,  # Default threshold
                        marks={
                            0.0: '0%', 0.2: '20%', 0.4: '40%',
                            0.6: '60%', 0.8: '80%', 1.0: '100%'
                        },
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Div(id='threshold-output', className="text-muted text-center mt-2")
                ])
            ])
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='graph-output',
                config={
                    'displayModeBar': True, # This enables the toolbar with export options
                    'toImageButtonOptions': {
                        'format': 'png', # Default format for image download
                        'filename': 'product_configuration_chart',
                        'height': 600,
                        'width': 1000,
                        'scale': 2 # Increase resolution for better quality
                    },
                    'displaylogo': False # Hide the Plotly logo
                }
            )
        ], width=12)
    ], className="mt-4"),
    dbc.Row([
        dbc.Col([
            html.H3(id={"type": "i18n", "key": "product_usage.config_details"}, className="text-center text-secondary mt-4"),
            html.Div(id='config-table-container', children=[
                html.Div(id='config-table')
            ]),
            # Button to trigger table export
            dbc.Button("", id="export-table-button", color="primary", className="mt-3 mb-3 text-center")
        ], width=12)
    ], className="mt-4 mb-4")
])

# Callback to display current threshold value
@callback(
    Output('threshold-output', 'children'),
    Input('threshold-slider', 'value')
)
def display_threshold_value(value):
    return f"{fn.outside_translate("product_usage.selected_threshold")}: {value * 100:.0f}%"

@callback(
Output('source-no-dropdown', 'placeholder'),
    Output('source-no-dropdown2', 'placeholder'),
    Output("chart-type-selector", "options"),
    Output("export-table-button", "children"),
    Input('language-store', 'data')
)

def update_dropdown_placeholders(language):
    first = fn.outside_translate("product_usage.dropdown_placeholder")
    second = fn.outside_translate("product_usage.dropdown_placeholder_optional")
    chart_type_options = [
        {'label': fn.outside_translate("product_usage.bar_chart"), 'value': 'bar'},
        {'label': fn.outside_translate("product_usage.combined_radar_chart"), 'value': 'radar_combined'},
        {'label': fn.outside_translate("product_usage.separate_radar_charts"), 'value': 'radar_separate'}
    ]
    table_button = fn.outside_translate("product_usage.export_table_button")
    return first, second, chart_type_options, table_button

# When selecting products in the dropdown update graphs
@callback(
    Output('graph-output', 'figure'),
    Output('config-table', 'children'),
    Output('table-pivot-store', 'data'), # Store the data for export
    Input('source-no-dropdown', 'value'),
    Input('source-no-dropdown2', 'value'),
    Input('chart-type-selector', 'value'),
    Input('threshold-slider', 'value'),
    prevent_initial_call=True
)
def update_visualizations(
        source_no1: Optional[str],
        source_no2: Optional[str],
        chart_type: str,
        threshold: float
) -> Tuple[go.Figure, Any, Optional[List[Dict]]]:
    """Updates the graph, table, and stores table data based on selections made in the dropdown"""
    selected_products: List[str] = [p for p in [source_no1, source_no2] if p is not None]
    title = fn.outside_translate("product_usage.product_selection")
    empty_fig = go.Figure(layout={"title": title})
    empty_table_html_content: Optional[dbc.Table | dbc.Alert] = None
    empty_store_data: Optional[List[Dict]] = None

    if not selected_products:
        return empty_fig, empty_table_html_content, empty_store_data

    posten_df: pd.DataFrame = fn.load_posten_file()

    if posten_df.empty:
        no_data_avail = fn.outside_translate("product_usage.no_data_available")
        dbc_alert = fn.outside_translate("product_usage.posten_data_missing")
        return go.Figure(layout={"title": no_data_avail}), dbc.Alert(dbc_alert, color="warning"), empty_store_data

    plot_data_list: List[Dict[str, Any]] = []
    info_messages: List[str] = []

    for product_id in selected_products:
        try:
            _, standard_config, _ = fn.create_configuration_with_stats(posten_df, product_id, threshold=threshold)
            if standard_config:
                plot_data_list.extend([{fn.ARTICLE_NUMBER_COLUMN: str(k), 'RequiredQuantity': v, 'Product': product_id} for k, v in standard_config.items()])
            else:
                logging_info = fn.outside_translate("product_usage.no_standard_config_found")
                title = logging_info.format(
                    product_id=product_id,
                    threshold=threshold,
                )
                logging.info(title)
                tmpl = fn.outside_translate("product_usage.no_standard_config")
                msg = tmpl.format(
                    product=product_id,
                    threshold=threshold * 100
                )
                info_messages.append(msg)
        except Exception as e:
            logging_info = fn.outside_translate("product_usage.no_standard_config_not_found")
            title = logging_info.format(
                product_id=product_id,
                threshold=threshold,
                e = e
            )
            logging.error(title, exc_info=True)
            info_messages.append(f"Error loading configuration for product '{product_id}'. Details: {e}")

    if not plot_data_list:
        alert_content = dbc.Alert(html.Ul([html.Li(msg) for msg in info_messages]), color="info") if info_messages \
                        else dbc.Alert("No config data found for the selected threshold.", color="info")
        return empty_fig, alert_content, empty_store_data

    config_df = pd.DataFrame(plot_data_list)
    chart_title = fn.outside_translate("product_usage.chart_title")
    title = chart_title.format(
        selected_products=', '.join(selected_products),
        threshold=threshold * 100
    )
    chart_title: str = title
    chart_figure: go.Figure

    if chart_type == 'bar':
        chart_figure = dfn.create_bar_chart_figure(config_df, chart_title)
    elif chart_type == 'radar_combined':
        chart_figure = dfn.create_combined_radar_chart(config_df, selected_products)
    elif chart_type == 'radar_separate':
        chart_figure = dfn.create_separate_radar_charts(config_df, selected_products)
    else:
        title_text = fn.outside_translate("product_usage.invalid_chart_type")
        chart_figure = go.Figure(layout={"title": title_text})

    table_components = []
    if info_messages:
        table_components.append(dbc.Alert(html.Ul([html.Li(msg) for msg in info_messages]), color="info", className="mb-3"))

    table_df_pivot = pd.DataFrame() # Initialize to ensure it's defined
    try:
        table_df_pivot = config_df.pivot(index=fn.ARTICLE_NUMBER_COLUMN, columns='Product', values='RequiredQuantity').fillna(0).reset_index()
        table_components.append(dfn.create_table_from_dataframe(table_df_pivot))
    except Exception as ee:
        logging.error(f"Error creating table: {ee}", exc_info=True)
        error_msg = fn.outside_translate("product_usage.error_generating_table")
        table_components.append(dbc.Alert(error_msg, color="danger"))

    # Store the pivoted DataFrame as a list of dictionaries for export
    # Using to_dict('records') is generally safer and more portable for dcc.Store
    stored_data = table_df_pivot.to_dict('records') if not table_df_pivot.empty else None
    return chart_figure, html.Div(table_components), stored_data

# Callback to download table data
@callback(
    Output("download-table-data", "data"),
    Input("export-table-button", "n_clicks"),
    State("table-pivot-store", "data"),
    prevent_initial_call=True,
)
def export_table_csv(n_clicks, stored_data: Optional[List[Dict]]):
    if n_clicks and stored_data:
        # Reconstruct DataFrame from stored list of dictionaries
        df_to_export = pd.DataFrame.from_records(stored_data)
        if not df_to_export.empty:
            return dcc.send_data_frame(df_to_export.to_csv, "product_configuration_details.csv", index=False)
    return None
