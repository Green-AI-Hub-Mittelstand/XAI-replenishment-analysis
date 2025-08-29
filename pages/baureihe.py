import logging
import dash
from dash import callback, Input, Output, State, dash_table
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
from collections import Counter
from typing import Dict, List, Tuple, Union, Optional, Any
from . import functions as fn
from . import dash_functions as dfn

# Suppress warnings if needed
warnings.filterwarnings('ignore')

# Register the page with Dash
dash.register_page(__name__, path="/baureihe")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load initial data for dropdowns
br_data_dict = fn.create_br_dict()
baureihe_options = list(br_data_dict.keys())

# Layout Definition
layout = dbc.Container([
    dcc.Store(id='br-table-store'),
    dcc.Download(id="br-download-table"),
    dcc.Store(id='br-summary-table-store'),
    dcc.Download(id="br-download-summary-table"),

    dbc.Row(dbc.Col(html.H2(id={"type": "i18n", "key": "series.title"}, className="text-center text-success mb-4"))),

    dbc.Row([
        dbc.Col(md=6, children=[dbc.Card(dbc.CardBody([html.H5(id={"type":"i18n","key":"series.select_serie"}),
                                                       dcc.Dropdown(id='br-baureihe-dropdown', options=baureihe_options,
                                                                    placeholder="")]))]),
        dbc.Col(md=6, children=[dbc.Card(dbc.CardBody([html.H5(id={"type":"i18n","key":"series.select_product_optional"}),
                                                       dcc.Dropdown(id='br-product-dropdown',
                                                                    placeholder="")]))]),
    ], className="mb-4"),

    html.Div(id='br-summary-view', children=[
        dbc.Row(dbc.Col(html.H3("", id='br-summary-title',
                                className="text-center text-primary mb-3"))),

        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(
                                dbc.Switch( # Switch for aggregated summary view
                                    id='br-summary-view-toggle',
                                    label="",
                                    value=False,
                                ),
                                className="d-flex justify-content-center align-items-center"
                            )
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col(md=6, children=[
                                html.H6(id={"type":"i18n","key":"series.filter_usage"},
                                        className="text-center small text-muted"),
                                dcc.RangeSlider(
                                    id='br-baureihe-usage-filter-slider',
                                    min=0, max=100, step=1, value=[0, 100],
                                    marks={i: f'{i}%' for i in range(0, 101, 20)},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                    allowCross=False
                                )
                            ]),
                            dbc.Col(md=6, children=[
                                html.H6(id={"type":"i18n","key":"series.filter_probability"}, className="text-center small text-muted"),
                                dcc.RangeSlider(
                                    id='br-menge-frequency-filter-slider',
                                    min=0, max=100, step=1, value=[0, 100],
                                    marks={i: f'{i}%' for i in range(0, 101, 20)},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                    allowCross=False
                                )
                            ]),
                        ])
                    ])
                ),
                width=12
            )
        ], className="mb-3"),

        dbc.Row(dbc.Col(html.Div(id='br-summary-table'), width=12)),
        dbc.Row(
            dbc.Col(
                dbc.Button(id="br-export-summary-button", color="info", className="mt-3"),
                width="auto"
            ),
            justify="center",
            className="mb-3"
        ),

        #Row for the Barchart
        dbc.Row(
            dbc.Col(
                dcc.Graph(id='br-usage-barchart'), # Add the graph component
                width=12
            ),
            className="mt-5 mb-4"
        )

    ]),

    html.Div(id='br-product-view', style={'display': 'none'}, children=[
        html.Hr(),
        dbc.Row(dbc.Col(html.H3(id={"type":"i18n","key":"series.product_details_title"}, className="text-center text-primary mb-3"))),
        dbc.Row([
            dbc.Col(md=6, children=[
                dbc.Card(dbc.CardBody([
                    html.H5(id={"type":"i18n","key":"series.chart_type_label"}),
                    dcc.RadioItems(id='br-chart-type-selector', options=[{'label': '', 'value': 'bar'},
                                                                         {'label': '', 'value': 'radar'}],
                                   value='bar', labelStyle={'display': 'block'})
                ]))
            ]),
            dbc.Col(md=6, children=[
                dbc.Card(dbc.CardBody([
                    html.H5(id={"type":"i18n","key":"product_usage.occurrence_threshold"}),
                    dcc.Slider(id='br-threshold-slider', min=0.0, max=1.0, step=0.05, value=0.8,
                               marks={i / 10: f'{i * 10:.0f}%' for i in range(0, 11, 2)},
                               tooltip={"placement": "bottom", "always_visible": True})
                ]))
            ])
        ]),
        dbc.Row(dbc.Col(dcc.Graph(id='br-graph-output'), width=12), className="mt-4"),
        dbc.Row(dbc.Col([
            html.Div(id='br-config-table'),
            html.Div(
                dbc.Button(id="br-export-button", color="primary", className="mt-3"),
                className="text-center"
            )
        ], width=12), className="mt-4 mb-4")
    ]),

], fluid=True)

@callback(
    Output('br-baureihe-dropdown', 'placeholder'),
    Output('br-product-dropdown',   'placeholder'),
    Output('br-chart-type-selector', 'options'),
    Output('br-export-summary-button','children'),
    Output('br-export-button',        'children'),
    Output('br-summary-view-toggle', 'label'),
    Input('language-store', 'data'),
)
def update_series_i18n(language):
    # Dropdown placeholders
    ph1 = fn.outside_translate("series.dropdown_baureihe_placeholder")
    ph2 = fn.outside_translate("series.dropdown_product_placeholder")
    # Radio options
    opts = [
        {'label': fn.outside_translate("series.option_bar_chart"),       'value': 'bar'},
        {'label': fn.outside_translate("series.option_combined_radar"),  'value': 'radar'},
    ]
    # Button texts
    btn1 = fn.outside_translate("series.export_summary_csv")
    btn2 = fn.outside_translate("series.export_table_csv")
    # Switch label
    switch_label = fn.outside_translate("series.aggregated_summary_switch_label")

    return ph1, ph2, opts, btn1, btn2, switch_label


# Update Dropdowns upon selected baureihe
@callback(
    Output('br-product-dropdown', 'options'),
    Output('br-product-dropdown', 'value'),
    Input('br-baureihe-dropdown', 'value')
)
def update_product_dropdowns(selected_baureihe: Optional[str]) -> Tuple[List[Dict], None]:
    if not selected_baureihe: return [], None
    br_dict = fn.create_br_dict()
    product_df = br_dict.get(selected_baureihe)
    if product_df is None or product_df.empty: return [], None
    product_options = [{'label': p, 'value': p} for p in product_df["Nr."].tolist()]
    return product_options, None

# Update Main Views depending on inputs when selected baureihe
@callback(
    Output('br-summary-view', 'style'),
    Output('br-product-view', 'style'),
    Output('br-summary-table', 'children'),
    Output('br-summary-title', 'children'),
    Output('br-usage-barchart', 'figure'),
    Output('br-graph-output', 'figure'),
    Output('br-config-table', 'children'),
    Output('br-table-store', 'data'),
    Output('br-summary-table-store', 'data'),
    Input('br-baureihe-dropdown', 'value'),
    Input('br-product-dropdown', 'value'),
    Input('br-chart-type-selector', 'value'),
    Input('br-threshold-slider', 'value'),
    Input('br-summary-view-toggle', 'value'),
    Input('br-baureihe-usage-filter-slider', 'value'),
    Input('br-menge-frequency-filter-slider', 'value')
)
def update_main_views(
        selected_br: Optional[str],
        selected_p1: Optional[str],
        chart_type: str,
        threshold: float,
        show_aggregated_summary: bool,
        baureihe_usage_range: List[float],
        menge_frequency_range: List[float]
) -> Tuple[Dict, Dict, Any, str, go.Figure, go.Figure, Any, Optional[List[Dict]], Optional[List[Dict]]]:
    # define beginning states for views
    style_hidden = {'display': 'none'}
    style_visible = {'display': 'block'}
    empty_fig = go.Figure(layout={"height": 100}) # Give empty figs a small height
    empty_bar_fig = go.Figure(layout={"title": fn.outside_translate("series.select_baureihe_usage"), "height": 100})
    default_summary_title = fn.outside_translate("series.default_summary_title")
    empty_product_config_store = None
    empty_summary_store = None

    if not selected_br:
        return style_hidden, style_hidden, None, default_summary_title, empty_bar_fig, empty_fig, None, empty_product_config_store, empty_summary_store

    if selected_br and not selected_p1:
        detailed_summary_df = fn.generate_component_quantity_probabilities2(
            selected_br,
            threshold=threshold,
            baureihe_usage_filter=baureihe_usage_range,
            menge_frequency_filter=menge_frequency_range
        )

        current_summary_df = pd.DataFrame()
        summary_table_children: Any
        summary_store_data: Optional[List[Dict]] = None
        summary_title = default_summary_title
        usage_bar_fig = go.Figure(layout={"height": 100}) # Default empty

        # Define column names
        col_produkt = fn.outside_translate("series.col_produkt")
        col_komponente = fn.outside_translate("series.col_komponente")
        col_baureihe_wide_usage_numeric = fn.outside_translate("series.col_baureihe_wide_usage_numeric")
        col_prob_menge_numeric = fn.outside_translate("series.col_prob_menge_numeric")
        col_prod_koeff_numeric = fn.outside_translate("series.col_prod_koeff_numeric")
        col_weighted_avg_prod_koeff_numeric = fn.outside_translate("series.col_weighted_avg_prod_koeff_numeric")
        display_col_baureihe_wide_usage = fn.outside_translate("series.display_col_baureihe_wide_usage")
        display_col_prob_menge = fn.outside_translate("series.display_col_prob_menge")
        if detailed_summary_df.empty:
            summary_table_children = dbc.Alert(fn.outside_translate("series.no_summary_data"),
                                               color="warning")
            usage_bar_fig = go.Figure(layout={"title": fn.outside_translate("series.no_usage_chart"), "height": 100})
        else:
            # Generate Barchart Figure
            try:
                # Get unique usage per component, sort, and take top N
                usage_data = detailed_summary_df[[col_komponente, col_baureihe_wide_usage_numeric]].drop_duplicates()
                usage_data = usage_data.sort_values(col_baureihe_wide_usage_numeric, ascending=True) # Top 25, ascending for px.bar

                if not usage_data.empty:
                    usage_bar_fig = px.bar(
                        usage_data,
                        y=col_komponente,
                        x=col_baureihe_wide_usage_numeric,
                        orientation='h',
                        title=fn.outside_translate("series.usage_within_baureihe").format(baureihe=selected_br),
                        labels={col_baureihe_wide_usage_numeric: fn.outside_translate("series.label_usage_percent"), col_komponente: fn.outside_translate("series.label_component")},
                        height=max(600, len(usage_data) * 25) # Dynamic height
                    )
                    usage_bar_fig.update_layout(
                        yaxis_title=None,
                        margin=dict(l=150) # Add left margin if labels are long
                        )
                else:
                    usage_bar_fig = go.Figure(layout={"title": fn.outside_translate("series.no_usage_data"), "height": 100})
            except Exception as e:
                logging.error(f"Error creating usage bar chart: {e}")
                usage_bar_fig = go.Figure(layout={"title": fn.outside_translate("series.error_usage_chart").format(error=str(e)), "height": 100})

            # Generate Summary Table
            if show_aggregated_summary:
                summary_title = fn.outside_translate("series.aggregated_summary_title")
                idx_most_frequent_menge = detailed_summary_df.groupby([col_produkt, col_komponente])[col_prob_menge_numeric].idxmax()
                aggregated_df_most_frequent = detailed_summary_df.loc[idx_most_frequent_menge].reset_index(drop=True)
                current_summary_df = aggregated_df_most_frequent[[
                    col_komponente, col_produkt,
                    display_col_baureihe_wide_usage,
                    col_prod_koeff_numeric,
                    display_col_prob_menge,
                    col_weighted_avg_prod_koeff_numeric
                ]].rename(columns={
                    col_prod_koeff_numeric: fn.outside_translate("series.col_prod_koeff_numeric_most_used"),
                    display_col_prob_menge: fn.outside_translate("series.display_col_prob_menge_most_freq"),
                    display_col_baureihe_wide_usage: fn.outside_translate("series.display_col_baureihe_wide_usage_short"),
                    col_weighted_avg_prod_koeff_numeric: fn.outside_translate("series.col_weighted_avg_prod_koeff_numeric_short")
                })
            else:
                summary_title = fn.outside_translate("series.detailed_summary_title")
                current_summary_df = detailed_summary_df[[
                    col_komponente, col_produkt,
                    display_col_baureihe_wide_usage,
                    col_prod_koeff_numeric,
                    display_col_prob_menge,
                    col_weighted_avg_prod_koeff_numeric
                ]].rename(columns={
                    col_prod_koeff_numeric: fn.outside_translate("series.col_prod_koeff_numeric"),
                    display_col_prob_menge: fn.outside_translate("series.display_col_prob_menge_2"),
                    display_col_baureihe_wide_usage: fn.outside_translate("series.display_col_baureihe_wide_usage_short_2"),
                    col_weighted_avg_prod_koeff_numeric: fn.outside_translate("series.col_weighted_avg_prod_koeff_numeric_short")
                })

            if current_summary_df.empty:
                summary_table_children = dbc.Alert(fn.outside_translate("series.no_summary_data"), color="info")
            else:
                summary_df_styled = current_summary_df.sort_values(by=[col_produkt, col_komponente]).reset_index(drop=True)
                current_product_for_color = None
                color_group = 0
                groups = []
                for prod_val in summary_df_styled[col_produkt]:
                    if prod_val != current_product_for_color:
                        color_group = 1 - color_group
                        current_product_for_color = prod_val
                    groups.append(color_group)
                summary_df_styled['row_color_group'] = groups
                styles = [
                    {'if': {'filter_query': '{row_color_group} = 1'}, 'backgroundColor': '#f2f2f2'},
                    {'if': {'filter_query': '{row_color_group} = 0'}, 'backgroundColor': 'white'},
                ]
                table_columns = [{"name": i, "id": i} for i in current_summary_df.columns if i != 'row_color_group']
                summary_table_children = dash_table.DataTable(
                    columns=table_columns, data=summary_df_styled.to_dict('records'),
                    page_size=25, style_cell={'textAlign': 'left', 'padding': '5px',
                                              'fontFamily': '"Helvetica Neue", Helvetica, Arial, sans-serif'},
                    style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                    filter_action="native", sort_action="native",
                    style_table={'overflowX': 'auto', 'minWidth': '100%'},
                    style_data_conditional=styles
                )
            summary_store_data = current_summary_df.to_dict('records')

        return style_visible, style_hidden, summary_table_children, summary_title, usage_bar_fig, empty_fig, None, empty_product_config_store, summary_store_data

    elif selected_p1: # Product view
        # DONT need the barchart return an empty one
        empty_bar_fig = go.Figure(layout={"height": 100})
        selected_products = [selected_p1]
        posten_df = fn.load_posten_file()
        if posten_df.empty:
            return style_hidden, style_visible, None, default_summary_title, empty_bar_fig, go.Figure(
                layout={"title": fn.outside_translate("series.posten_data_missing")}), dbc.Alert(fn.outside_translate("series.posten_data_missing"),
                                                                     color="danger"), empty_product_config_store, empty_summary_store
        plot_data_list = []
        try:
            _, std_cfg, _ = fn.create_configuration_with_stats(posten_df, selected_p1, threshold)
            plot_data_list.extend(
                [{fn.ARTICLE_NUMBER_COLUMN: str(k), 'RequiredQuantity': v, 'Product': selected_p1} for k, v in std_cfg.items()])
        except Exception as e:
            logging.error(f"Error for {selected_p1}: {e}", exc_info=True)
        if not plot_data_list:
            return style_hidden, style_visible, None, default_summary_title, empty_bar_fig, go.Figure(
                layout={"title": fn.outside_translate("series.no_config_data_found")}), dbc.Alert(fn.outside_translate("series.no_config_data"),
                                                                       color="info"), empty_product_config_store, empty_summary_store
        config_df = pd.DataFrame(plot_data_list)
        chart_title = fn.outside_translate("series.standard_config_title").format(
            product=selected_p1, threshold=threshold * 100
        )
        if chart_type == 'bar':
            chart_figure = dfn.create_bar_chart_figure(config_df, chart_title)
        elif chart_type == 'radar':
            chart_figure = dfn.create_combined_radar_chart(config_df, selected_products)
        else:
            chart_figure = go.Figure(layout={"title": "Invalid chart type."})
        details_table_children = None
        table_df_pivot = pd.DataFrame()
        try:
            table_df_pivot = config_df.pivot(index=fn.ARTICLE_NUMBER_COLUMN, columns='Product', values='RequiredQuantity').fillna(
                0).reset_index()
            table_df_pivot.rename_axis(None, axis=1, inplace=True)
            details_table_children = dfn.create_table_from_dataframe(table_df_pivot)
        except Exception as e:
            logging.error(f"Error creating details table: {e}", exc_info=True)
            details_table_children = dbc.Alert(fn.outside_translate("series.error_generating_details_table"), color="danger")
        product_config_store_data = table_df_pivot.to_dict('records') if not table_df_pivot.empty else None
        return style_hidden, style_visible, None, default_summary_title, empty_bar_fig, chart_figure, details_table_children, product_config_store_data, empty_summary_store

    return style_hidden, style_hidden, None, default_summary_title, empty_bar_fig, empty_fig, None, empty_product_config_store, empty_summary_store


# Export Callbacks
@callback(
    Output("br-download-table", "data"),
    Input("br-export-button", "n_clicks"),
    State("br-table-store", "data"),
    prevent_initial_call=True,
)
def export_br_table_csv(n_clicks: Optional[int], stored_data: Optional[List[Dict]]):
    if n_clicks and stored_data:
        df_to_export = pd.DataFrame.from_records(stored_data)
        if not df_to_export.empty:
            filename = "product_config_details.csv"
            if len(df_to_export.columns) == 2 and fn.ARTICLE_NUMBER_COLUMN in df_to_export.columns:
                product_col_name = [col for col in df_to_export.columns if col != fn.ARTICLE_NUMBER_COLUMN][0]
                filename = f"config_{product_col_name}.csv"
            return dcc.send_data_frame(df_to_export.to_csv, filename, index=False)
    return None


@callback(
    Output("br-download-summary-table", "data"),
    Input("br-export-summary-button", "n_clicks"),
    State("br-summary-table-store", "data"),
    State("br-baureihe-dropdown", "value"),
    State('br-summary-view-toggle', 'value'),
    prevent_initial_call=True,
)
def export_br_summary_table_csv(n_clicks: Optional[int], stored_data: Optional[List[Dict]],
                                selected_baureihe: Optional[str], aggregated_view: bool):
    if n_clicks and stored_data:
        df_to_export = pd.DataFrame.from_records(stored_data)
        if not df_to_export.empty:
            view_type = "aggregated_summary" if aggregated_view else "detailed_summary"
            filename = f"baureihe_{view_type}.csv"
            if selected_baureihe:
                sane_baureihe_key = "".join(c if c.isalnum() else "_" for c in selected_baureihe)
                filename = f"baureihe_{sane_baureihe_key}_{view_type}.csv"
            return dcc.send_data_frame(df_to_export.to_csv, filename, index=False)
    return None