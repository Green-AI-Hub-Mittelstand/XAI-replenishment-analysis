import math
from datetime import datetime
import dash
from dash import html, dcc, callback, Output, Input,State
import dash.exceptions
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from pandas import DataFrame
import logging
import pulp
import re
from . import functions as fn
from . import dash_functions as dfn
from typing import Dict, List, Tuple, Union, Optional, Any
from urllib.parse import parse_qs,urlencode # For robust URL query parsing and encoding
# Dash Page for Replenishment Simulation
dash.register_page(__name__, path="/simulation", title="Replenishment Simulation")
dropdown_options = fn.get_article_dropdown_options()
layout = dbc.Container([
    dcc.Location(id='simulation-url', refresh=False), # URL to simulation from planning
    dcc.Store(id='planned-demand-from-url-store'),
    dbc.Row([
            dbc.Col([
                html.H2(id={"type": "i18n", "key": "simulation.replenishment_simulation_title"}, className="text-center text-success mb-4")
            ])
    ]),
    dbc.Row([
        dbc.Col([

            dcc.Dropdown(
                id='end-product-dropdown',
                options=dropdown_options,
                placeholder="",
                className="mb-3"
            ),
            html.Div(id='selected-product-output3', className="text-primary fw-bold"),

        ], width=6),

    ], className="mt-3 justify-content-center"),

    # Placeholders which will be filled after Callbacks
    dbc.Row([
        dbc.Col([ # Forecasting Graph Visualization
            dcc.Graph(id='forecast-graph')
        ], width=12)
    ], className="mt-4"),
    dbc.Row([ # Guidelines section from the data
        dbc.Col(html.Div(id='guideline-text'), width=12)
    ], className="mt-4"),
    dbc.Row([ # Prediction Values normal, minimum, maximum
        dbc.Col(html.Div(id='pred-value-table'), width=12)
    ], className="mt-4"),
    dbc.Row([
            dbc.Col(html.Div(id='plan-vs-forecast-comparison-table'), width=12)
        ], className="mt-4"),
    dbc.Row([ # Lead time calculation
        dbc.Col(html.Div(id='leadtime-text'), width=12)
    ], className="mt-4"),
    dbc.Row([
        dbc.Col(html.Div(id='reorder-table'), width=12)
    ], className="mt-4"),

    # Replenishment plans stored in here
    dcc.Store(id="normal_replenishment"),
    dcc.Store(id="minimum_replenishment"),
    dcc.Store(id="maximum_replenishment"),
    dcc.Store(id="planned_replenishment"),
    dcc.Store(id="average_replenishment"),

    # dbc.Row([
    #     dbc.Col(html.Div(id='scenario-buttons-container'), width=12)
    # ], className="mt-3 justify-content-center"),
    dbc.Row([ # Scenario Choosing buttons
        dbc.Col(html.Div([
            dbc.Row(
                dbc.Col(html.H5(id={"type": "i18n", "key": "simulation.choose_scenario_label"}, className="mt-3 text-center")),
                justify="center"
            ),
            dbc.Row([
                dbc.Col(dbc.Button("", id="min-btn", color="secondary", className="me-2", n_clicks=0), width="auto"),
                dbc.Col(dbc.Button("", id="normal-btn", color="success", className="me-2", n_clicks=0), width="auto"),
                dbc.Col(dbc.Button("", id="max-btn", color="secondary", n_clicks=0), width="auto"),
                dbc.Col(dbc.Button("", id="planned-btn", color="danger", className="me-2", n_clicks=0), width="auto"),
                dbc.Col(dbc.Button("", id="avg-btn", color="warning", n_clicks=0), width="auto"),
            ], justify="center", className="mt-2")
        ], id='scenario-buttons-container'), width=12)
    ], className="mt-3 justify-content-center"),

    dbc.Row([ # Replenishment table output section
        dbc.Col(html.Div(id='scenario-recommendation-output'), width=12)
    ], className="mt-3 justify-content-center"),

], fluid=True)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
@callback(
    Output('end-product-dropdown', 'placeholder'),
    Output('min-btn', 'children'),
    Output('normal-btn', 'children'),
    Output('max-btn', 'children'),
    Output('planned-btn', 'children'),
    Output('avg-btn', 'children'),
    Input('language-store', 'data')
)

def update_dropdown_placeholders(language):
    first = fn.outside_translate("simulation.dropdown_placeholder")
    min_btn = fn.outside_translate("simulation.min_demand")
    normal_btn = fn.outside_translate("simulation.normal_demand")
    max_btn = fn.outside_translate("simulation.max_demand")
    planned_btn = fn.outside_translate("simulation.planned_demand")
    avg_btn = fn.outside_translate("simulation.avg_calc_demand")

    return first, min_btn, normal_btn, max_btn, planned_btn, avg_btn

# callback to store planned demand from URL post
@callback(
    Output('planned-demand-from-url-store', 'data'),
    Input('simulation-url', 'search'),
    prevent_initial_call=True
)
def store_planned_demand_from_url(search: Optional[str]):
    """
    Parses the URL for 'weeks' and 'demands' parameters,
    converts them to a plottable format, and stores them.
    """
    if not search:
        return None # No data in URL

    try:
        parsed_params = parse_qs(search.lstrip("?"))
        weeks_str = parsed_params.get('weeks', [None])[0]
        demands_str = parsed_params.get('demands', [None])[0]

        if weeks_str and demands_str:
            week_labels = weeks_str.split(',')
            demand_values = [int(d) for d in demands_str.split(',')]

            # Convert week strings (e.g., "2025-W23") to datetime objects for plotting
            dates = [datetime.strptime(f'{week}-0', "%Y-W%W-%w") for week in week_labels]

            # Store as a dictionary, which is easy to handle
            return {
                'dates': [d.strftime('%Y-%m-%d') for d in dates], # Store as strings
                'demands': demand_values
            }
    except (ValueError, TypeError, IndexError) as e:
        logging.error(f"Could not parse planned demand from URL: {search}. Error: {e}")
        return None # Return None on error

    return None # Return None if parameters are missing

# callback set dropdown value from Planning page
@callback(
    Output('end-product-dropdown', 'value'),
    Input('simulation-url', 'search'),
    State('de-anonymized-store', 'data'),  # Get the current global anonymization setting
    prevent_initial_call=True
)
def set_dropdown_value_from_url(search: Optional[str], anonymization_store_data: Optional[Dict]):
    if not search:
        logging.debug("Simulation URL search string is empty, no update to dropdown value.")
        return dash.no_update

    item_id_from_url = None
    try:
        parsed_search = parse_qs(search.lstrip("?"))  # Example: {'item': ['6-00001']}
        item_id_list = parsed_search.get('item')
        if item_id_list and item_id_list[0]:
            item_id_from_url = item_id_list[0]
    except Exception as e:
        logging.error(f"Error parsing URL search string '{search}': {e}")
        return dash.no_update

    if not item_id_from_url:
        logging.debug("No 'item' parameter found in simulation URL search string.")
        return dash.no_update

    logging.info(f"Attempting to set dropdown from URL item: {item_id_from_url}")

    # if anonymization_store_data is None:
    #     logging.warning("Anonymization store data is None, cannot determine correct options set.")
    #     return dash.no_update
    #
    # # Determine which set of options *should* be active based on global settings,
    # # mirroring the logic in app.py's swap_dropdown_simulation.
    # # The 'dynamic_drop_down' function's 'anonymization' param:
    # # True = use anonymized labels, False = use de-anonymized labels.
    # is_global_setting_de_anonymized = anonymization_store_data.get('de-anonymized', False)
    #
    # # If global setting is de-anonymized, call dynamic_drop_down with anonymization=False.
    # # If global setting is anonymized, call dynamic_drop_down with anonymization=True.
    # options_that_should_be_in_dropdown = fn.dynamic_drop_down("item", not is_global_setting_de_anonymized)
    #
    # # Check if the item_id_from_url exists as a 'value' in these expected options
    # if any(opt['value'] == item_id_from_url for opt in options_that_should_be_in_dropdown):
    #     logging.info(f"Item ID '{item_id_from_url}' found in expected options. Setting dropdown value.")
    return item_id_from_url
    # else:
    #     logging.warning(
    #         f"Item ID '{item_id_from_url}' from URL not found in expected dropdown options (generated based on current anonymization state). Dropdown not set.")
    #     return dash.no_update

# When selecting an item in the dropdown predictions. Preparing Replenishments for later scenario selection
@callback(
    [
        Output("forecast-graph", "figure"),
        Output("guideline-text", "children"),
        Output("pred-value-table", "children"),
        Output("plan-vs-forecast-comparison-table", "children"),
        Output("normal_replenishment", "data"),
        Output("minimum_replenishment", "data"),
        Output("maximum_replenishment", "data"),
        Output("planned_replenishment", "data"),
        Output("average_replenishment", "data"),
    ],
    [Input("end-product-dropdown", "value"),
     State('planned-demand-from-url-store', 'data')]
)
def update_forecast_graph(selected_item,planned_demand_data):
    """
    1. Fetches & processes data for selected item
    2. Makes forecast for that item
    3. Displays replenishment simulation
    4. Shows a reorder plan
    """
    logging.info(f"Selected item: {selected_item}.")
    # Landing page with only an empty graph and information to select an item
    if not selected_item:
        info = fn.outside_translate("simulation.replenishment_select_item_info")
        info_md = dcc.Markdown(info)
        return go.Figure(), info_md, "", "", "","","","",""

    # get time series data to forecast the future for 28 weeks
    X_train, future_dates = fn.get_data(selected_item)
    logging.info("Data fetched. Running Forecasting...")
    # make predictions for 28 weeks
    predictions, lower_bounds, upper_bounds, feature_list = fn.recursive_prediction_conformal_darts(
        selected_item,
        X_train,
        future_dates,
        quantile=[0.025, 0.5, 0.975],
        sample=5000,
        lag=15
    )
    logging.info("Forecasts Done. Processing Predictions...")

    # convert to prediction values to dataframe
    results_df = pd.DataFrame(predictions,
                              columns=[fn.PREDICTED_DATE_COLUMN,
                                       fn.PREDICTED_COLUMN])
    results_df[fn.LOWER_BOUND_COLUMN] = lower_bounds
    results_df[fn.UPPER_BOUND_COLUMN] = upper_bounds
    results_df = results_df.sort_values(by=fn.PREDICTED_DATE_COLUMN)  # Sort the DataFrame by date for proper plotting.
    last_day: pd.Timestamp = results_df[
        fn.PREDICTED_DATE_COLUMN].min()  # plot this vertical line to indicate a starting prediction
    # extract monthly normal, minimal and maximal demand
    lower_bounds_demand_dict, normal_demand_dict, upper_bounds_demand_dict = fn.aggregate_monthly_data(results_df)

    month_str = fn.outside_translate("functions.line_month")
    conservative_str = fn.outside_translate("simulation.min_forecast")
    standard_str = fn.outside_translate("simulation.exp_forecast")
    opt_str = fn.outside_translate("simulation.max_forecast")

    # aggregated to monthly demand for table
    monthly_demand = fn.aggregate_weekly_results(results_df)
    logging.info("Aggregated monthly demand data")
    translated_table = monthly_demand.rename(columns={'Month': month_str,
                                       'Minimal prognostizierter Vebrauch (Konservativ)': conservative_str,
                                       'Erwarteter prognostizierter (Standard)': standard_str}).copy()
    translated_table.rename(columns={'Maximal prognostizierter Verbrauch (Optimistisch)': opt_str}, inplace=True)
    monthly_demand_dash = dbc.Table.from_dataframe(
        translated_table, striped=True, bordered=True, hover=True)

    # get inventory info for the selected item
    inventory_data = fn.extract_inventory_data()
    inventory_info = fn.get_info(inventory_data, selected_item)
    safety_stock = inventory_info.safety_stock  # plot this horizontal line in the graph

    # extract current stock level (last date of the entry of the item)
    _, final_stock = fn.get_stock_data(selected_item)

    # fetch guidelines
    guidelines = fn.display_replenishment_guidelines(inventory_info.safety_stock, inventory_info.lead_time,inventory_info.min_order_qty,inventory_info.max_order_qty,final_stock)
    # turn string in Markdown for dashboard
    guideline_md = dcc.Markdown(guidelines)
    planned_replenishment_data = []
    average_replenishment_data = []

    logging.info(f"Running Inventory Simulation for {selected_item}...")
    # Run inventory simulations to store them in dedicated dcc.Store
    normal_demand = fn.run_inventory_simulation(normal_demand_dict,
                                                  "Normal Demand Quantity",inventory_info,final_stock)
    minimum_demand = fn.run_inventory_simulation(lower_bounds_demand_dict,
                                              "Minimum Demand Quantity", inventory_info,final_stock)
    maximum_demand = fn.run_inventory_simulation(upper_bounds_demand_dict,
                                              "Maximum Demand Quantity", inventory_info, final_stock)

    # save tables for later upon pressing buttons
    if normal_demand.empty or minimum_demand.empty or maximum_demand.empty:
        normal_demand_table = html.Div("No replenishment needed.")
        minimum_demand_table = html.Div("No replenishment needed.")
        maximum_demand_table = html.Div("No replenishment needed.")
    else:
        # Convert to serializable objects for dash
        normal_demand_table = normal_demand.to_dict('records')
        minimum_demand_table = minimum_demand.to_dict('records')
        maximum_demand_table = maximum_demand.to_dict('records')
        logging.info("Saved replenishment scenarios.")

    # Create plotly figure
    name = selected_item
    fig = go.Figure()
    # Add trace for historical demand
    historical_demand = X_train.tail(25)
    histo_demand = fn.outside_translate("simulation.historical_demand")
    pred_demand = fn.outside_translate("article.pred_demand")
    upper_bound = fn.outside_translate("article.upper_bound")
    lower_bound = fn.outside_translate("article.lower_bound")
    dfn.add_fig_trace(fig,historical_demand.index,historical_demand['Menge'],"lines+markers",histo_demand,dict(color='black'))
    # trace for predicted demand
    dfn.add_fig_trace(fig,x=results_df[fn.PREDICTED_DATE_COLUMN],y=results_df[fn.PREDICTED_COLUMN],mode='lines+markers',name=pred_demand,line=dict(color='green'))
    # trace for upper bound
    dfn.add_fig_trace(fig,x=results_df[fn.PREDICTED_DATE_COLUMN],y=results_df[fn.UPPER_BOUND_COLUMN],mode='lines',name=upper_bound,line=dict(color='grey'))
    # trace for lower bound
    dfn.add_fig_trace(fig,x=results_df[fn.PREDICTED_DATE_COLUMN],y=results_df[fn.LOWER_BOUND_COLUMN],mode='lines',name=lower_bound,line=dict(color='grey'),fill='tonexty')

    xaxis = fn.outside_translate("article.xaxis_title")
    yaxis = fn.outside_translate("article.yaxis_title")

    # Check if planned demand data exists and plot it
    if planned_demand_data and 'dates' in planned_demand_data and 'demands' in planned_demand_data:
        logging.info("Overlaying planned demand from URL on forecast graph.")
        plan_start_date = pd.to_datetime(planned_demand_data['dates'][0])
        plan_end_date = pd.to_datetime(planned_demand_data['dates'][-1])
        # Show a few weeks of history before the plan starts
        graph_start_date = plan_start_date - pd.Timedelta(weeks=8)
        x_axis_range = [graph_start_date, plan_end_date]
        planned_dates = planned_demand_data['dates']
        planned_demands = planned_demand_data['demands']
        title_tmp = fn.outside_translate("simulation.planned_demand_label")
        dfn.add_fig_trace(fig,x=planned_dates,y=planned_demands,mode='lines+markers',name=title_tmp,line=dict(color='red', width=3))
        title_template = fn.outside_translate("simulation.forecast_with_planned_title")
        title = title_template.format(name=name)
        fig.update_layout(
            title=title,
            xaxis_title=xaxis,
            yaxis_title=yaxis,
            hovermode="x unified",
            xaxis_tickformat="W%V - %y",
        )
    # Update layout with titles and axis labels
    else:
        title_template = fn.outside_translate("simulation.forecast_title")
        title = title_template.format(name=name)
        fig.update_layout(
            title=title,
            xaxis_title=xaxis,
            yaxis_title=yaxis,
            hovermode="x unified",
            xaxis_tickformat = "W%V - %y"
        )

    #comparison table from planned
    comparison_table_div = html.Div()  # Default to empty Div
    if planned_demand_data:
        try:
            plan_start_date = pd.to_datetime(planned_demand_data['dates'][0])
            plan_end_date = pd.to_datetime(planned_demand_data['dates'][-1])
            x_axis_range = [plan_start_date, plan_end_date]
            df_planned = pd.DataFrame(planned_demand_data)
            df_planned.rename(columns={'demands': 'Geplanter Bedarf'}, inplace=True)
            df_planned['dates'] = pd.to_datetime(df_planned['dates']).dt.strftime('%Y-%m-%d')

            results_df_copy = results_df.copy()
            results_df_copy[fn.PREDICTED_DATE_COLUMN] = results_df_copy[fn.PREDICTED_DATE_COLUMN].dt.strftime(
                '%Y-%m-%d')

            df_merged = pd.merge(df_planned, results_df_copy, left_on='dates', right_on=fn.PREDICTED_DATE_COLUMN,
                                 how='left')

            df_display = df_merged[
                ['dates', 'Geplanter Bedarf', fn.PREDICTED_COLUMN, fn.LOWER_BOUND_COLUMN, fn.UPPER_BOUND_COLUMN]]
            df_display.columns = ['Woche', 'Geplanter Bedarf', 'Standard', 'Konservativ',
                                  'Optimistisch']

            df_display = df_display.fillna(0)
            for col in df_display.columns:
                if col != 'Woche':
                    df_display[col] = df_display[col].apply(lambda x: math.ceil(x) if x > 0 else 0)

            numeric_cols = ['Geplanter Bedarf', 'Standard', 'Konservativ', 'Optimistisch']
            df_display['Durchschnitt'] = df_display[numeric_cols].sum(axis=1) * 0.25
            df_display['Durchschnitt'] = df_display['Durchschnitt'].apply(lambda x: math.ceil(x) if x > 0 else 0)


            df_display['Month'] = pd.to_datetime(df_display['Woche']).dt.strftime('%m-%Y')
            planned_demand_dict = df_display.groupby('Month')['Geplanter Bedarf'].sum().to_dict()
            average_demand_dict = df_display.groupby('Month')['Durchschnitt'].sum().to_dict()

            # Run new simulations
            planned_sim = fn.run_inventory_simulation(planned_demand_dict, "Planned Demand", inventory_info,
                                                      final_stock)
            average_sim = fn.run_inventory_simulation(average_demand_dict, "Average Demand", inventory_info,
                                                      final_stock)

            # df_display['Month'] = pd.to_datetime(df_display['Woche']).dt.strftime('%m-%Y')
            df_display['Woche'] = pd.to_datetime(df_display['Woche']).dt.strftime("W%V - %y")
            comparison_table = dbc.Table.from_dataframe(df_display, striped=True, bordered=True, hover=True,
                                                        responsive=True)

            comparison_table_div = html.Div([
                html.H4(id={"type": "i18n", "key": "simulation.comparison_heading"}, className="mt-4 mb-3"),
                comparison_table
            ])


            planned_replenishment_data = planned_sim.to_dict('records') if not planned_sim.empty else []
            average_replenishment_data = average_sim.to_dict('records') if not average_sim.empty else []
        except Exception as e:
            logging.error(f"Error in simulation callback: {e}")
            error_msg = fn.outside_translate("simulation.generic_error_message").format(e=e)
            error_alert = dbc.Alert(error_msg, color="danger")
            # Return empty values for all 9 outputs
            return go.Figure(), error_alert, "", "", "", "", "", "", ""

    logging.info("Returned Figure, guideline, reorder table")
    # Return Graph, Guidelines, Monthly Demand, normal, minimal and maximal replenishment scenario
    return fig, guideline_md, monthly_demand_dash, comparison_table_div, normal_demand_table, minimum_demand_table, maximum_demand_table, planned_replenishment_data, average_replenishment_data

# when pressing the scenario button, calculated replenishment tables should be loaded from dcc.Store
@callback(
    Output("scenario-recommendation-output", "children"),
    [
        Input("min-btn", "n_clicks"),
        Input("normal-btn", "n_clicks"),
        Input("max-btn", "n_clicks"),
        Input("planned-btn", "n_clicks"),
        Input("avg-btn", "n_clicks"),
    ],
    [
        State("minimum_replenishment", "data"),
        State("normal_replenishment", "data"),
        State("maximum_replenishment", "data"),
        State("planned_replenishment", "data"),
        State("average_replenishment", "data"),
        State("end-product-dropdown", "value"),
    ],
    prevent_initial_call=True
)
def update_scenario_table(min_clicks, normal_clicks, max_clicks, planned_clicks, avg_clicks,
                          minimum_data, normal_data, maximum_data, planned_data, average_data, selected_item):
    if not selected_item:
        raise dash.exceptions.PreventUpdate

    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    scenario_title = "Scenario"
    data_to_display = None

    month = fn.outside_translate("article.month")
    stock_level_start = fn.outside_translate("simulation.stock_level_start")
    arriving_stock_quantity = fn.outside_translate("simulation.arriving_stock_quantity")
    stock_after_arrival = fn.outside_translate("simulation.stock_after_arrival")
    expected_demand_quantity = fn.outside_translate("simulation.expected_demand_quantity")
    stock_after_demand_consumption = fn.outside_translate("simulation.stock_after_demand_consumption")
    shortfall = fn.outside_translate("simulation.shortfall")
    order_trigger_level = fn.outside_translate("simulation.order_trigger_level")
    inventory_level_end = fn.outside_translate("simulation.inventory_level_end")
    order_decision = fn.outside_translate("simulation.order_decision")
    order_quantity = fn.outside_translate("simulation.order_quantity")
    expected_stock_arrival = fn.outside_translate("simulation.expected_stock_arrival")
    stock_on_order_end = fn.outside_translate("simulation.stock_on_order_end")
    reasoning_for_order_decision = fn.outside_translate("simulation.reasoning_for_order_decision")
    min_dem = fn.outside_translate("simulation.min_demand")
    normal_dem = fn.outside_translate("simulation.normal_demand")
    max_dem = fn.outside_translate("simulation.max_demand")
    planned_dem = fn.outside_translate("simulation.planned_demand")
    avg_dem = fn.outside_translate("simulation.avg_calc_demand")
    yes = fn.outside_translate("functions.yes")
    scenario_title_tmp = min_dem

    if triggered_id == "min-btn" and minimum_data:
        data_to_display = minimum_data
        scenario_title = "Minimum Demand"
    elif triggered_id == "normal-btn" and normal_data:
        data_to_display = normal_data
        scenario_title = "Normal Demand"
        scenario_title_tmp = normal_dem
    elif triggered_id == "max-btn" and maximum_data:
        data_to_display = maximum_data
        scenario_title = "Maximum Demand"
        scenario_title_tmp = max_dem
    elif triggered_id == "planned-btn" and planned_data:
        data_to_display = planned_data
        scenario_title = "Planned Demand"
        scenario_title_tmp = planned_dem
    elif triggered_id == "avg-btn" and average_data:
        data_to_display = average_data
        scenario_title = "Average Calculated Demand"
        scenario_title_tmp = avg_dem

    if data_to_display is None:
        return dbc.Alert(id={"type": "i18n", "key": "simulation.no_replenishment_data_available"}, color="warning")

    if not data_to_display:
        return html.Div(fn.outside_translate("simulation.no_replenishment_needed").format(scenario_title=scenario_title))

    df = pd.DataFrame(data_to_display)

    # Summary Calculation
    total_ordered = df['Order Quantity'].sum()
    total_shortfall = df['Shortfall'].sum()
    total_shortfall_for_summary = df['Shortfall'].sum()  # Update shortfall for the summary card
    num_orders = df[df['Order Decision'] == yes].shape[0]
    # Ensure 'Stock Level After Demand Consumption' is numeric before calculating the mean
    avg_stock_level = pd.to_numeric(df['Stock Level After Demand Consumption'], errors='coerce').mean()

    df.rename(columns={
        'Month': month,
        'Stock Level @ Start': stock_level_start,
        'Arriving Stock Quantity': arriving_stock_quantity,
        'Stock Level After Arrival of Stocks': stock_after_arrival,
        'Expected Demand Quantity': expected_demand_quantity,
        'Stock Level After Demand Consumption': stock_after_demand_consumption,
        'Shortfall': shortfall,
        'Order Trigger Level': order_trigger_level,
        'Inventory Level @ End': inventory_level_end,
        'Order Decision': order_decision,
        'Order Quantity': order_quantity,
        'Expected Stock Arrival': expected_stock_arrival,
        'Stock On Order @ End': stock_on_order_end,
        'Reasoning For Order Decision': reasoning_for_order_decision
    }, inplace=True)

    # Prepare data for plotting green metrics across all scenarios
    scenario_green_metrics = []
    scenario_data_map = {
        min_dem: minimum_data,
        normal_dem: normal_data,
        max_dem: maximum_data,
        planned_dem: planned_data,
        avg_dem: average_data,
    }

    for name, data in scenario_data_map.items():
        if data:
            scenario_df = pd.DataFrame(data)
            s_avg_stock_level = pd.to_numeric(scenario_df['Stock Level After Demand Consumption'], errors='coerce').mean()
            s_num_orders = scenario_df[scenario_df['Order Decision'] == yes].shape[0]
            scenario_green_metrics.append({
                "Szenario": name,
                "Durchschnittlicher Lagerbestand": s_avg_stock_level,
                "Geschätzte Lieferungen": s_num_orders,
            })
        else:
            scenario_green_metrics.append({
                "Szenario": name,
                "Durchschnittlicher Lagerbestand": None,
                "Geschätzte Lieferungen": None,
            })

    metrics_df = pd.DataFrame(scenario_green_metrics)

    # Get metrics for the currently selected scenario to pass to the recommendation function
    selected_scenario_metrics_for_text = metrics_df[metrics_df["Szenario"] == scenario_title_tmp].iloc[0].to_dict()

    # Create a bar chart for Average Inventory
    fig_avg_inventory = go.Figure(
        data=[go.Bar(x=metrics_df["Szenario"], y=metrics_df["Durchschnittlicher Lagerbestand"], marker_color='lightgreen')],
        layout=go.Layout(
            title_text=fn.outside_translate("simulation.avg_stocklevel_all_scenarios"),
            xaxis_title=fn.outside_translate("simulation.scenario"),
            yaxis_title=fn.outside_translate("simulation.stocklvl"),
            height=300, margin=dict(t=50, b=50, l=50, r=50)
        )
    )

    # Create a bar chart for Estimated Shipments
    fig_estimated_shipments = go.Figure(
        data=[go.Bar(x=metrics_df["Szenario"], y=metrics_df["Geschätzte Lieferungen"], marker_color='lightblue')],
        layout=go.Layout(
            title_text=fn.outside_translate("simulation.exp_ship"),
            xaxis_title=fn.outside_translate("simulation.scenario"),
            yaxis_title=fn.outside_translate("simulation.amount_needed"),
            height=300, margin=dict(t=50, b=50, l=50, r=50)
        )
    )
    # get item origin
    origin = fn.get_origin(selected_item)
    # get inventory data
    inventory_data = fn.extract_inventory_data()
    inventory_info = fn.get_info(inventory_data, selected_item)
    safety_stock = inventory_info.safety_stock
    # Add a horizontal line for safety stock if available
    if safety_stock is not None:
        fig_avg_inventory.add_shape(type="line",
                                    x0=-0.5, y0=safety_stock,
                                    x1=len(metrics_df["Szenario"]) - 0.5, y1=safety_stock,
                                    line=dict(color="Red", width=2, dash="dash"),
                                    name=fn.outside_translate("simulation.saftey_stock"))
        title = fn.outside_translate("simulation.safety_stock").format(safety_stock=f"{safety_stock:,.0f}")
        fig_avg_inventory.add_annotation(x=metrics_df["Szenario"].iloc[-1],
                                        y=safety_stock,
                                        text=title,
                                        showarrow=False,
                                        yshift=10, # Adjust position to avoid overlapping with line
                                        xshift=-50, # Adjust position to avoid overlapping with line
                                        font=dict(color="Red"))

    recommendation_and_tradeoff_content = fn.generate_recommendation_text(
        scenario_title,
        selected_scenario_metrics_for_text,
        metrics_df,  # Pass the full metrics_df for comparative analysis in the function
        total_shortfall_for_summary,
        origin,
        inventory_info,
        total_ordered
    )

    total_ord_qty = fn.outside_translate("simulation.total_ordered_quantity").format(total_ordered=f"{total_ordered:,.0f}")
    shortfall_str = fn.outside_translate("simulation.total_shortfall").format(total_shortfall=f"{total_shortfall:,.0f}")
    metrics_title = fn.outside_translate("simulation.green_efficiency_metrics").format(scenario_title=scenario_title)
    avg_inv = fn.outside_translate("simulation.avg_inventory").format(avg_stock_level=avg_stock_level)
    est_ship = fn.outside_translate("simulation.estimated_shipments").format(num_orders=num_orders)
    origin_str = fn.outside_translate("simulation.origin").format(origin=origin)
    all_scenarios = fn.outside_translate("simulation.green_efficiency_metrics_all")
    # Summary card with summary information of table
    summary_card_children = [
        html.H5(fn.outside_translate("simulation.summary_for").format(scenario_title=scenario_title), className="card-title"),
        dcc.Markdown(f"""
            - {total_ord_qty}
            - {shortfall_str}
            ---
            ##### {metrics_title}
            - {avg_inv}
            - {est_ship}
            - {origin_str}
            ---
            ##### {all_scenarios}
        """),
        dcc.Graph(figure=fig_avg_inventory, config={'displayModeBar': False}),
        dcc.Graph(figure=fig_estimated_shipments, config={'displayModeBar': False}),
        html.Hr(),  # Separator
        *recommendation_and_tradeoff_content
    ]

    summary_card = dbc.Card(
        dbc.CardBody(summary_card_children),
        className="mt-4"
    )

    header_cells = [html.Th([col, dbc.Tooltip(fn.COLUMN_EXPLANATIONS.get(col, fn.outside_translate("simulation.no_expl")),
                                              target=f"{col.replace(' ', '_')}_tooltip", placement="top")],
                            id=f"{col.replace(' ', '_')}_tooltip") for col in df.columns]

    table_header = html.Thead(html.Tr(header_cells))
    table_body = html.Tbody([html.Tr([html.Td(row[col]) for col in df.columns]) for _, row in df.iterrows()])

    table = dbc.Table([table_header, table_body], bordered=True, hover=True, striped=True, responsive=True)

    # Return a list containing the table and the new summary card
    return [table, summary_card]
