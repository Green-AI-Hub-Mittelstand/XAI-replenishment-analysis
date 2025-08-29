import logging
import logging
import dash
from dash import callback, Input, Output, State, dash_table, html, dcc, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
from collections import Counter
from typing import Dict, List, Tuple, Union, Optional, Any
from . import functions as fn # Assuming fn.create_br_dict() is loaded elsewhere
from . import dash_functions as dfn # Not used in this specific callback
from dash.exceptions import PreventUpdate
import math
from collections import defaultdict
from datetime import datetime, timedelta # Needed for date calculations
from urllib.parse import urlencode


dash.register_page(__name__, path="/planning")


try:
    br_data_dict_planning = fn.create_br_dict()
    baureihe_options_planning = [{'label': br, 'value': br} for br in br_data_dict_planning.keys()]
except Exception as e:
    logging.error(f"Could not load Baureihe data for planning page: {e}")
    baureihe_options_planning = []

# Page Layout
layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1(id={"type": "i18n", "key": "planung.header_title"}, className="text-center text-primary my-4"))),
    dbc.Row([
            dbc.Row(dbc.Col(html.H2(id={"type": "i18n", "key": "planung.subheader_select_period"}, className="text-center text-primary my-4"))),
            dcc.DatePickerRange(
                id="date-picker-range-planning",
                start_date_placeholder_text="",
                end_date_placeholder_text="",
                calendar_orientation="horizontal",
                className="text-center text-primary my-4",
                first_day_of_week=1,
            )
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(id="baureihe-input-container")
        ], width=12)
    ]),
    dbc.Row([
        dbc.Button("", id="submit-planning-button", color="primary", className="mt-3 text-center ", n_clicks=0)
    ]),
    # Table Storage
    dcc.Store(id='weekly-plan-storage',storage_type='session'),
    dcc.Store(id='product-distribution-storage',storage_type='session'),
    dcc.Store(id='planning-page-user-inputs-storage', storage_type='session'),
    # Output Divs
    dbc.Row([ # row for displaying submission output
        dbc.Col(html.Div(id="weekly-plan", className="mt-4"), width=12)
    ]),
    dbc.Row([ # row for displaying submission output
        dbc.Col(html.Div(id="planning-submission-output", className="mt-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id="product-distribution-output", className="mt-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id="material-distribution-output", className="mt-4"), width=12)
    ]),
    dcc.Store(id='all-forecasts-storage', storage_type='session'),
    dcc.Store(id='planned-data-storage', storage_type='session'),
    dbc.Row([
        dbc.Col(html.Div(id="forecasted-material-output", className="mt-4"), width=12)
    ])
], fluid=True)

@callback(
    Output('date-picker-range-planning', 'start_date_placeholder_text'),
    Output('date-picker-range-planning', 'end_date_placeholder_text'),
    Output('submit-planning-button', 'children'),
    Input('language-store', 'data')
)
def update_datepicker_placeholders(lang):
    start_ph = fn.outside_translate("planung.datepicker_start_placeholder")
    end_ph   = fn.outside_translate("planung.datepicker_end_placeholder")
    submit_text = fn.outside_translate("planung.submit_button_text")
    return start_ph, end_ph, submit_text

# Update the input fields when the date range is selected for planning production.
@callback(
    Output("baureihe-input-container", "children"),
    Input("date-picker-range-planning", "start_date"),
    Input("date-picker-range-planning", "end_date")
)
def generate_baureihe_inputs(start_date: Optional[str], end_date: Optional[str]) -> List[dbc.Row]: # type: ignore
    """
    Generates input fields for each Baureihe, arranged in two columns,
    when a date range is selected for planning production.
    """
    if start_date is not None and end_date is not None:
        if not baureihe_options_planning:
            # Ensure the alert is a list of components if the output expects a list.
            return [dbc.Alert(id={"type": "i18n", "key": "planung.no_baureihe_loaded"}, color="warning")] # type: ignore

        generated_layout_rows = []
        num_options = len(baureihe_options_planning)

        # Add a row for each baureihe making 2 columns for each pair of Baureihe
        for i in range(0, num_options, 2):
            # Create layout for the first item in the pair
            br_option1 = baureihe_options_planning[i]
            baureihe_label1 = br_option1['label']
            input_id1 = {"type": "baureihe-input", "index": baureihe_label1}
            label_template = fn.outside_translate("planung.anzahl_baureihe_label")
            label_text = label_template.format(baureihe=baureihe_label1)
            item1_layout = dbc.Row([
                dbc.Col(html.Label(label_text), width=4, className="text-end align-self-center"),
                dbc.Col(
                    dbc.Input(
                        id=input_id1,
                        type="number",
                        value=0,
                        min=0,
                    ),
                    width=8
                )
            ], className="mb-2 align-items-center")
            col1 = dbc.Col(item1_layout, md=6)

            # Create layout for the second item in the pair
            if i + 1 < num_options:
                br_option2 = baureihe_options_planning[i+1]
                baureihe_label2 = br_option2['label']
                input_id2 = {"type": "baureihe-input", "index": baureihe_label2}
                label_template = fn.outside_translate("planung.anzahl_baureihe_label")
                label_text = label_template.format(baureihe=baureihe_label2)
                item2_layout = dbc.Row([
                    dbc.Col(html.Label(label_text), width=4, className="text-end align-self-center"),
                    dbc.Col(
                        dbc.Input(
                            id=input_id2,
                            type="number",
                            value=0,
                            min=0,
                        ),
                        width=8
                    )
                ], className="mb-2 align-items-center")
                col2 = dbc.Col(item2_layout, md=6)
                current_row = dbc.Row([col1, col2])
            else:
                # If there's an odd number of items, the last item takes a half-width column
                current_row = dbc.Row([col1]) # col1 is already md=6

            generated_layout_rows.append(current_row)

        return generated_layout_rows
    return [] # Return an empty list if no date range is selected

# When button is pressed create editable weekly production table.
@callback(
    Output("weekly-plan", "children"),
    Output("planning-submission-output", "children"),  # For status messages
    Output("weekly-plan-storage", "data"),  # To store table data
    Input("submit-planning-button", "n_clicks"),
    State("date-picker-range-planning", "start_date"),
    State("date-picker-range-planning", "end_date"),
    State({"type": "baureihe-input", "index": dash.ALL}, "value"),
    State({"type": "baureihe-input", "index": dash.ALL}, "id"),
    prevent_initial_call=True
)
def create_editable_weekly_plan(
        n_clicks: Optional[int],
        start_date: Optional[str],
        end_date: Optional[str],
        baureihe_values: Optional[List[Optional[Union[int, float]]]],  # Values can be None or numbers
        baureihe_ids: Optional[List[Dict[str, str]]]
) -> Tuple[
    Union[dash_table.DataTable, dash.development.base_component.Component], dbc.Alert, Optional[List[Dict[str, Any]]]]:
    """
    Create an editable weekly production table.
    Distributes entered Baureihe quantities across weeks in the selected date range.
    Saves the table data to a dcc.Store.
    """
    if not n_clicks or n_clicks == 0:
        return no_update, no_update, no_update

    if not start_date or not end_date:
        return no_update, dbc.Alert(id={"type": "i18n", "key": "planung.select_start_end_date"}, color="warning"), no_update

    # 1. Extract Baureihe quantities
    planned_quantities: Dict[str, int] = {}
    if baureihe_values and baureihe_ids:
        for id_dict, value in zip(baureihe_ids, baureihe_values):
            br_label = id_dict.get("index")
            if br_label and value is not None:
                try:
                    quantity = int(value)
                    if quantity > 0:
                        planned_quantities[br_label] = quantity
                except ValueError:
                    logging.warning(f"Invalid quantity for {br_label}: {value}")

    if not planned_quantities:
        return html.Div(), dbc.Alert(id={"type": "i18n", "key": "planung.no_positive_quantities_entered"}, color="info"), no_update

    # 2. Determine weeks in the selected range
    weeks = dfn.get_weeks_in_range_planning(start_date, end_date)
    if not weeks:
        msg = fn.outside_translate("planung.no_valid_weeks_in_period")
        if start_date and end_date and datetime.strptime(start_date, "%Y-%m-%d") > datetime.strptime(end_date,
                                                                                                     "%Y-%m-%d"):
            msg = fn.outside_translate("planung.start_date_after_end_date")
        return html.Div(), dbc.Alert(msg, color="warning"), no_update

    num_weeks = len(weeks)

    # 3. Prepare data for DataFrame
    table_data = []
    for br_label, total_quantity in planned_quantities.items():
        row_data: Dict[str, Any] = {"Baureihe": br_label}

        base_weekly_qty = total_quantity // num_weeks
        remainder = total_quantity % num_weeks

        distributed_sum = 0
        for i, week_label in enumerate(weeks):
            qty_for_this_week = base_weekly_qty + (1 if i < remainder else 0)
            row_data[week_label] = qty_for_this_week
            distributed_sum += qty_for_this_week

        # Ensure the sum is correct
        if distributed_sum != total_quantity and weeks:
            row_data[weeks[-1]] += (total_quantity - distributed_sum)

        row_data["Total"] = total_quantity
        table_data.append(row_data)

    if not table_data:  # Should be caught by planned_quantities check, but as a safeguard
        return html.Div(), dbc.Alert(id={"type": "i18n", "key": "planung.no_data_generated_for_table"}, color="info"), no_update

    df = pd.DataFrame(table_data)
    baureihe = fn.outside_translate("planung.baureihe")
    df.rename(columns={"Baureihe": baureihe}, inplace=True)

    # Ensure correct column order: Baureihe, Week1, Week2, ..., Total
    column_order = [baureihe] + weeks + ["Total"]
    df = df[column_order]

    #  4. Create DataTable
    # Week columns are editable; 'Baureihe' and 'Total' are not.
    dt_columns = [
        {"name": col, "id": col, "editable": (col not in [baureihe, "Total"])}
        for col in df.columns
    ]

    editable_table = dash_table.DataTable(
        id='editable-weekly-plan-table',
        columns=dt_columns,
        data=df.to_dict('records'),
        editable=True,
        style_table={'overflowX': 'auto', 'minWidth': '100%'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_cell={
            'textAlign': 'center',
            'minWidth': '100px', 'width': '120px', 'maxWidth': '150px',
            'whiteSpace': 'normal',
            'height': 'auto',
            'padding': '5px'
        },
        style_cell_conditional=[  # Non-editable columns
            {
                'if': {'column_id': 'Baureihe'},
                'editable': False,
                'backgroundColor': 'rgb(240, 240, 240)'
            },
            {
                'if': {'column_id': 'Total'},
                'editable': False,
                'backgroundColor': 'rgb(240, 240, 240)'
            }
        ],
        row_deletable=False,
        page_size=20,  # Adjust as needed
        filter_action='native',
        sort_action='native',
    )

    stored_data = df.to_dict('records')

    # translations
    template = fn.outside_translate("planung.production_plan_for_period")
    title = template.format(start_date=start_date, end_date=end_date)
    info_text = fn.outside_translate("planung.table_editing_instruction")
    label_text = fn.outside_translate("planung.select_product_distribution_mode")
    label_1 = fn.outside_translate("planung.auto_historical_product_distribution")
    label_2 = fn.outside_translate("planung.manual_input_per_product")
    button = fn.outside_translate("planung.calculate_product_distribution")


    output_component_for_weekly_plan = html.Div([
        html.H4(title,
                className="mt-4 mb-2"),
        html.P(
            info_text,
            className="text-muted small mb-3"
        ),
        editable_table,
        dbc.Row([  # Row for mode selector
            dbc.Col(
                dbc.Form([
                    dbc.Label(label_text, html_for="product-distribution-mode-selector",
                              className="me-2"),
                    dbc.RadioItems(
                        id="product-distribution-mode-selector",
                        options=[
                            {'label': label_1, 'value': 'automatic'},
                            {'label': label_2, 'value': 'manual'},
                        ],
                        value='automatic',  # Default value
                        inline=True,
                        className="mt-1"
                    )
                ]),
                width={"size": 8, "offset": 0}  # Adjust width and offset as needed
            )
        ], className="mt-3 align-items-center"),  # Added className for vertical alignment
        dbc.Row([  # Row for the button
            dbc.Col(
                dbc.Button(button, id="btn-calculate-prod-dist", color="primary",
                           className="mt-3",
                           n_clicks=0),
                width={"size": "auto", "offset": 0}  # Auto width for button
            )
        ])
    ])

    return output_component_for_weekly_plan, dbc.Alert(id={"type": "i18n", "key": "planung.weekly_plan_created_success"},
                                                       color="success",
                                                       duration=4000), stored_data

# Update weekly totals when a weekly quantity is edited in the DataTable.
@callback(
    Output('editable-weekly-plan-table', 'data', allow_duplicate=True),
    Output('weekly-plan-storage', 'data', allow_duplicate=True),
    Input('editable-weekly-plan-table', 'data'),
    State('editable-weekly-plan-table', 'columns'),
    prevent_initial_call=True  # only run on user edits, not initial table load
)
def update_totals_on_edit(
        current_table_data: Optional[List[Dict[str, Any]]],
        table_columns: Optional[List[Dict[str, str]]]
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
    """
    Updates the 'Total' column in the table and the dcc.Store
    when a weekly quantity is edited in the DataTable.
    """
    if current_table_data is None or table_columns is None:
        return no_update, no_update

    # Identify which columns are week columns (i.e., not 'Baureihe' or 'Total')
    baureihe = fn.outside_translate("planung.baureihe")
    week_column_ids = [
        col['id'] for col in table_columns
        if col['id'] not in [baureihe, 'Total']
    ]

    updated_data_for_table = []

    for row in current_table_data:
        new_row = row.copy()  # Work on a copy of the row
        current_row_total = 0
        for week_col_id in week_column_ids:
            try:
                cell_value = new_row.get(week_col_id)
                if cell_value is None or str(cell_value).strip() == "":
                    val_for_sum = 0
                else:
                    val_for_sum = int(float(str(cell_value)))
                current_row_total += val_for_sum
            except (ValueError, TypeError):
                pass

        new_row['Total'] = current_row_total
        updated_data_for_table.append(new_row)

    # Return the modified data to update both the table display and the dcc.Store
    return updated_data_for_table, updated_data_for_table

#update input when weekly plan was edited
@callback(
    Output({"type": "baureihe-input", "index": dash.ALL}, "value"),
    Input("weekly-plan-storage", "data"), # Triggered when table totals are updated
    State({"type": "baureihe-input", "index": dash.ALL}, "id"), # To know which input corresponds to which Baureihe
    prevent_initial_call=True #
)
def sync_baureihe_inputs_with_table_totals(
    weekly_plan_data: Optional[List[Dict[str, Any]]],
    baureihe_input_ids: List[Dict[str, str]]
) -> List[Any]: # Returns a list of values for each baureihe-input
    """
    When the weekly plan table data (especially totals) changes in weekly-plan-storage,
    this callback updates the corresponding baureihe-input fields.
    """
    if not weekly_plan_data or not baureihe_input_ids:
        # If there's no table data or no input IDs, prevent update for all inputs
        return [no_update] * len(baureihe_input_ids if baureihe_input_ids else [])

    # Create a dictionary to quickly look up totals by Baureihe name
    # from the weekly_plan_data (which comes from the table)
    table_totals_map: Dict[str, int] = {}
    baureihe = fn.outside_translate("planung.baureihe")
    for row in weekly_plan_data:
        baureihe_name = row.get(baureihe)
        total_value = row.get("Total")
        if baureihe_name and total_value is not None:
            try:
                table_totals_map[baureihe_name] = int(float(total_value)) # Ensure it's an int
            except (ValueError, TypeError):
                pass

    # Prepare the list of new values for the baureihe-input fields.
    # The order of this list MUST match the order of baureihe_input_ids.
    new_input_values = []
    updated_any = False
    for id_dict in baureihe_input_ids:
        baureihe_index = id_dict.get("index")
        if baureihe_index in table_totals_map:
            new_input_values.append(table_totals_map[baureihe_index])
            updated_any = True
        else:
            new_input_values.append(no_update)
    if not updated_any and not new_input_values: # Handles case where baureihe_input_ids was empty initially
         raise PreventUpdate
    elif not updated_any and new_input_values: # All were no_update
         raise PreventUpdate

    return new_input_values


# calculate product distribution
@callback(
    Output('product-distribution-output', 'children'),
    Output('product-distribution-storage', 'data'),
    Input('btn-calculate-prod-dist', 'n_clicks'),
    State('weekly-plan-storage', 'data'),
    State('product-distribution-mode-selector', 'value'),
    prevent_initial_call=True
)
def calculate_and_display_product_distribution(
        n_clicks: Optional[int],
        weekly_br_plan_data: Optional[List[Dict[str, Any]]],
        distribution_mode: Optional[str] #
) -> Tuple[Union[html.Div, dash.development.base_component.Component], Optional[List[Dict[str, Any]]]]:
    """
    Calculates and displays the product distribution based on the weekly Baureihe plan.
    If mode is 'automatic', uses historical percentages.
    If mode is 'manual', initializes weekly product quantities to 0 for user input.
    The resulting table shows planned quantities for each ArtikelNr per week and is editable.
    """
    if not n_clicks or n_clicks == 0:
        return no_update, no_update

    if not weekly_br_plan_data:
        return dbc.Alert(id={"type": "i18n", "key": "planung.no_weekly_plan_for_distribution"},
                         color="warning"), no_update

    df_weekly_br_plan = pd.DataFrame(weekly_br_plan_data)
    if df_weekly_br_plan.empty:
        return dbc.Alert(id={"type": "i18n", "key": "planung.empty_weekly_plan"}, color="info"), no_update

    # 1. Get Product Distribution (Historical Percentages) - Needed for 'automatic' mode
    # And to know which products belong to which Baureihe even in 'manual' mode
    try:
        df_prod_dist = fn.create_product_distribution()
    except Exception as e:
        logging.error(f"Error calling create_product_distribution: {e}")
        error_msg_template = fn.outside_translate("planung.load_product_distribution_error")
        error_msg = error_msg_template.format(error=e)
        return dbc.Alert(error_msg, color="danger"), no_update

    if df_prod_dist.empty:
        return dbc.Alert(id={"type": "i18n", "key": "planung.no_product_distribution_found"},
                         color="warning"), no_update

    df_prod_dist['PctProduct'] = pd.to_numeric(df_prod_dist['PctProduct'], errors='coerce').fillna(0.0)

    # 2. Identify Week Columns from the Baureihe Plan
    baureihe = fn.outside_translate("planung.baureihe")
    produktNr = fn.outside_translate("planung.produktNr")
    totalArtikel = fn.outside_translate("planung.totalArtikel")
    week_columns = [col for col in df_weekly_br_plan.columns if col not in [baureihe, 'Total']]
    if not week_columns:
        return dbc.Alert(id={"type": "i18n", "key": "planung.no_week_columns_found"}, color="warning"), no_update

    # 3. Calculate Product-Level Weekly Quantities
    product_level_plan_data = []
    df_weekly_br_plan = df_weekly_br_plan.rename(columns={
        baureihe: "Baureihe"
    })

    for _, br_row in df_weekly_br_plan.iterrows():
        current_baureihe_name = br_row['Baureihe']
        products_in_current_br = df_prod_dist[df_prod_dist['Baureihe'] == current_baureihe_name]

        if products_in_current_br.empty:
            logging.info(f"Keine Produkte in der Verteilungsliste für Baureihe {current_baureihe_name} gefunden.")
            continue

        for _, prod_row in products_in_current_br.iterrows():
            artikel_nr = prod_row['ProduktNr']
            pct_product = float(prod_row['PctProduct'])

            product_data_row: Dict[str, Any] = {
                "Baureihe": current_baureihe_name,
                "ProduktNr": artikel_nr
            }
            total_artikel_quantity = 0.0

            for week_col in week_columns:
                baureihe_weekly_qty = br_row.get(week_col, 0)
                if baureihe_weekly_qty is None: baureihe_weekly_qty = 0.0

                try:
                    baureihe_weekly_qty_numeric = float(str(baureihe_weekly_qty))
                except (ValueError, TypeError):
                    baureihe_weekly_qty_numeric = 0.0

                product_weekly_qty = 0.0 # Default for manual or if issues
                if distribution_mode == 'automatic':
                    product_weekly_qty = baureihe_weekly_qty_numeric * pct_product
                elif distribution_mode == 'manual':
                    product_weekly_qty = 0.0

                product_data_row[week_col] = round(product_weekly_qty,4) # Values will be 0 for manual mode here
                total_artikel_quantity += product_weekly_qty # Will be 0 for manual mode here

            product_data_row["TotalArtikel"] = round(total_artikel_quantity, 4) # Will be 0 for manual
            product_level_plan_data.append(product_data_row)

    if not product_level_plan_data:
        return dbc.Alert(
            id={"type": "i18n", "key": "planung.no_product_quantities_computed"},
            color="info"), no_update

    df_product_level_plan = pd.DataFrame(product_level_plan_data)

    #  4. Create Editable DataTable for Product Distribution
    display_columns_order = ["Baureihe", "ProduktNr"] + week_columns + ["TotalArtikel"]
    # Ensure all columns exist, even if all values are 0 (e.g., in manual mode initially)
    for col in display_columns_order:
        if col not in df_product_level_plan.columns:
            df_product_level_plan[col] = 0.0 if col not in ["Baureihe", "ProduktNr"] else ""


    df_product_level_plan = df_product_level_plan[display_columns_order]
    df_product_level_plan.rename(columns={'Baureihe': baureihe, 'ProduktNr': produktNr, 'TotalArtikel': totalArtikel}, inplace=True)

    dt_prod_columns = [
        {"name": col, "id": col, "type": "numeric", "editable": (col in week_columns)} # Week columns are editable
        for col in df_product_level_plan.columns
    ]

    product_distribution_table = dash_table.DataTable(
        id='editable-product-distribution-table',
        columns=dt_prod_columns,
        data=df_product_level_plan.to_dict('records'),
        editable=True,
        style_table={'overflowX': 'auto', 'minWidth': '100%'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        style_cell={
            'textAlign': 'center', 'minWidth': '100px', 'width': '120px', 'maxWidth': '150px',
            'whiteSpace': 'normal', 'height': 'auto', 'padding': '5px'
        },
        style_cell_conditional=[
            {'if': {'column_id': c}, 'editable': False, 'backgroundColor': 'rgb(240, 240, 240)'}
            for c in ['Baureihe', 'ProduktNr', 'TotalArtikel']
        ],
        row_deletable=False,
        page_size=10,
        filter_action='native',
        sort_action='native',
    )

    manual_mode_hint = ""
    if distribution_mode == 'manual':
        instruction_text = fn.outside_translate("planung.manual_mode_instructions")
        manual_mode_hint = instruction_text

    button_label = fn.outside_translate("planung.calculate_material_consumption")
    output_div = html.Div([
        html.H4(id={"type": "i18n", "key": "planung.product_detail_planning_header"}, className="mt-4 mb-2"),
        (dbc.Alert(manual_mode_hint, color="info", className="mt-2 mb-3",duration=4000) if manual_mode_hint else html.Div()), # Show hint if in manual mode
        html.Div(id="product-distribution-validation-alert"),  # Placeholder for validation alerts
        product_distribution_table,
        dbc.Row([
            dbc.Col(
                dbc.Button(button_label, id="btn-calculate-material", color="primary",
                           className="mt-3",
                           n_clicks=0),
                width={"size": "auto", "offset": 0}  # Auto width for button
            )
        ])
    ])

    return output_div, df_product_level_plan.to_dict('records')


# edit products table
@callback(
    Output('editable-product-distribution-table', 'data', allow_duplicate=True),
    Output('product-distribution-storage', 'data', allow_duplicate=True),
    Output('product-distribution-validation-alert', 'children', allow_duplicate=True), # New Output for alerts
    Input('editable-product-distribution-table', 'data_timestamp'), # Trigger on cell edit
    State('editable-product-distribution-table', 'data'), # Current table data
    State('editable-product-distribution-table', 'data_previous'), # Data before the edit
    State('editable-product-distribution-table', 'active_cell'), # Info about the edited cell
    State('editable-product-distribution-table', 'columns'),
    State('weekly-plan-storage', 'data'), # Baureihe weekly plan
    State('product-distribution-mode-selector', 'value'), # To apply this validation mainly in manual mode
    prevent_initial_call=True
)
def update_product_totals_on_edit(
    timestamp: Optional[int],
    current_product_data: Optional[List[Dict[str, Any]]],
    previous_product_data: Optional[List[Dict[str, Any]]],
    active_cell: Optional[Dict[str, Any]],
    product_table_columns: Optional[List[Dict[str, str]]],
    baureihe_weekly_plan: Optional[List[Dict[str, Any]]],
    distribution_mode: Optional[str]
) -> Tuple[
    Optional[List[Dict[str, Any]]],
    Optional[List[Dict[str, Any]]],
    Optional[dbc.Alert]
]:
    """
    Updates 'TotalArtikel', validates weekly product sum against Baureihe weekly plan,
    and updates storage. Caps input if it exceeds Baureihe's weekly limit.
    """
    alert_message = no_update # Default to no alert
    baureihe = fn.outside_translate("planung.baureihe")
    produktNr = fn.outside_translate("planung.produktNr")
    totalArtikel = fn.outside_translate("planung.totalArtikel")

    if not timestamp or current_product_data is None or previous_product_data is None or active_cell is None or \
       product_table_columns is None or distribution_mode != 'manual':
        if current_product_data and product_table_columns:
            updated_data_no_cap, _ = dfn.recalculate_product_totals(current_product_data, product_table_columns)
            return updated_data_no_cap, updated_data_no_cap, no_update
        return no_update, no_update, no_update


    # Identify which columns are week columns
    week_column_ids = [
        col['id'] for col in product_table_columns
        if col['id'] not in [baureihe, produktNr, totalArtikel]
    ]

    # Create a mutable copy for potential modifications
    processed_product_data = [row.copy() for row in current_product_data]

    # Validation
    if active_cell and 'row_id' not in active_cell and active_cell['column_id'] in week_column_ids and baureihe_weekly_plan:
        edited_row_idx = active_cell['row']
        edited_col_id = active_cell['column_id'] # This is the week column, e.g., "KW23"

        edited_product_row = processed_product_data[edited_row_idx]
        current_br_name = edited_product_row.get(baureihe)

        # 1. Find Baureihe's total for that week
        br_plan_for_week = 0
        for br_row_plan in baureihe_weekly_plan:
            if br_row_plan.get(baureihe) == current_br_name:
                br_plan_for_week = int(float(br_row_plan.get(edited_col_id, 0)))
                break

        # 2. Sum quantities of all products in the same Baureihe for the edited week
        sum_of_products_in_br_for_week = 0
        for i, p_row in enumerate(processed_product_data):
            if p_row.get(baureihe) == current_br_name:
                try:
                    val = float(str(p_row.get(edited_col_id, 0)))
                    sum_of_products_in_br_for_week += val
                except (ValueError, TypeError):
                    pass # Ignore non-numeric values for this sum

        # 3. Check and Cap
        if sum_of_products_in_br_for_week > br_plan_for_week:
            # Exceeded! Calculate how much to reduce the current edited value
            overage = sum_of_products_in_br_for_week - br_plan_for_week
            try:
                current_edited_value = float(str(edited_product_row.get(edited_col_id, 0)))
                capped_value = current_edited_value - overage
                if capped_value < 0: capped_value = 0 # Ensure it's not negative

                # Update the cell in our working copy of the data
                processed_product_data[edited_row_idx][edited_col_id] = round(capped_value, 4)
                msg_template = fn.outside_translate("planung.cap_product_quantity_warning")
                msg = msg_template.format(
                    produkt=edited_product_row.get(produktNr),
                    woche=edited_col_id,
                    menge=capped_value,
                    limit=br_plan_for_week
                )
                alert_message = dbc.Alert(msg, color="warning", duration=8000, dismissable=True)
            except (ValueError, TypeError):
                 pass


    #  Recalculate 'TotalArtikel' for all rows based on (potentially capped) processed_product_data
    final_updated_data, changed_something_total = dfn.recalculate_product_totals(processed_product_data, product_table_columns, week_column_ids)

    if not changed_something_total and alert_message is no_update: # If neither totals changed nor an alert was generated
        # If an alert was generated, it means capping happened, so we should update.
        if not isinstance(alert_message, dbc.Alert) and not dfn.final_updated_data_differs(current_product_data, final_updated_data):
             raise PreventUpdate
    return final_updated_data, final_updated_data, alert_message


# calculates the material with the produkt distribution table
@callback(
    Output("material-distribution-output", "children"),
    Input("btn-calculate-material", "n_clicks"),
    State("product-distribution-storage", "data"),
    prevent_initial_call=True
)
def calculate_material_distribution(
        n_clicks: Optional[int],
        product_plan_data: Optional[List[Dict[str, Any]]]
) -> Union[dbc.Alert, html.Div]:
    """
    Calculates the material distribution based on the product plan data provided
    in the state and using coefficients for each component associated with
    product series (Baureihe).
    """
    baureihe = fn.outside_translate("planung.baureihe")
    produktNr = fn.outside_translate("planung.produktNr")
    totalArtikel = fn.outside_translate("planung.totalArtikel")
    komponente = fn.outside_translate("series.col_komponente")
    weighted = fn.outside_translate("series.col_weighted_avg_prod_koeff_numeric")
    forecast_button_label = fn.outside_translate("planung.calculate_demand_forecast_button")
    if not n_clicks or n_clicks == 0:
        raise PreventUpdate

    if not product_plan_data:
        return dbc.Alert(id={"type": "i18n", "key": "planung.no_product_detail_plan_found"},
                         color="warning")

    component_material_weekly_req = defaultdict(lambda: defaultdict(float))
    cached_component_coeffs_for_baureihe = {}
    all_week_columns_in_plan = set()
    df_product_plan = pd.DataFrame(product_plan_data)

    if df_product_plan.empty:
        return dbc.Alert(id={"type": "i18n", "key": "planung.empty_product_detail_plan"}, color="info")

    potential_week_cols = [col for col in df_product_plan.columns if
                           col not in [baureihe, produktNr, totalArtikel]]
    for wc in potential_week_cols: all_week_columns_in_plan.add(wc)

    for product_row_idx, product_row_series in df_product_plan.iterrows():
        product_row = product_row_series.to_dict()
        baureihe_name = product_row.get(baureihe)
        artikel_nr = product_row.get(produktNr)

        if not baureihe_name or not artikel_nr:
            logging.warning(f"Zeile in Produktdetailplanung ohne Baureihe oder ProduktNr: {product_row}")
            continue

        baureihe_level_coeffs = cached_component_coeffs_for_baureihe.get(baureihe_name)
        if baureihe_level_coeffs is None:
            try:
                components_info_df = fn.generate_component_quantity_probabilities2(baureihe_key=baureihe_name)
                components_info_df.rename(columns={'Component': komponente, 'Gew. Mittel Produktionskoeff. (Komponente)': weighted}, inplace=True)
                print(components_info_df.columns)
                print(komponente, weighted)
                if not components_info_df.empty and \
                        komponente in components_info_df.columns and \
                        weighted in components_info_df.columns:
                    unique_baureihe_components_coeffs = components_info_df.drop_duplicates(subset=[komponente])
                    baureihe_level_coeffs = pd.Series(
                        unique_baureihe_components_coeffs[weighted].values,
                        index=unique_baureihe_components_coeffs[komponente],
                    ).to_dict()
                else:
                    baureihe_level_coeffs = {}
            except Exception as e:
                logging.error(f"Fehler beim Abrufen der Komponentendaten für Baureihe {baureihe_name}: {e}",
                              exc_info=True)
                baureihe_level_coeffs = {}
            cached_component_coeffs_for_baureihe[baureihe_name] = baureihe_level_coeffs

        if not baureihe_level_coeffs:
            continue

        for week_col_name in potential_week_cols:
            try:
                planned_artikel_qty = float(product_row.get(week_col_name, 0.0))
            except (ValueError, TypeError):
                planned_artikel_qty = 0.0
            if planned_artikel_qty > 0:
                for component_name, coefficient in baureihe_level_coeffs.items():
                    if pd.isna(coefficient):
                        continue
                    material_needed_for_component_in_week = planned_artikel_qty * coefficient
                    component_material_weekly_req[component_name][
                        week_col_name] += material_needed_for_component_in_week

    if not component_material_weekly_req:
        return dbc.Alert(id={"type": "i18n", "key": "planung.no_material_demand_computed"},
                         color="info")

    ordered_week_cols = sorted(list(all_week_columns_in_plan))
    table_records = []
    for component_name, weekly_data in component_material_weekly_req.items():
        record = {komponente: component_name}
        component_total_sum = 0.0
        for week_col in ordered_week_cols:
            raw_qty = weekly_data.get(week_col, 0.0)
            ceiled_qty = math.ceil(raw_qty) if raw_qty > 0 else 0
            record[week_col] = ceiled_qty
            component_total_sum += ceiled_qty
        record['Total'] = component_total_sum
        table_records.append(record)

    if not table_records:
        return dbc.Alert(id={"type": "i18n", "key": "planung.no_material_demand_for_table"}, color="info")

    material_df = pd.DataFrame(table_records)
    if komponente not in material_df.columns:
        logging.error("Spalte 'Komponente' fehlt im finalen Material-DataFrame.")
        return dbc.Alert(id={"type": "i18n", "key": "planung.could_not_create_component_column"}, color="danger")

    df_column_order = [komponente] + ordered_week_cols + ['Total']
    df_column_order = [col for col in df_column_order if col in material_df.columns]
    material_df = material_df[df_column_order]
    material_df = material_df.sort_values(by=komponente).reset_index(drop=True)

    data_for_table_with_markdown_links = []
    for row_dict in material_df.to_dict('records'):
        new_row = row_dict.copy()
        komponente_id = new_row[komponente]

        # Extract week labels and demand values for the URL
        week_labels = ordered_week_cols
        demand_values = [str(new_row.get(week, 0)) for week in week_labels]

        # Create the query string payload
        params = {
            'item': komponente_id,
            'weeks': ",".join(week_labels),
            'demands': ",".join(demand_values)
        }
        query_string = urlencode(params)

        # Create a Markdown link string with the encoded data
        new_row[komponente] = f"[{komponente_id}](/simulation?{query_string})"
        data_for_table_with_markdown_links.append(new_row)

    columns_for_table = []
    for col_name in material_df.columns:
        if col_name == komponente:
            # Presentation markdown is needed to render the Markdown link string as an HTML <a> tag
            columns_for_table.append({"name": col_name, "id": col_name, "presentation": "markdown"})
        else:
            columns_for_table.append({"name": col_name, "id": col_name})

    material_table = dash_table.DataTable(
        id='material-distribution-table',
        columns=columns_for_table,
        data=data_for_table_with_markdown_links,  # Use data with Markdown strings
        markdown_options={'html': True},  # Important for rendering HTML from Markdown if needed,
        # but primarily 'presentation: "markdown"' enables this.
        sort_action="native",
        filter_action="native",
        page_size=10,
        style_table={'overflowX': 'auto', 'minWidth': '100%'},
        style_header={'backgroundColor': 'rgb(220, 220, 220)', 'fontWeight': 'bold', 'textAlign': 'center'},
        style_cell={
            'textAlign': 'center', 'padding': '8px',
            'minWidth': '100px', 'width': '120px', 'maxWidth': '180px',
            'whiteSpace': 'normal', 'height': 'auto',
        },
        style_cell_conditional=[
            {'if': {'column_id': 'Komponente'},
             'textAlign': 'left',  # Keep text aligned left for readability
             'minWidth': '180px', 'width': '250px', 'maxWidth': '400px'}
        ],
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}
        ]
    )
    return html.Div([
        dbc.Alert(id={"type": "i18n", "key": "planung.material_consumption_created"}, color="success", duration=2000),
        html.H4(id={"type": "i18n", "key": "planung.weekly_material_demand_title"}, className="mt-4 mb-3"),
        html.P(
             id={"type": "i18n", "key": "planung.material_consumption_info"},
            className="text-muted small mb-3"
        ),
        material_table,
        dbc.Row([
            dbc.Col(
                dbc.Button(forecast_button_label, id="btn-calculate-forecast", color="success",
                           className="mt-3", n_clicks=0),
                width={"size": "auto", "offset": 0}
            )
        ])
    ])



@callback(
    Output("forecasted-material-output", "children"),
    Output("all-forecasts-storage", "data"),
    Output("planned-data-storage", "data"),
    Input("btn-calculate-forecast", "n_clicks"),
    State("material-distribution-table", "data"),
    prevent_initial_call=True
)
def process_and_store_all_data(
        n_clicks: Optional[int],
        material_data: Optional[List[Dict[str, Any]]],
) -> Tuple[Union[dbc.Alert, html.Div], Optional[Dict], Optional[Dict]]:
    """
    Calculates all forecast scenarios and extracts planned data,
    storing both in session storage for the table-rendering callback.
    """
    if not n_clicks or n_clicks == 0:
        raise PreventUpdate

    if not material_data:
        return dbc.Alert(id={"type": "i18n", "key": "planung.no_material_forecast_data"}, color="warning"), no_update, no_update
    komponente = fn.outside_translate("series.col_komponente")
    try:
        # Extract planned data into a clean dictionary {component: {week: value}}
        planned_data_dict = {}
        for row in material_data:
            comp_id = row[komponente].split(']')[0].lstrip('[')
            planned_data_dict[comp_id] = {k: v for k, v in row.items() if k not in [komponente, 'Total']}

        komponenten_ids = list(planned_data_dict.keys())

        # Calculate all forecast scenarios
        all_forecasts = {'normal': {}, 'lower': {}, 'upper': {}}
        for item_code in komponenten_ids:
            try:
                X_train, future_dates = fn.get_data(item_code)
                predictions, lower_bounds, upper_bounds, _ = fn.recursive_prediction_conformal_darts(
                    item_code, X_train, future_dates
                )

                scenarios = {'normal': predictions, 'lower': zip(future_dates, lower_bounds),
                             'upper': zip(future_dates, upper_bounds)}
                for name, points in scenarios.items():
                    all_forecasts[name][item_code] = {date.strftime("%Y-W%V"): math.ceil(val) if val > 0 else 0 for
                                                      date, val in points}

            except Exception as e:
                logging.error(f"Fehler bei der Prognose für Komponente {item_code}: {e}")
                for name in all_forecasts:
                    all_forecasts[name][item_code] = {}  # Empty dict for failed forecasts

        normal_forecast_label = fn.outside_translate("planung.normal_forecast")
        lower_forecast_label = fn.outside_translate("planung.lower_forecast")
        upper_forecast_label = fn.outside_translate("planung.upper_forecast")
        # Create the UI shell
        scenario_selector = dbc.RadioItems(
            id="forecast-scenario-selector",
            options=[
                {'label': normal_forecast_label, 'value': 'normal'},
                {'label': lower_forecast_label, 'value': 'lower'},
                {'label': upper_forecast_label, 'value': 'upper'},
            ],
            value='normal', inline=True, className="mb-3"
        )

        output_div = html.Div([
            dbc.Alert( id={"type": "i18n", "key": "planung.comparison_table_calculated"}, color="success", duration=2000),
            html.H4( id={"type": "i18n", "key": "planung.comparison_heading"}, className="mt-4 mb-3"),
            html.P(id={"type": "i18n", "key": "planung.select_forecast_variant"},
                   className="text-muted small"),
            scenario_selector,
            html.Div(id='merged-table-container')  # Placeholder for the merged table
        ])

        return output_div, all_forecasts, planned_data_dict

    except Exception as e:
        logging.error(f"Genereller Fehler in process_and_store_all_data: {e}")
        return dbc.Alert(id={"type": "i18n", "key": "planung.unexpected_error"}, color="danger"), no_update, no_update



@callback(
    Output('merged-table-container', 'children'),
    Input('forecast-scenario-selector', 'value'),
    Input('all-forecasts-storage', 'data'),
    Input('planned-data-storage', 'data'),
    prevent_initial_call=True
)
def generate_and_display_merged_table(
        selected_scenario: str,
        all_forecasts: Optional[Dict],
        planned_data: Optional[Dict]
) -> List[Union[dbc.Alert, dash_table.DataTable, dcc.Graph]]:
    """
    Builds and displays the merged table and a comparison bar chart
    based on the selected forecast scenario.
    """
    if not all_forecasts or not planned_data or not selected_scenario:
        raise PreventUpdate

    forecast_for_scenario = all_forecasts.get(selected_scenario, {})

    # Get the list of weeks from the first component in the planned data
    planned_weeks = sorted(list(next(iter(planned_data.values())).keys()))

    #  Build the merged data records
    table_records = []
    for comp_id, weekly_planned_values in planned_data.items():
        row = {'Komponente': comp_id}
        total_planned = 0
        total_forecast = 0

        comp_forecasts = forecast_for_scenario.get(comp_id, {})

        for week in planned_weeks:
            planned_val = weekly_planned_values.get(week, 0)
            forecast_val = comp_forecasts.get(week, 0)

            row[f'{week}_planned'] = planned_val
            row[f'{week}_forecast'] = forecast_val
            total_planned += planned_val
            total_forecast += forecast_val

        row['total_planned'] = total_planned
        row['total_forecast'] = total_forecast
        table_records.append(row)
    komponente = fn.outside_translate("series.col_komponente")
    geplant = fn.outside_translate("planung.col_geplant")
    prognose = fn.outside_translate("planung.col_prognose")
    title_text = fn.outside_translate("planung.merged_table_title")
    yaxis_title = fn.outside_translate("planung.yaxis_total_quantity")
    legend_title = fn.outside_translate("planung.legend_title")

    #  Define multi-header columns
    columns = [{'name': ['', komponente], 'id': 'Komponente'}]
    for week in planned_weeks:
        columns.append({'name': [week, geplant], 'id': f'{week}_planned'})
        columns.append({'name': [week, prognose], 'id': f'{week}_forecast'})

    columns.append({'name': ['Total', geplant], 'id': 'total_planned'})
    columns.append({'name': ['Total', prognose], 'id': 'total_forecast'})

    # Create the DataTable
    merged_table = dash_table.DataTable(
        id='merged-forecast-table',
        columns=columns,
        data=table_records,
        merge_duplicate_headers=True,
        sort_action="native",
        filter_action="native",
        page_size=10,
        style_table={'overflowX': 'auto', 'minWidth': '100%'},
        style_header={'backgroundColor': 'rgb(220, 220, 220)', 'fontWeight': 'bold', 'textAlign': 'center'},
        style_cell={'textAlign': 'center', 'padding': '8px', 'minWidth': '80px', 'width': '100px', 'maxWidth': '120px'},
        style_cell_conditional=[
            {'if': {'column_id': 'Komponente'}, 'textAlign': 'left', 'minWidth': '180px', 'width': '250px',
             'maxWidth': '400px'}
        ]
    )

    # Create the Comparison Bar Chart
    df_plot = pd.DataFrame(table_records)
    # Ensure totals are numeric for plotting, ignoring any failed forecasts
    df_plot['total_planned'] = pd.to_numeric(df_plot['total_planned'], errors='coerce')
    df_plot['total_forecast'] = pd.to_numeric(df_plot['total_forecast'], errors='coerce')
    df_plot.dropna(subset=['total_planned', 'total_forecast'], inplace=True)
    df_plot.rename(columns={'Komponente': komponente}, inplace=True)
    print(df_plot.columns)
    fig = go.Figure(data=[
        go.Bar(name=geplant, x=df_plot[komponente], y=df_plot['total_planned'], marker_color='blue'),
        go.Bar(name=prognose, x=df_plot[komponente], y=df_plot['total_forecast'], marker_color='green')
    ])

    fig.update_layout(
        barmode='group',
        title_text=title_text,
        xaxis_title=komponente,
        yaxis_title= yaxis_title,
        legend_title_text=legend_title,
        template='plotly_white'
    )

    comparison_graph = dcc.Graph(figure=fig, className="mt-4")

    # Return a list containing both the table and the graph
    return [merged_table, comparison_graph]