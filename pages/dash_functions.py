from datetime import datetime, timedelta # Needed for date calculations
import dash
from dash import html, dcc, callback, Output, Input
import dash.exceptions
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from . import functions as fn
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Any, Tuple, Optional

def add_fig_trace(fig,x,y,mode,name,line,fill=None):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode=mode,
            name=name,
            line=line,
            fill=fill,
        )
    )

def create_table_from_dataframe(df: pd.DataFrame) -> dbc.Table:
    """Generates a Bootstrap styled table from a pandas DataFrame."""
    if df.empty: return dbc.Table()
    num_columns: int = len(df.columns)
    if num_columns == 0: return dbc.Table()
    col_width_percent: float = 100 / num_columns
    cell_style: Dict[str, str] = {
        'width': f'{col_width_percent}%', 'minWidth': f'{col_width_percent}%',
        'maxWidth': f'{col_width_percent}%', 'textAlign': 'center', 'wordWrap': 'break-word'
    }
    headers: List[html.Th] = [html.Th(col, style=cell_style) for col in df.columns]
    thead: html.Thead = html.Thead(html.Tr(headers))
    rows: List[html.Tr] = [html.Tr([html.Td(row[col], style=cell_style) for col in df.columns]) for _, row in df.iterrows()]
    tbody: html.Tbody = html.Tbody(rows)
    return dbc.Table([thead, tbody], bordered=True, striped=True, hover=True, responsive=True, style={'tableLayout': 'fixed', 'width': '100%'})

def create_bar_chart_figure(config_df: pd.DataFrame, title: str) -> go.Figure:
    """Helper function to create the bar chart figure."""
    req_quant = fn.outside_translate("dash_functions.required_quantity")
    prod = fn.outside_translate("dash_functions.product")
    item_code = fn.outside_translate("dash_functions.item_code")
    err_title = fn.outside_translate("dash_functions.error_bar_title")
    try:
        fig = px.bar(config_df, x=fn.ARTICLE_NUMBER_COLUMN, y='RequiredQuantity', color='Product', title=title,
                     labels={fn.ARTICLE_NUMBER_COLUMN: item_code, 'RequiredQuantity': req_quant, 'Product': prod},
                     barmode='group')
        fig.update_xaxes(categoryorder='total descending')
        logging.info("Successfully generated bar chart.")
        return fig
    except Exception as e:
        logging.error(f"Error creating plotly bar chart: {e}", exc_info=True)
        return go.Figure(layout={"title": err_title})

def create_combined_radar_chart(config_df: pd.DataFrame, selected_products: List[str]) -> go.Figure:
    """Helper function to create a single radar chart with multiple traces."""
    no_sel_text = fn.outside_translate("dash_functions.no_selection")
    title_text = fn.outside_translate("dash_functions.title")
    hover_tmplt = fn.outside_translate("dash_functions.hover_product")
    err_title = fn.outside_translate("dash_functions.error_title_combined")

    if not selected_products: return go.Figure(layout={"title": no_sel_text})
    try:
        pivoted_df = config_df.pivot(index=fn.ARTICLE_NUMBER_COLUMN, columns='Product', values='RequiredQuantity').fillna(0)
        for prod in selected_products:
            if prod not in pivoted_df.columns: pivoted_df[prod] = 0
        pivoted_df.sort_index(inplace=True)
        theta_values_base = pivoted_df.index.tolist()
        fig = go.Figure()
        for product_id in selected_products:
            r_values_base = pivoted_df[product_id].tolist()
            r_values = r_values_base + [r_values_base[0]] if r_values_base else []
            theta_values = theta_values_base + [theta_values_base[0]] if theta_values_base else []
            hovertext = hover_tmplt.format(product=product_id, item="%{theta}", qty="%{r}")
            fig.add_trace(go.Scatterpolar(r=r_values, theta=theta_values, fill='toself', name=product_id,
                                          hovertemplate=hovertext + "<extra></extra>"))
        max_val = config_df['RequiredQuantity'].max() * 1.1 if not config_df.empty else 1
        fig.update_layout(title_text=title_text,
                          polar=dict(radialaxis=dict(visible=True, range=[0, max_val])),
                          height=600, showlegend=True)
        logging.info("Successfully generated single radar chart.")
        return fig
    except Exception as e:
        logging.error(f"Error creating single radar chart: {e}", exc_info=True)
        return go.Figure(layout={"title": err_title})

def create_separate_radar_charts(config_df: pd.DataFrame, selected_products: List[str]) -> go.Figure:
    """Helper function to create separate radar charts using subplots."""

    no_sel = fn.outside_translate("dash_functions.no_selection")
    no_data = fn.outside_translate("dash_functions.no_data")
    too_many = fn.outside_translate("dash_functions.too_many")
    hover_tmpl = fn.outside_translate("dash_functions.hover_item_qty")
    title_txt = fn.outside_translate("dash_functions.title")
    err_title = fn.outside_translate("dash_functions.error_title_separate")

    if not selected_products: return go.Figure(layout={"title": no_sel})
    num_products = len(selected_products)
    if num_products == 0: return go.Figure(layout={"title": no_data})
    if num_products > 2: return go.Figure(layout={"title": too_many})

    rows = 1
    cols = num_products
    fig = make_subplots(rows=rows, cols=cols, specs=[[{'type': 'polar'}] * cols])

    try:
        max_val = config_df['RequiredQuantity'].max() * 1.1 if not config_df.empty else 1

        for i, product_id in enumerate(selected_products):
            product_data = config_df[config_df['Product'] == product_id]
            r_values, theta_values = [], []
            if not product_data.empty:
                product_data = product_data.sort_values(by=fn.ARTICLE_NUMBER_COLUMN)
                r_values = list(product_data['RequiredQuantity']) + [product_data['RequiredQuantity'].iloc[0]]
                theta_values = list(product_data[fn.ARTICLE_NUMBER_COLUMN]) + [product_data[fn.ARTICLE_NUMBER_COLUMN].iloc[0]]
            else:
                 logging.info(f"No data for radar chart for product {product_id}")
            hover_tmpl = hover_tmpl.format(theta="%{theta}", r="%{r}")
            fig.add_trace(go.Scatterpolar(r=r_values, theta=theta_values, fill='toself', name=product_id,
                                          hovertemplate=hover_tmpl + "<extra></extra>"),
                          row=1, col=i + 1)

        x_positions = []
        if num_products == 1:
            x_positions = [0.5]
        elif num_products == 2:
            domain1 = fig.layout.polar.domain.x
            domain2 = fig.layout.polar2.domain.x
            x_positions = [(domain1[0] + domain1[1]) / 2, (domain2[0] + domain2[1]) / 2]

        annotations = []
        for i, product_id in enumerate(selected_products):
            annotations.append(
                dict(
                    text=f"<b>Product: {product_id}</b>",
                    x=x_positions[i],
                    y=1.08,  # Increased Y value further
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14, color="black"),
                    xanchor="center",
                    yanchor="bottom"
                )
            )

        fig.update_layout(
            title_text=title_txt,
            height=600,
            showlegend=False,
            margin=dict(l=50, r=50, t=160, b=50),
            annotations=annotations
        )

        for i in range(1, num_products + 1):
            polar_key = f'polar{i if i > 1 else ""}'
            fig.layout[polar_key].angularaxis.showticklabels = True
            fig.layout[polar_key].radialaxis.angle = 90
            fig.layout[polar_key].radialaxis.showline = True
            fig.layout[polar_key].radialaxis.showticklabels = True
            fig.layout[polar_key].radialaxis.range = [0, max_val]

        logging.info("Successfully generated separate radar charts.")
        return fig
    except Exception as e:
        logging.error(f"Error creating separate radar charts: {e}", exc_info=True)
        return go.Figure(layout={"title": err_title})


def get_weeks_in_range_planning(start_date_str: Optional[str], end_date_str: Optional[str]) -> List[str]:
    """
    Generates a sorted list of unique weeks (YYYY-Www) within a given date range.
    """
    if not start_date_str or not end_date_str:
        return []

    try:
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError:
        logging.error(f"Invalid date format received in get_weeks_in_range_planning: {start_date_str}, {end_date_str}")
        return []

    if start_dt > end_dt:
        return []

    weeks_list = []
    current_dt = start_dt
    while current_dt <= end_dt:
        year, week_num, _ = current_dt.isocalendar()  # (ISO year, ISO week number, ISO weekday)
        week_str = f"{year}-W{week_num:02d}"  # Format as YYYY-Www
        if week_str not in weeks_list:
            weeks_list.append(week_str)
        current_dt += timedelta(days=1)

    weeks_list.sort()  # Ensures chronological order
    return weeks_list

def recalculate_product_totals(
    product_data: List[Dict[str, Any]],
    product_table_columns: List[Dict[str, str]],
    week_column_ids_param: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], bool]:
    """Helper to recalculate TotalArtikel for product data."""
    baureihe = fn.outside_translate("planung.baureihe")
    produktNr = fn.outside_translate("planung.produktNr")
    totalArtikel = fn.outside_translate("planung.totalArtikel")
    if week_column_ids_param:
        week_column_ids = week_column_ids_param
    else:
        week_column_ids = [
            col['id'] for col in product_table_columns
            if col['id'] not in [baureihe, produktNr, totalArtikel]
        ]

    recalculated_data = []
    any_total_changed = False
    for row in product_data:
        new_row_copy = row.copy()
        current_row_total_artikel = 0.0
        for week_col_id in week_column_ids:
            try:
                cell_value = new_row_copy.get(week_col_id)
                if cell_value is None or str(cell_value).strip() == "":
                    val_for_sum = 0.0
                else:
                    val_for_sum = float(str(cell_value))
                current_row_total_artikel += val_for_sum
            except (ValueError, TypeError):
                pass
        new_total_artikel = round(current_row_total_artikel, 4)
        if new_row_copy.get(totalArtikel) != new_total_artikel:
            any_total_changed = True
        new_row_copy[totalArtikel] = new_total_artikel
        recalculated_data.append(new_row_copy)
    return recalculated_data, any_total_changed



def final_updated_data_differs(original_data, new_data):
    # Simple helper to check if data actually changed, useful if only whitespace or type changes occurred
    return original_data != new_data



