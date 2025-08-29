#forecast.py
import pandas as pd
import holidays
from collections import Counter
from sklearn.metrics import mean_absolute_error,mean_squared_error, root_mean_squared_error
import logging
# from forecastobjects import BenchmarkStats, ScoreConfig
import matplotlib.pyplot as plt
import numpy as np
from darts.models import XGBModel, ConformalQRModel  # Darts for probabilistic forecasting
from darts import TimeSeries
import seaborn as sns
from darts.models import XGBModel
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pulp
from pandas import DataFrame
import json
import re
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
import math
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
from typing import Dict, List, Tuple, Union, Optional, Any
from config import settings
dir = os.getcwd()
warnings.filterwarnings("ignore")


dir = os.getcwd()
warnings.filterwarnings("ignore")
# For different datasets and column names please adjust the configs accordingly.
# === File Paths ===
POSTEN_FILTERED_FILE: str = settings.get('POSTEN_FILTERED_FILE')              # Path to the main filtered transaction data file.
CACHED_INVENTORY_FILE: str = settings.get('CACHED_INVENTORY_FILE')             # Path to the cached inventory data for faster loading.
ARTIKEL_HERKUNFTSDATEN: str = settings.get('ARTIKEL_HERKUNFTSDATEN')            # Path to the file with article origin country data.
BR_MAPS: str = settings.get('BR_MAPS')                                       # Path to the file mapping articles to product series (Baureihe).
PLOT_FOLDER: str = settings.get('PLOT_FOLDER')                                 # Directory for saving generated plots.
SAVE_DIR_FOLDER: str = settings.get('SAVE_DIR_FOLDER')                         # Directory for saving trained machine learning models.

# === Column Names & Constants ===
# --- General Data Columns ---
DATE_COLUMN: str = settings.get('DATE_COLUMN')                                 # Column name for the transaction date.
QUANTITY_COLUMN: str = settings.get('QUANTITY_COLUMN')                         # Column name for the transaction quantity.
MENGE_COLUMN: str = settings.get('MENGE_COLUMN')                               # Target column name for quantity, used in forecasts.
SOURCE_NUMBER_COLUMN: str = settings.get('SOURCE_NUMBER_COLUMN')               # Column for the source identifier (e.g., production order).
CUMMULATIVE_COLUMN: str = settings.get('CUMMULATIVE_COLUMN')                   # Column name for calculated cumulative quantities.
ARTICLE_NUMBER_COLUMN: str = settings.get('ARTICLE_NUMBER_COLUMN')             # Column name for the unique article number.
POSTEN_TYPE_COLUMN: str = settings.get('POSTEN_TYPE_COLUMN')                   # Column defining the transaction type.

# --- Posten Type Constants ---
SALES_POSTEN_TYPE: str = settings.get('SALES_POSTEN_TYPE')                     # Value representing a 'Sales' transaction.
CONSUMPTION_POSTEN_TYPE: str = settings.get('CONSUMPTION_POSTEN_TYPE')         # Value for a 'Consumption' transaction (e.g., used in production).
ABGANG_POSTEN_TYPE: str = settings.get('ABGANG_POSTEN_TYPE')                   # Value for an 'Outflow' or stock removal transaction.
ZUGANG_POSTEN_TYPE: str = settings.get('ZUGANG_POSTEN_TYPE')                   # Value for an 'Inflow' or stock addition transaction.
EINKAUF_POSTEN_TYPE: str = settings.get('EINKAUF_POSTEN_TYPE')                 # Value for a 'Purchase' transaction.
ISTMELDUNG_POSTEN_TYPE: str = settings.get('ISTMELDUNG_POSTEN_TYPE')           # Value for an 'Actuals Reporting' transaction.

# --- Inventory Forecast Columns ---
LOWER_BOUND_COLUMN: str = settings.get('LOWER_BOUND_COLUMN')                   # Column for the conservative (lower bound) forecast.
UPPER_BOUND_COLUMN: str = settings.get('UPPER_BOUND_COLUMN')                   # Column for the optimistic (upper bound) forecast.
PREDICTED_COLUMN: str = settings.get('PREDICTED_COLUMN')                       # Column for the standard (median) forecast.
PREDICTED_DATE_COLUMN: str = settings.get('PREDICTED_DATE_COLUMN')             # Column for the date of a forecast prediction.
PREDICTED_TRUE_COLUMN: str = settings.get('PREDICTED_TRUE_COLUMN')             # Column for the actual value to compare against a forecast.

# --- Item Info & Simulation Columns ---
ITEM_CODE_COLUMN: str = settings.get('ITEM_CODE_COLUMN')                       # Generic column name for a unique item identifier.
ORDER_COLUMN: str = settings.get('ORDER_COLUMN')                               # Column for the normal calculated order quantity.
WORSE_CASE_ORDER_COLUMN: str = settings.get('WORSE_CASE_ORDER_COLUMN')         # Column for the order quantity in the minimum demand scenario.
BEST_CASE_ORDER_COLUMN: str = settings.get('BEST_CASE_ORDER_COLUMN')           # Column for the order quantity in the maximum demand scenario.
PREDICTED_DEMAND_COLUMN: str = settings.get('PREDICTED_DEMAND_COLUMN')         # Column for a predicted demand value.
PREDECTED_STOCK_COLUMN: str = settings.get('PREDECTED_STOCK_COLUMN')           # Column for a predicted stock level.
REORDER_DATE_COLUMN: str = settings.get('REORDER_DATE_COLUMN')                 # Column for the calculated date to place a new order.
REORDER_MONTH_COLUMN: str = settings.get('REORDER_MONTH_COLUMN')               # Column for the month of a reorder action.
REORDER_POINT_COLUMN: str = settings.get('REORDER_POINT_COLUMN')               # The inventory level that triggers a reorder.
INVENTORY_LEVEL_COLUMN: str = settings.get('INVENTORY_LEVEL_COLUMN')           # General column name for the inventory level.
STOCK_LEVEL_AT_MONTH_START_COLUMN: str = settings.get('STOCK_LEVEL_AT_MONTH_START_COLUMN') # Stock level at the beginning of a simulation period.
ARRIVING_STOCK_COLUMN: str = settings.get('ARRIVING_STOCK_COLUMN')             # Quantity of stock arriving in a period.
STOCK_LEVEL_AFTER_ARRIVAL_COLUMN: str = settings.get('STOCK_LEVEL_AFTER_ARRIVAL_COLUMN') # Stock level after new arrivals.
EXPECTED_DEMAND_COLUMN: str = settings.get('EXPECTED_DEMAND_COLUMN')           # Expected demand for a period.
STOCK_LEVEL_AFTER_DEMAND_COLUMN: str = settings.get('STOCK_LEVEL_AFTER_DEMAND_COLUMN') # Stock level after demand is fulfilled.
SHORTFALL_COLUMN: str = settings.get('SHORTFALL_COLUMN')                       # The amount of demand that could not be met.
ORDER_DECISION_COLUMN: str = settings.get('ORDER_DECISION_COLUMN')             # Column indicating if an order was placed (e.g., 'Yes'/'No').
OREDER_QUANTITY_COLUMN: str = settings.get('OREDER_QUANTITY_COLUMN')           # The actual quantity ordered in a period.
EXPECTED_STOCK_ARRIVAL_COLUMN: str = settings.get('EXPECTED_STOCK_ARRIVAL_COLUMN') # The expected arrival date for an order.
STOCK_ON_ORDER_COLUMN: str = settings.get('STOCK_ON_ORDER_COLUMN')             # The total quantity of stock currently on order.
REASONING_COLUMN: str = settings.get('REASONING_COLUMN')                       # An explanation for why an order decision was made.
LEAD_TIME: str = settings.get('LEAD_TIME')                                     # The time between ordering and receiving an item.
NEEDED: str = settings.get('NEEDED')                                           # The calculated quantity needed to reach a target stock level.
ORDERED: str = settings.get('ORDERED')                                         # The final quantity ordered after applying constraints.

# --- Baureihe (Product Series) Mapping Columns ---
BAUREHIE_ITEM_NUMBER_COLUMN: str = settings.get('BAUREHIE_ITEM_NUMBER_COLUMN') # Column for the article number in the Baureihe mapping file.
BAUREHIE_COLUMN: str = settings.get('BAUREHIE_COLUMN')                         # Column for the name of the product series.
MELDEBESTAND_NUMBER_COLUMN: str = settings.get('MELDEBESTAND_NUMBER_COLUMN')   # Column for the article number in the safety stock file.

# --- Inventory Data Columns ---
INVERTORY_DATA_SHEET_NAME: str = settings.get('INVERTORY_DATA_SHEET_NAME')      # The name of the Excel sheet containing inventory data.
INVENTORY_DATA_ARTICLE_NUMBER_COLUMN: str = settings.get('INVENTORY_DATA_ARTICLE_NUMBER_COLUMN') # Column for the article number in the main inventory file.
INVENTORY_DATA_SAFETY_STOCK_COLUMN: str = settings.get('INVENTORY_DATA_SAFETY_STOCK_COLUMN') # Column for the safety stock level.
INVENTORY_DATA_SAFETY_LEAD_TIME_COLUMN: str = settings.get('INVENTORY_DATA_SAFETY_LEAD_TIME_COLUMN') # Column for the safety lead time.
INVENTORY_DATA_MAX_INVENTORY_COLUMN: str = settings.get('INVENTORY_DATA_MAX_INVENTORY_COLUMN') # Column for the maximum allowed inventory.
INVENTORY_DATA_LEAD_TIME_COLUMN: str = settings.get('INVENTORY_DATA_LEAD_TIME_COLUMN') # Column for the standard procurement lead time.
INVENTORY_DATA_MIN_ORDER_QTY_COLUMN: str = settings.get('INVENTORY_DATA_MIN_ORDER_QTY_COLUMN') # Column for the minimum order quantity.
INVENTORY_DATA_MAX_ORDER_QTY_COLUMN: str = settings.get('INVENTORY_DATA_MAX_ORDER_QTY_COLUMN') # Column for the maximum order quantity.

# === Miscellaneous ===
TRAIN_TEST_SPLIT: float = settings.get('TRAIN_TEST_SPLIT')                     # Ratio to split data into training and testing sets for ML.
DATE_FORMAT: str = settings.get('DATE_FORMAT')                                 # Default string format for dates (e.g., '%Y-%m-%d').
MONTH_FORMAT: str = settings.get('MONTH_FORMAT')                               # Default string format for months (e.g., '%m-%Y').
CURRENT_MONTH: int = settings.get('CURRENT_MONTH')                             # The reference month for starting simulations.
CURRENT_YEAR: int = settings.get('CURRENT_YEAR')                               # The reference year for starting simulations.
PROD_START:str = settings.get('PROD_START')                                     # The prefix used to identify finished product numbers
ITEM_START:str = settings.get('ITEM_START')                                     # The prefix for semi-finished items or components
PART_START:str = settings.get('PART_START')                                     # The prefix for raw parts or materials.
# === Scenario Explainations ===
COLUMN_EXPLANATIONS = {
    "Month": "",
    "Stock Level @ Start": "",
    "Arriving Stock Quantity": "",
    "Stock Level After Arrival of Stocks": "",
    "Expected Demand Quantity": "",
    "Stock Level After Demand Consumption": "",
    "Shortfall": "",
    "Order Trigger Level": "",
    "Inventory Level @ End": "",
    "Order Decision": "",
    "Order Quantity": "",
    "Expected Stock Arrival": "",
    "Stock On Order @ End": "",
    "Reasoning For Order Decision": ""
}

with open("translations/de.json", encoding="utf-8") as file:
    DE = json.load(file)

with open("translations/en.json", encoding="utf-8") as file:
    EN = json.load(file)

TRANSLATIONS = {"en": EN, "de": DE}

_language = {'language': 'de'}



@dataclass
class InventoryInfo:
    safety_stock: int
    safety_lead_time: str
    max_inventory: int
    lead_time: str
    min_order_qty: int
    max_order_qty: int


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def load_posten_file ():
    """
    Loads and processes the posten file by reading a CSV file and converting
    its date column to a pandas datetime object. The data is then returned
    as a pandas DataFrame.
    :return: Processed data read from the posten CSV file, where the date column
             is converted to datetime.
    :rtype: pandas.DataFrame
    """
    df = pd.read_csv(POSTEN_FILTERED_FILE)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df
def get_product_list ():
    """
    Retrieves and processes a list of product identifiers from a source file.
    The function filters product entries whose source identifiers begin with PROD_START
    and excludes entries that do not meet this condition. It then extracts unique
    source identifiers, sorts them, and returns the sorted list.

    :return: A sorted list containing unique product identifiers that begin with PROD_START
    :rtype: numpy.ndarray
    """
    prods = load_posten_file()
    prods = prods.loc[prods[SOURCE_NUMBER_COLUMN].str.startswith(PROD_START, na=False)]
    products = prods[SOURCE_NUMBER_COLUMN].unique()
    sorted_prods = np.sort(products)
    sorted_prods = sorted_prods.tolist()
    return sorted_prods

def get_demand_data(item_id: str) -> pd.DataFrame:
    """
    Retrieves and processes demand data for a specific product from a CSV file.
    This function reads demand data from a CSV file, filters it by the provided
    product number, cleans and transforms the data, and returns the demand
    aggregated by week.

    Args:
        item_id: The item number ARTICLE_NUMBER_COLUMN) to filter the demand data for.

    Returns:
        A pandas DataFrame containing the weekly aggregated demand for the
        specified item, with DATE_COLUMN as the index and 'Menge' as the
        summed demand quantity.
    """
    file_name: str = POSTEN_FILTERED_FILE  # Path to the demand data CSV file.
    demand: pd.DataFrame = pd.read_csv(file_name)  # Read the CSV file into a pandas DataFrame.

    demand[DATE_COLUMN] = pd.to_datetime(demand[DATE_COLUMN])  # Convert the DATE_COLUMN column to datetime objects.

    # Filter by product_no and ensure it's a copy to avoid SettingWithCopyWarning.
    prod_demand: pd.DataFrame = demand.loc[demand[ARTICLE_NUMBER_COLUMN] == item_id].copy()
    prod_demand = prod_demand.loc[
        prod_demand[POSTEN_TYPE_COLUMN].isin([CONSUMPTION_POSTEN_TYPE, ABGANG_POSTEN_TYPE, SALES_POSTEN_TYPE])]

    # Only get negative instances which represent demand
    prod_demand = prod_demand[prod_demand[QUANTITY_COLUMN] < 0]

    # Drop unnecessary columns and convert QUANTITY_COLUMN to absolute integers.
    prod_demand = prod_demand.drop(columns=[ARTICLE_NUMBER_COLUMN, POSTEN_TYPE_COLUMN])
    prod_demand[QUANTITY_COLUMN] = prod_demand[QUANTITY_COLUMN].abs().astype(float)

    # Group demand by DATE_COLUMN and sum the QUANTITY_COLUMN for each date.
    prod_demand_grouped: pd.DataFrame = prod_demand.groupby(DATE_COLUMN, as_index=True).agg({QUANTITY_COLUMN: 'sum'})

    # Resample to weekly frequency and sum the QUANTITY_COLUMN for each week.
    prod_demand_weekly: pd.DataFrame = prod_demand_grouped.resample('W').sum()

    # Reset the index to make DATE_COLUMN a regular column.
    prod_demand_weekly = prod_demand_weekly.reset_index()
    prod_demand_weekly.rename(columns={QUANTITY_COLUMN: MENGE_COLUMN}, inplace=True)

    return prod_demand_weekly  # Return the DataFrame with weekly aggregated demand.


def get_stock_data(product_no: str):
    """
    Retrieves and processes stock movement for a given product number.
    :param product_no: The product number for which stock data will be retrieved.
    :type product_no: str
    :return: A tuple where the first element is a DataFrame containing grouped and cumulative stock movement data by
             date, including cumulative sums, and the second element is the total accumulated quantity
             of the product.
    :rtype: tuple[pandas.DataFrame, float]
    """
    file_name: str = POSTEN_FILTERED_FILE  # Path to the demand data CSV file.
    posten: pd.DataFrame = pd.read_csv(file_name)  # Read the CSV file into a pandas DataFrame.

    posten[DATE_COLUMN] = pd.to_datetime(posten[DATE_COLUMN])  # Convert dates.

    # Filter by product and make a copy.
    prod_stock: pd.DataFrame = posten.loc[posten[ARTICLE_NUMBER_COLUMN] == product_no].copy()

    # Drop irrelevant settings.
    prod_stock = prod_stock.drop(columns=[ARTICLE_NUMBER_COLUMN, POSTEN_TYPE_COLUMN])

    # Aggregate quantity by date.
    prod_stock_grouped: pd.DataFrame = prod_stock.groupby(DATE_COLUMN, as_index=True).agg({QUANTITY_COLUMN: 'sum'})
    prod_stock_grouped.rename(columns={QUANTITY_COLUMN: MENGE_COLUMN}, inplace=True)

    # Compute cumulative stock.
    prod_stock_grouped[CUMMULATIVE_COLUMN] = prod_stock_grouped[MENGE_COLUMN].cumsum()

    # Reset index for output.
    prod_stock_grouped = prod_stock_grouped.reset_index()

    # Get final cumulative stock.
    final_stock: int = int(round(prod_stock_grouped.tail(1)[CUMMULATIVE_COLUMN]))

    return prod_stock_grouped, final_stock


def get_product_sales(product_id: str) -> pd.DataFrame:
    """
    Retrieve the sales data of a specific product.

    :param product_id: A string representing the unique identifier of the product for
        which the demand data is requested.
    :type product_id: str
    :return: A Pandas DataFrame containing the demand details of the specified product.
        The DataFrame may include columns related to demand metrics like quantity,
        timestamps, and other relevant details.
    :rtype: pd.DataFrame
    """
    file_name: str = POSTEN_FILTERED_FILE  # Path to the demand data CSV file.
    sale: pd.DataFrame = pd.read_csv(file_name)  # Read the CSV file into a pandas DataFrame.

    sale[DATE_COLUMN] = pd.to_datetime(sale[DATE_COLUMN])  # Convert the DATE_COLUMN column to datetime objects.

    # Filter by product_no and ensure it's a copy to avoid SettingWithCopyWarning.
    prod_demand: pd.DataFrame = sale.loc[sale[ARTICLE_NUMBER_COLUMN] == product_id].copy()
    prod_demand = prod_demand.loc[prod_demand[POSTEN_TYPE_COLUMN].isin([SALES_POSTEN_TYPE])]

    # Only get negative instances which represent demand
    prod_demand = prod_demand[prod_demand[QUANTITY_COLUMN] < 0]

    # Drop unnecessary columns and convert QUANTITY_COLUMN to absolute integers.
    prod_demand = prod_demand.drop(columns=[ARTICLE_NUMBER_COLUMN, POSTEN_TYPE_COLUMN])
    prod_demand[QUANTITY_COLUMN] = prod_demand[QUANTITY_COLUMN].abs().astype(float)

    # Group demand by DATE_COLUMN and sum the 'Menge' for each date.
    prod_demand_grouped: pd.DataFrame = prod_demand.groupby(DATE_COLUMN, as_index=True).agg({QUANTITY_COLUMN: 'sum'})

    # Resample to weekly frequency and sum the 'Menge' for each week.
    prod_demand_weekly: pd.DataFrame = prod_demand_grouped.resample('W').sum()

    # Reset the index to make DATE_COLUMN a regular column.
    prod_demand_weekly = prod_demand_weekly.reset_index()
    prod_demand_weekly = prod_demand_weekly.rename(columns={QUANTITY_COLUMN: MENGE_COLUMN})

    return prod_demand_weekly  # Return the DataFrame with wekly aggregated sales data of a product.


def get_item_mapping(article_id: str) -> List[str]:
    """
    Retrieves the unique source number(s) (Product mapping) corresponding to the given item id.
    We filter out only PROD_START items.
    Function returns a list of end products in which the item was used.

    :param article_id: The identifier of the article to search for.
    :type article_id: str
    :return: A list of unique source number(s) (end products) related to the given item id.
    """
    posten: pd.DataFrame = pd.read_csv(POSTEN_FILTERED_FILE)
    posten = posten.loc[posten[POSTEN_TYPE_COLUMN] == CONSUMPTION_POSTEN_TYPE]
    item: pd.DataFrame = posten.loc[posten[ARTICLE_NUMBER_COLUMN] == article_id]
    source: pd.DataFrame = item.loc[item[SOURCE_NUMBER_COLUMN].str.startswith(PROD_START)]
    return source[SOURCE_NUMBER_COLUMN].unique().tolist()


def get_baureihe_mapping(product_id: str) -> str:
    """
    Gets the Baureihe mapping for a given product id.
    This function reads a mapping CSV file, filters it by the product identifier provided as input, and retrieves the corresponding Baureihe.

    :param product_id: product id (ie. '1-00056')
    :type product_id: str
    :return: The Baureihe corresponding to the provided product identifier.
    :rtype: str
    """
    baureihen: pd.DataFrame = pd.read_csv(BR_MAPS)
    filtered: pd.DataFrame = baureihen.loc[baureihen[BAUREHIE_ITEM_NUMBER_COLUMN] == product_id]

    if filtered.empty:
        raise ValueError(f"🚨 No Baureihe found for product ID: {product_id}")

    return str(filtered[BAUREHIE_COLUMN].values[0])


def get_product_posten(product_id: str) -> pd.DataFrame:
    """
    Fetches product Posten data based on the provided product ID.

    :param product_id: Product id.
    :type product_id: str
    :return: A pandas DataFrame containing the product Posten to work with.
    :rtype: pd.DataFrame
    """
    posten: pd.DataFrame = pd.read_csv(POSTEN_FILTERED_FILE)
    posten = posten.loc[posten[ARTICLE_NUMBER_COLUMN] == product_id]
    return posten


def get_data(item_code: str):
    """
    Retrieves, processes, and splits demand data for a given item code into training and testing sets.

    This function fetches demand data for a specified item code using the `get_demand_data`
    function, sets the DATE_COLUMN column as the index, splits the data into training
    and testing sets, and prepares the data for time series forecasting.

    Args:
        item_code (str): The item code ARTICLE_NUMBER_COLUMN) for which to retrieve and process demand data.

    Returns:
        Tuple[pd.DataFrame, List[pd.Timestamp], List[float]]: A tuple containing:
            - X_train: The training DataFrame with an additional row containing NaN for the first test date.
            - future_dates: A list of dates from the test set.
            - y_true: A list of actual demand values from the test set.
    """
    item: pd.DataFrame = get_demand_data(item_code)  # Retrieve demand data for the given item code.
    item = item.set_index(DATE_COLUMN, inplace=False)  # Set DATE_COLUMN as the index.

    # Split data into training and testing sets (90% training, 10% testing).
    # split_index: int = int(len(item) * TRAIN_TEST_SPLIT)
    train: pd.DataFrame = item.copy()  # Create a copy to avoid potential issues.
    end_date: pd.Timestamp = train.index[-1]  # Get the last date in the training set.

    # Extract future dates and true demand values from the test set.
    future_dates: List[pd.Timestamp] = pd.date_range(start=end_date + timedelta(days=7), periods=28,
                                                     freq='W').to_list()  # Generate future dates for the next 7 days.
    # y_true: List[float] = test[MENGE_COLUMN].to_list() # Extract the actual demand values from the test set.

    # Prepare the training set by adding a row with NaN for the first test date.
    X_train: pd.DataFrame = train.copy()  # Create a copy of the training data.
    X_train.loc[future_dates[0]] = np.nan  # Add a row with NaN for the first date in the test set.

    return X_train, future_dates  # Return the prepared training data, future dates, and true values.

def get_data_with_true(item_code: str) -> Tuple[pd.DataFrame, List[pd.Timestamp], List[float]]:
    """
    Retrieves, processes, and splits demand data for a given item code into training and testing sets.

    This function fetches demand data for a specified item code using the `get_demand_data`
    function, sets the DATE_COLUMN column as the index, splits the data into training
    and testing sets, and prepares the data for time series forecasting.

    Args:
        item_code (str): The item code ARTICLE_NUMBER_COLUMN) for which to retrieve and process demand data.

    Returns:
        Tuple[pd.DataFrame, List[pd.Timestamp], List[float]]: A tuple containing:
            - X_train: The training DataFrame with an additional row containing NaN for the first test date.
            - future_dates: A list of dates from the test set.
            - y_true: A list of actual demand values from the test set.
    """
    item: pd.DataFrame = get_demand_data(item_code)  # Retrieve demand data for the given item code.
    item = item.set_index(DATE_COLUMN, inplace=False)  # Set DATE_COLUMN as the index.

    # Split data into training and testing sets (90% training, 10% testing).
    split_index: int = int(len(item) * TRAIN_TEST_SPLIT)
    train: pd.DataFrame = item.iloc[:split_index].copy()  # Create a copy to avoid potential issues.
    test: pd.DataFrame = item.iloc[split_index:].copy()

    # Extract future dates and true demand values from the test set.
    future_dates: List[pd.Timestamp] = test.index.to_list()  # Extract the dates from the test set index.
    y_true: List[float] = test[MENGE_COLUMN].to_list()  # Extract the actual demand values from the test set.

    # Prepare the training set by adding a row with NaN for the first test date.
    X_train: pd.DataFrame = train.copy()  # Create a copy of the training data.
    X_train.loc[future_dates[0]] = np.nan  # Add a row with NaN for the first date in the test set.

    return X_train, future_dates, y_true  # Return the prepared training data, future dates, and true values.


def add_features_train(X_train: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds various time-based and statistical features to the training DataFrame.

    This function enhances the training data by creating features such as week, year,
    month, quarter, lag values, rolling means and standard deviations, week-over-week
    differences, and exponentially weighted moving averages (EWMA).

    Enhances feature engineering by adding:
    - Temporal indicators (day of week, weekend, quarter)
    - Holiday effects
    - Demand anomalies
    - Lag-based and rolling statistical features

    Args:
        X_train: The training DataFrame with 'Menge' (demand) and DATE_COLUMN as index.

    Returns:
        A tuple containing:
            - X_train: The modified training DataFrame with added features.
            - feature_list: A list of the generated feature names.
    """
    # Ensure a copy is used to avoid modifying the original DataFrame directly
    X_train_processed = X_train.copy()

    # Initialize the feature list with known temporal features
    feature_list: List[str] = ['week', 'year', 'month', 'quarter', 'day_of_week', 'is_weekend', 'is_holiday']

    # --- Temporal Features ---
    X_train_processed['week'] = X_train_processed.index.isocalendar().week.astype(int)
    X_train_processed['year'] = X_train_processed.index.year
    X_train_processed['month'] = X_train_processed.index.month
    X_train_processed['quarter'] = X_train_processed.index.quarter
    X_train_processed['day_of_week'] = X_train_processed.index.dayofweek
    X_train_processed['is_weekend'] = (X_train_processed.index.dayofweek >= 5).astype(int)

    # --- Holiday Flags ---
    # Ensure holidays are for the specific years present in the index
    country_holidays = holidays.Germany(years=X_train_processed.index.year.unique().tolist())
    X_train_processed['is_holiday'] = X_train_processed.index.map(lambda x: 1 if x in country_holidays else 0)

    # --- Lag Features ---
    # Shift 'MENGE_COLUMN' to create lag features
    for lag_val in [1, 2, 3, 4, 5, 6, 7, 8]:
        col_name = f'lag_{lag_val}'
        X_train_processed[col_name] = X_train_processed[MENGE_COLUMN].shift(lag_val)
        feature_list.append(col_name) # Add to feature_list

    # --- Rolling Mean and Standard Deviation Features ---
    # Apply rolling calculations on shifted demand to avoid data leakage
    for window in [3, 4, 5, 6]:
        mean_col_name = f'rolling_mean_{window}'
        std_col_name = f'rolling_std_{window}'
        X_train_processed[mean_col_name] = X_train_processed[MENGE_COLUMN].shift(1).rolling(window=window).mean()
        X_train_processed[std_col_name] = X_train_processed[MENGE_COLUMN].shift(1).rolling(window=window).std()
        feature_list.append(mean_col_name) # Add to feature_list
        feature_list.append(std_col_name)

    # --- Week-over-Week Demand Difference ---
    # Using shifted values to prevent data leakage
    X_train_processed['demand_diff_1'] = (X_train_processed[MENGE_COLUMN].shift(1) - X_train_processed[MENGE_COLUMN].shift(2))
    feature_list.append('demand_diff_1')

    # --- Exponentially Weighted Moving Averages (EWMA) ---
    # Apply EWMA on shifted demand
    for span_val in [1, 2, 3, 4, 5, 8]:
        ewma_col_name = f'ewma_{span_val}'
        X_train_processed[ewma_col_name] = X_train_processed[MENGE_COLUMN].shift(1).ewm(span=span_val, adjust=False).mean()
        feature_list.append(ewma_col_name)

    # --- Handle NaNs created by lags and rolling functions ---
    X_train_processed = X_train_processed.fillna(method='ffill').fillna(method='bfill')
    # If there are still NaNs (e.g., if the entire column is NaN), fill with 0 as a last resort
    X_train_processed = X_train_processed.fillna(0)


    # --- Scale Features ---
    scaler = MinMaxScaler()
    # Only scale the columns that are actually in feature_list and exist in the DataFrame
    features_to_scale = [f for f in feature_list if f in X_train_processed.columns]
    if features_to_scale: # Only try to scale if there are features to scale
        X_train_processed[features_to_scale] = scaler.fit_transform(X_train_processed[features_to_scale])

    return X_train_processed, feature_list




def load_or_train_model(
    model_name: str,
    model_instance: ConformalQRModel,
    train_series: TimeSeries, # Added type hint
    past_covariates: Optional[TimeSeries] = None # Added type hint and default
) -> ConformalQRModel:
    """
    Loads a pre-trained Darts XGBModel if it exists, otherwise trains the provided
    model instance using the given data, saves it, and returns it.

    Args:
        model_name (str): A unique name for the model used to construct the filename
                          (e.g., "monthly_sales_xgb"). Excludes the file extension.
        model (XGBModel): An instance of darts.models.XGBModel, potentially
                          pre-configured but not yet trained. This instance will be
                          trained and saved if no pre-trained model is found.
        train_series (TimeSeries): The target time series data to train the model on.
        past_covariates (Optional[TimeSeries]): Optional past-observed covariates
                                                  series to use during training. Defaults to None.
                                                  Ensure this matches the model's requirements.

    Returns:
        XGBModel: The loaded pre-trained model, or the newly trained model instance.
    """
    # Construct the full path for the model file, including the directory
    model_path = os.path.join(SAVE_DIR_FOLDER, f"{model_name}.pkl")
    model_dir = os.path.dirname(model_path) # Get the directory part of the path

    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(model_path):
        print(f"📦 Loading pre-trained model from: {model_path}")
        if isinstance(model_instance, ConformalQRModel):
            loaded_model = ConformalQRModel.load(model_path)
        elif isinstance(model_instance, XGBModel):
            loaded_model = XGBModel.load(model_path)
        else:
            raise TypeError(f"Unsupported model type {type(model_instance)} for loading.")
        return loaded_model
    else:
        print(f"🚀 Training new model and saving to: {model_path}")
        model_instance.fit(series=train_series, past_covariates=past_covariates)
        model_instance.save(model_path)
        return model_instance


def recursive_prediction_darts(
        item_code: str,
        X_train: pd.DataFrame,
        future_dates: List[pd.Timestamp],
        quantile: List[float] = [0.025, 0.5, 0.975],
        sample: int = 5000,
        lag: int = 15
) -> Tuple[List[Tuple[pd.Timestamp, float, float]], List[float], List[float], List[str]]:
    """
    Performs recursive prediction using the XGBModel from the Darts library, incorporating feature engineering and prediction intervals.

    This function takes a training DataFrame, future dates, and true values, engineers features,
    trains an XGBModel with quantile regression, and generates predictions with prediction intervals.

    Args:
        X_train: The training DataFrame containing demand data and potentially other features.
        future_dates: A list of dates for which predictions are to be made.
        y_true: A list of true demand values corresponding to the future dates.
        quantile: A list of quantiles for prediction intervals (default: [0.025, 0.5, 0.975]).
        sample: The number of samples to draw for prediction intervals (default: 5000).
        lag: The number of lag values to use for the model (default: 15).

    Returns:
        A tuple containing:
            - predictions: A list of tuples, each containing (date, predicted_median, true_value).
            - lower_bounds: A list of lower bound predictions.
            - upper_bounds: A list of upper bound predictions.
            - feature_list: A list of features used in the model.
    """

    predictions: List[Tuple[pd.Timestamp, float, float]] = []
    lower_bounds: List[float] = []
    upper_bounds: List[float] = []

    # Apply feature engineering to the training data.
    X_train, feature_list = add_features_train(X_train)

    # Fill NaN values in the features using forward fill, backward fill, and then 0.
    X_train = X_train.fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Sort the index to ensure proper time series order.
    X_train = X_train.sort_index()

    # Ensure no future dates are included in the training data.
    X_train = X_train.loc[X_train.index <= future_dates[0]]

    # Convert the training DataFrame to Darts TimeSeries format.
    train_series: TimeSeries = TimeSeries.from_dataframe(X_train, value_cols=[MENGE_COLUMN])
    past_covariates: TimeSeries = TimeSeries.from_dataframe(X_train, value_cols=feature_list)

    # Align the time indices of the training series and past covariates.
    past_covariates = past_covariates.slice_intersect(train_series)

    # Check if the time indices match between the training series and past covariates.
    if not (train_series.time_index.equals(past_covariates.time_index)):
        raise ValueError("🚨 Mismatch in time indices! Ensure feature engineering keeps timestamps aligned.")

    # Create a unique model name
    model_name = f"{item_code}_model_nosplit".replace(" ", "_")

    # Initialize and train the XGBModel with quantile regression.
    model_xgb = XGBModel(
        lags=lag,
        lags_past_covariates=lag,
        output_chunk_length=len(future_dates),
        likelihood="quantile",
        quantiles=quantile,
        random_state=42
    )

    # Load or train the model
    model_xgb = load_or_train_model(model_name, model_xgb, train_series, past_covariates)

    # Prepare the target variable (y_train) and ensure no NaN values.
    y_train: pd.Series = X_train[MENGE_COLUMN].fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Drop rows where y_train still has NaN (failsafe).
    if y_train.isna().sum() > 0:
        print(f" Warning: NaN values detected in y_train! Removing NaN rows...")
        X_train = X_train.loc[~y_train.isna()]
        y_train = y_train.dropna()

    # Final check to ensure y_train does not contain any NaN values.
    assert not y_train.isna().any(), "🚨 ERROR: y_train still contains NaN!"

    # Fit the model with the training series and past covariates.
    # model_xgb.fit(series=train_series, past_covariates=past_covariates)

    # Generate predictions for the future dates.
    predictions_series = model_xgb.predict(len(future_dates), past_covariates=past_covariates, num_samples=sample)

    # Extract quantile predictions.
    y_pred_low: List[float] = predictions_series.quantile_timeseries(quantile[0]).values().flatten()
    y_pred_median: List[float] = predictions_series.quantile_timeseries(quantile[1]).values().flatten()
    y_pred_high: List[float] = predictions_series.quantile_timeseries(quantile[2]).values().flatten()

    # Store the predictions and prediction intervals.
    for i, date in enumerate(future_dates):
        predictions.append((date, y_pred_median[i]))
        lower_bounds.append(y_pred_low[i])
        upper_bounds.append(y_pred_high[i])

    return predictions, lower_bounds, upper_bounds, feature_list  # Return the predictions, bounds, and feature list.

def recursive_prediction_darts_with_split(
        item_code: str,
        X_train: pd.DataFrame,
        future_dates: List[pd.Timestamp],
        y_true: List[float],
        quantile: List[float] = [0.025, 0.5, 0.975],
        sample: int = 5000,
        lag: int = 15
) -> Tuple[List[Tuple[pd.Timestamp, float, float]], List[float], List[float], List[str]]:
    """
    Performs recursive prediction using the XGBModel from the Darts library, incorporating feature engineering and prediction intervals.

    This function takes a training DataFrame, future dates, and true values, engineers features,
    trains an XGBModel with quantile regression, and generates predictions with prediction intervals.

    Args:
        X_train: The training DataFrame containing demand data and potentially other features.
        future_dates: A list of dates for which predictions are to be made.
        y_true: A list of true demand values corresponding to the future dates.
        quantile: A list of quantiles for prediction intervals (default: [0.025, 0.5, 0.975]).
        sample: The number of samples to draw for prediction intervals (default: 5000).
        lag: The number of lag values to use for the model (default: 15).

    Returns:
        A tuple containing:
            - predictions: A list of tuples, each containing (date, predicted_median, true_value).
            - lower_bounds: A list of lower bound predictions.
            - upper_bounds: A list of upper bound predictions.
            - feature_list: A list of features used in the model.
    """

    predictions: List[Tuple[pd.Timestamp, float, float]] = []
    lower_bounds: List[float] = []
    upper_bounds: List[float] = []

    # Apply feature engineering to the training data.
    X_train, feature_list = add_features_train(X_train)

    # Fill NaN values in the features using forward fill, backward fill, and then 0.
    X_train = X_train.fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Sort the index to ensure proper time series order.
    X_train = X_train.sort_index()

    # Ensure no future dates are included in the training data.
    X_train = X_train.loc[X_train.index <= future_dates[0]]

    # Convert the training DataFrame to Darts TimeSeries format.
    train_series: TimeSeries = TimeSeries.from_dataframe(X_train, value_cols=[MENGE_COLUMN])
    past_covariates: TimeSeries = TimeSeries.from_dataframe(X_train, value_cols=feature_list)

    # Align the time indices of the training series and past covariates.
    past_covariates = past_covariates.slice_intersect(train_series)

    # Check if the time indices match between the training series and past covariates.
    if not (train_series.time_index.equals(past_covariates.time_index)):
        raise ValueError("🚨 Mismatch in time indices! Ensure feature engineering keeps timestamps aligned.")

    model_name = f"{item_code}_model_split".replace(" ", "_")

    # Initialize and train the XGBModel with quantile regression.
    model_xgb = XGBModel(
        lags=lag,
        lags_past_covariates=lag,
        output_chunk_length=len(future_dates),
        likelihood="quantile",
        quantiles=quantile,
        random_state=42
    )

    # Load or train the model
    model_xgb = load_or_train_model(model_name, model_xgb, train_series, past_covariates)

    # Prepare the target variable (y_train) and ensure no NaN values.
    y_train: pd.Series = X_train[MENGE_COLUMN].fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Drop rows where y_train still has NaN (failsafe).
    if y_train.isna().sum() > 0:
        print(f" Warning: NaN values detected in y_train! Removing NaN rows...")
        X_train = X_train.loc[~y_train.isna()]
        y_train = y_train.dropna()

    # Final check to ensure y_train does not contain any NaN values.
    assert not y_train.isna().any(), "🚨 ERROR: y_train still contains NaN!"

    # Fit the model with the training series and past covariates.
    #model_xgb.fit(series=train_series, past_covariates=past_covariates)

    # Generate predictions for the future dates.
    predictions_series = model_xgb.predict(len(future_dates), past_covariates=past_covariates, num_samples=sample)

    # Extract quantile predictions.
    y_pred_low: List[float] = predictions_series.quantile_timeseries(quantile[0]).values().flatten()
    y_pred_median: List[float] = predictions_series.quantile_timeseries(quantile[1]).values().flatten()
    y_pred_high: List[float] = predictions_series.quantile_timeseries(quantile[2]).values().flatten()

    # Store the predictions and prediction intervals.
    for i, date in enumerate(future_dates):
        predictions.append((date, y_pred_median[i], y_true[i]))
        lower_bounds.append(y_pred_low[i])
        upper_bounds.append(y_pred_high[i])

    return predictions, lower_bounds, upper_bounds, feature_list  # Return the predictions, bounds, and feature list.


def recursive_prediction_conformal_darts(
        item_code: str,
        X_train: pd.DataFrame,
        future_dates: List[pd.Timestamp],
        quantile: List[float] = [0.025, 0.5, 0.975],
        sample: int = 5000,
        lag: int = 15
) -> Tuple[List[Tuple[pd.Timestamp, float]], List[float], List[float], List[str]]:
    """
    Performs recursive time series prediction using a Conformal Quantile Regression Model (ConformalQRModel)
    with an XGBoost base model from the Darts library. This function also handles feature engineering
    and loading/training of models.

    Args:
        item_code (str): A unique identifier for the item being predicted, used for model naming.
        X_train (pd.DataFrame): Training data containing the target variable and features.
                                Its index should be a datetime index.
        future_dates (List[pd.Timestamp]): A list of future dates for which predictions are required.
        quantile (List[float], optional): A list of quantiles to predict. Defaults to [0.025, 0.5, 0.975]
                                           for 2.5th, 50th (median), and 97.5th percentiles.
        sample (int, optional): The number of samples to generate for probabilistic forecasting.
                                Defaults to 5000.
        lag (int, optional): The number of past time steps to use as input features (lags) for the models.
                             Defaults to 15.

    Returns:
        Tuple[List[Tuple[pd.Timestamp, float]], List[float], List[float], List[str]]:
            A tuple containing:
            - predictions (List[Tuple[pd.Timestamp, float]]): A list of tuples, where each tuple
                                                              contains a future date and its median prediction.
            - lower_bounds (List[float]): A list of lower bound predictions corresponding to the first quantile.
            - upper_bounds (List[float]): A list of upper bound predictions corresponding to the last quantile.
            - feature_list (List[str]): A list of column names used as features (past covariates).
    """

    # Initialize lists to store prediction results
    predictions: List[Tuple[pd.Timestamp, float]] = []
    lower_bounds: List[float] = []
    upper_bounds: List[float] = []

    # --- Feature Engineering ---
    # Apply custom feature engineering to the training data.
    # The `add_features_train` function is expected to add new columns to X_train.
    X_train, feature_list = add_features_train(X_train)

    # Fill any NaN values that might have been introduced during feature engineering.
    # Uses forward fill, then backward fill, then replaces remaining NaNs with 0.
    X_train = X_train.fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Ensure the DataFrame is sorted by its index (datetime) for time series operations.
    X_train = X_train.sort_index()

    # Trim the training data to only include records up to the first future prediction date.
    # This ensures that the model is trained only on "known" past data.
    X_train = X_train.loc[X_train.index <= future_dates[0]]

    # --- Convert to Darts TimeSeries ---
    # Convert the target variable column (e.g., 'Menge') into a Darts TimeSeries object.
    # MENGE_COLUMN is assumed to be defined globally or passed as an argument.
    train_series: TimeSeries = TimeSeries.from_dataframe(X_train, value_cols=[MENGE_COLUMN])

    # Convert the engineered features into a Darts TimeSeries object to be used as past covariates.
    past_covariates: TimeSeries = TimeSeries.from_dataframe(X_train, value_cols=feature_list)

    # Ensure that the time indices of the target series and covariates series are aligned.
    # This is crucial for Darts models.
    past_covariates = past_covariates.slice_intersect(train_series)

    # Validate that the time indices match. If not, raise an error.
    if not (train_series.time_index.equals(past_covariates.time_index)):
        raise ValueError("🚨 Mismatch in time indices! Ensure feature engineering keeps timestamps aligned.")

    # --- Model Names ---
    # Generate unique names for the base model and the conformal model based on the item code.
    base_model_name = f"{item_code}_xgb_base_model_nosplit".replace(" ", "_")
    conformal_model_name = f"{item_code}_conformal_model_nosplit".replace(" ", "_")

    # Step 1: Define and load/train the base XGBModel
    # Initialize the XGBModel (base model) with specified lags, output chunk length,
    # likelihood (for quantile regression), and quantiles.
    base_model_init = XGBModel(
        lags=lag,  # Lags for the target series
        lags_past_covariates=lag,  # Lags for the past covariates
        output_chunk_length=len(future_dates),  # The number of steps to predict at once
        likelihood="quantile",  # Specifies quantile regression
        quantiles=quantile,  # The quantiles to be predicted by the base model
        random_state=42  # For reproducibility
    )

    # Load a pre-trained base model if it exists, otherwise train a new one.
    # The `load_or_train_model` function handles this logic and returns a fitted model.
    print(f"Loading or training base model for {item_code}...")
    base_model = load_or_train_model(base_model_name, base_model_init, train_series, past_covariates)

    # Step 2: Initialize ConformalQRModel with the *fitted* base model
    # The ConformalQRModel wraps a fitted base model to provide conformal prediction intervals.
    # It takes the already fitted `base_model` as its core.
    conformal_model_init = ConformalQRModel(
        model=base_model,  # Pass the already fitted GlobalForecastingModel (XGBModel in this case)
        cal_stride=1,  # Stride for calibration set creation (how many steps to move forward)
        cal_length=int(len(train_series) * 0.2),  # Length of the calibration set (20% of training data)
        quantiles=quantile,  # Quantiles to be output by the conformal model
        random_state=42,  # For reproducibility
        symmetric=False,  # If true, the prediction intervals are symmetric around the median prediction
    )

    # Step 3: Load or train the ConformalQRModel
    # Similar to the base model, this step loads or trains the conformal model.
    # Training involves using a calibration set to adjust the prediction intervals.
    print(f"Loading or training conformal model for {item_code}...")
    conformal_model = load_or_train_model(conformal_model_name, conformal_model_init, train_series, past_covariates)

    # --- Predict ---
    # Use the loaded/trained conformal model to make predictions for the specified future dates.
    # `num_samples` is used for probabilistic forecasting, even for quantile models.
    print(f"Generating predictions for {item_code}...")
    predictions_series = conformal_model.predict(len(future_dates), past_covariates=past_covariates, num_samples=sample)

    # --- Extract Quantiles ---
    # Extract the predicted time series for each specified quantile.
    # `values().flatten()` converts the Darts TimeSeries to a 1D NumPy array.
    y_pred_low: List[float] = predictions_series.quantile_timeseries(quantile[0]).values().flatten()
    y_pred_median: List[float] = predictions_series.quantile_timeseries(quantile[1]).values().flatten()
    y_pred_high: List[float] = predictions_series.quantile_timeseries(quantile[2]).values().flatten()

    # Iterate through the future dates and populate the prediction, lower, and upper bound lists.
    for i, date in enumerate(future_dates):
        predictions.append((date, y_pred_median[i]))
        lower_bounds.append(max(0.0, y_pred_low[i]))
        upper_bounds.append(y_pred_high[i])

    # Return the aggregated predictions, bounds, and the list of features used.
    return predictions, lower_bounds, upper_bounds, feature_list


def aggregate_weekly_results (result_df:pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates weekly prediction results into monthly totals.

    This function takes a DataFrame containing weekly prediction results and processes
    it to compute the aggregated monthly totals for specific settings. The aggregation
    is performed by summing up the values of the respective columns grouped by the
    `REORDER_MONTH_COLUMN`. The resulting DataFrame is organized with its index sorted
    in chronological order and formatted as per the `MONTH_FORMAT`.

    :param result_df: Input DataFrame containing weekly results with predictions and
        their related bounds. Expected columns include `PREDICTED_COLUMN`,
        `LOWER_BOUND_COLUMN`, and `UPPER_BOUND_COLUMN`.
    :type result_df: pd.DataFrame
    :return: A DataFrame containing aggregated prediction results on a monthly basis.
        The index is formatted as per the specified month format.
    :rtype: pd.DataFrame
    """
    results_month_aggregated = result_df.groupby(REORDER_MONTH_COLUMN)[
        [LOWER_BOUND_COLUMN,PREDICTED_COLUMN, UPPER_BOUND_COLUMN]].sum()
    results_month_aggregated.index = pd.to_datetime(results_month_aggregated.index, format=MONTH_FORMAT)
    results_month_aggregated = results_month_aggregated.sort_index()
    results_month_aggregated.index = results_month_aggregated.index.strftime(MONTH_FORMAT)
    results_month_aggregated = results_month_aggregated.reset_index()
    # turn value columns into integers
    results_month_aggregated[PREDICTED_COLUMN] = results_month_aggregated[PREDICTED_COLUMN].astype(int)
    results_month_aggregated[LOWER_BOUND_COLUMN] = results_month_aggregated[LOWER_BOUND_COLUMN].astype(int)
    results_month_aggregated[UPPER_BOUND_COLUMN] = results_month_aggregated[UPPER_BOUND_COLUMN].astype(int)
    return results_month_aggregated

def compute_stock_levels(
        predictions: pd.DataFrame,
        replenishment: pd.DataFrame,
        final_stock_level: float,
) -> pd.DataFrame:
    """
    Computes stock levels over time using recursive demand predictions and monthly replenishment data.

    Parameters:
    - predictions: DataFrame with ['date', 'predicted_demand'] settings. 'date' should be datetime.
    - replenishment: DataFrame with ['Reorder Date', 'Order Quantity'] settings.
                     'Reorder Date' should be datetime.
    - final_stock_level: The initial stock level (float) before the prediction period begins.

    Returns:
    - DataFrame with ['date', 'predicted_demand', 'stock_level'] columns, containing the
      calculated stock level for each date in the predictions DataFrame.
    """
    predictions[PREDICTED_DATE_COLUMN] = pd.to_datetime(predictions[PREDICTED_DATE_COLUMN])
    replenishment[REORDER_DATE_COLUMN] = pd.to_datetime(replenishment[REORDER_DATE_COLUMN])

    stock_level: float = final_stock_level
    stock_levels: list[float] = []

    for i, row in predictions.iterrows():
        date: pd.Timestamp = row[PREDICTED_DATE_COLUMN]
        demand: float = row[PREDICTED_DEMAND_COLUMN]

        month_period = date.to_period("M")
        replenishment_this_month = replenishment.loc[replenishment[REORDER_MONTH_COLUMN] == month_period, ORDER_COLUMN]
        received_stock: float = replenishment_this_month.sum() if not replenishment_this_month.empty else 0

        stock_level = stock_level - demand + received_stock
        stock_levels.append(stock_level)

    predictions[PREDECTED_STOCK_COLUMN] = stock_levels
    return predictions


def plot_stock_levels(
        stock_data: pd.DataFrame,
        replenishment: pd.DataFrame,
        item_code: str
) -> pd.DataFrame:
    """
    Plots stock levels over time, including replenishment events and safety stock.

    This function visualizes how stock levels change over time based on demand and
    replenishment events.

    Args:
        stock_data: A DataFrame with ['Date', 'predicted_demand', 'stock_level'] settings.
        replenishment: A DataFrame with ['Reorder Date', 'Order Quantity'] settings.
        item_code: The item code for which the stock levels are plotted.

    Returns:
        A pandas DataFrame containing stock levels over time.
    """
    # Ensure date columns are in datetime format
    stock_data[PREDICTED_DATE_COLUMN] = pd.to_datetime(stock_data[PREDICTED_DATE_COLUMN])
    replenishment[REORDER_DATE_COLUMN] = pd.to_datetime(replenishment[REORDER_DATE_COLUMN])

    # Sort stock data by date
    stock_data = stock_data.sort_values(by=PREDICTED_DATE_COLUMN)

    # Extract safety stock information (assuming it exists)
    inventory_data = extract_inventory_data()
    inventory_info = get_info(inventory_data, item_code)
    safety_stock = inventory_info.safety_stock

    # Create the stock levels plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=stock_data, x=PREDICTED_DATE_COLUMN, y=PREDECTED_STOCK_COLUMN, label='Stock Level', marker="o",
                 color="blue")

    # Highlight replenishment events
    for _, row in replenishment.iterrows():
        plt.axvline(row[REORDER_DATE_COLUMN], linestyle="--", color="green", alpha=0.7,
                    label="Replenishment" if _ == 0 else "")

    # Safety stock threshold
    # plt.axhline(y=safety_stock, color="red", linestyle="--", label="Safety Stock Level")

    # Formatting
    plt.xticks(rotation=45)
    plt.title(f"Stock Levels for Item Code: {item_code}")
    plt.xlabel("Date")
    plt.ylabel("Stock Level")
    plt.legend()
    plt.grid(True)

    # Save and display the plot
    plt.savefig(f'{PLOT_FOLDER}Stock Levels for {item_code}.png', dpi=300)
    plt.show()


def plot_predictions_data(
        predictions: List[Tuple[pd.Timestamp, float, float]],
        lower_bounds: List[float],
        upper_bounds: List[float],
        item_code: str
) -> pd.DataFrame:
    """
    Generates an enhanced visualization of predictions with prediction intervals and saves the plot.

    This function creates a plot comparing predicted and actual demand values, including a
    shaded area representing the prediction interval. It also saves the plot to a file.

    Args:
        predictions: A list of tuples, each containing (date, predicted_value, true_value).
        lower_bounds: A list of lower bound prediction values.
        upper_bounds: A list of upper bound prediction values.
        item_code: The item code for which the predictions were made.

    Returns:
        A pandas DataFrame containing the prediction results with lower and upper bounds.
    """
    results_df: pd.DataFrame = pd.DataFrame(predictions, columns=[PREDICTED_DATE_COLUMN, PREDICTED_COLUMN])
    results_df[LOWER_BOUND_COLUMN] = lower_bounds
    results_df[UPPER_BOUND_COLUMN] = upper_bounds

    results_df[PREDICTED_DATE_COLUMN] = pd.to_datetime(
        results_df[PREDICTED_DATE_COLUMN])  # Ensure the 'Date' column is in datetime format.
    results_df = results_df.sort_values(by=PREDICTED_DATE_COLUMN)  # Sort the DataFrame by date for proper plotting.
    last_day: pd.Timestamp = results_df[PREDICTED_DATE_COLUMN].min()

    inventory_data = extract_inventory_data()
    inventory_info = get_info(inventory_data, item_code)
    safety_stock = inventory_info.safety_stock

    # Create an enhanced plot using seaborn and matplotlib.
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results_df, x=PREDICTED_DATE_COLUMN, y=PREDICTED_COLUMN, label=PREDICTED_COLUMN, marker="o",
                 color="blue")
    # sns.lineplot(data=results_df, x=PREDICTED_DATE_COLUMN, y=PREDICTED_TRUE_COLUMN, label=PREDICTED_TRUE_COLUMN, marker="s", color="black")

    # Fill the area between the lower and upper bounds to represent the prediction interval.
    plt.fill_between(
        results_df[PREDICTED_DATE_COLUMN], results_df[LOWER_BOUND_COLUMN], results_df[UPPER_BOUND_COLUMN],
        color="gray", alpha=0.3, label="95% Prediction Interval"
    )
    plt.axvline(last_day, linestyle="--", label="Day of prediction", color="pink")
    plt.axhline(y=safety_stock, color="red", linestyle="--", label="Safety Stock Menge")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability.
    plt.title(f"Forecast for Item Code: {item_code}")  # Set the plot title.
    plt.xlabel(PREDICTED_DATE_COLUMN)  # Set the x-axis label.
    plt.ylabel(MENGE_COLUMN)  # Set the y-axis label.
    plt.legend()  # Display the legend.
    plt.grid(True)  # Add grid lines to the plot.

    # Save the plot to a file.
    plt.savefig(f'{PLOT_FOLDER}Recursive Forecasting for {item_code}.png', dpi=300)
    plt.show()  # Display the plot.

    return results_df  # Return the DataFrame containing the prediction results.


def extract_inventory_data() -> pd.DataFrame:
    """
    Extracts inventory-related data from an Excel file.

    This function reads inventory data from an Excel file, selects relevant columns,
    renames the item code column, and sets it as the index.

    Returns:
        A pandas DataFrame containing inventory data with 'ItemCode' as the index.
    """
    cache_path: str = CACHED_INVENTORY_FILE

    return pd.read_csv(cache_path, index_col=ITEM_CODE_COLUMN)

def parse_lead_time(lead_time: str) -> int:
    """
    Converts lead time from weeks or working days (WD) to actual days.

    This function parses a lead time string, which can be in weeks or working days (WD),
    and converts it to the equivalent number of calendar days.

    Args:
        lead_time: A string representing the lead time (e.g., '2', '5WD').

    Returns:
        The lead time in calendar days. Returns 30 if parsing fails.
    """
    try:
        if 'WD' in lead_time:
            days: int = int(lead_time.replace('WD', ''))
            return days * 7 // 5  # Convert working days to calendar days
        return int(lead_time) * 7  # Convert weeks to days
    except ValueError:
        return 3  # Default to 30 days if parsing fails


def display_replenishment_guidelines(
        safety_stock: int,
        lead_time: str,
        min_order_qty: int,
        max_order_qty: int,
        current_stock: int
):
    """
    Generates and prints decision support guidelines for replenishment.

    This function calculates lead time in days and prints guidelines related to order timing,
    replenishment triggers, and order quantities.

    Args:
        safety_stock: The safety stock Menge.
        lead_time: The lead time string (e.g., '2', '5WD').
        min_order_qty: The minimum order Menge.
        max_order_qty: The maximum order Menge.
    """
    lead_days: int = parse_lead_time(lead_time)
    decisionSupp = outside_translate("functions.guidelines_header")
    currect_stock_str = outside_translate("functions.current_stock").format(current_stock=current_stock)
    order_timing = outside_translate("functions.order_timing").format(lead_days=lead_days)
    replenish_rule = outside_translate("functions.replenish_rule").format(safety_stock=safety_stock)
    min_order_str = outside_translate("functions.min_order_qty").format(min_order_qty=max(min_order_qty, 1))
    if max_order_qty == 0:
        max_oder_str = outside_translate("functions.max_order_qty").format(max_order_qty=min_order_qty * 2)
    else:
        max_oder_str = outside_translate("functions.max_order_qty").format(max_order_qty=max_order_qty)
    fluctaution_buffer = outside_translate("functions.fluctuation_buffer")
    demand_summ = outside_translate("functions.demand_summary")

    guidelines: str = f"""
        {decisionSupp}
        - {currect_stock_str}
        - {order_timing}
        - {replenish_rule}
        - {min_order_str}
        - {max_oder_str}
        - {fluctaution_buffer}
        - {demand_summ}
        """
    return guidelines  # Return the guidelines string.


def get_info(item_data: pd.DataFrame, item_code: str) -> InventoryInfo:
    """
    Retrieves inventory information for a specific item code.

    This function retrieves safety stock, safety lead time, maximum inventory, lead time,
    minimum order quantity, and maximum order quantity for a given item code.

    Args:
        item_data: The DataFrame containing inventory data.
        item_code: The item code to retrieve information for.

    Returns:
        An InventoryInfo object containing all relevant inventory attributes.
    """
    safety_stock: int = item_data.loc[item_code, INVENTORY_DATA_SAFETY_STOCK_COLUMN]
    safety_lead_time: str = item_data.loc[item_code, INVENTORY_DATA_SAFETY_LEAD_TIME_COLUMN]
    max_inventory: int = item_data.loc[item_code, INVENTORY_DATA_MAX_INVENTORY_COLUMN]
    lead_time: str = item_data.loc[item_code, INVENTORY_DATA_LEAD_TIME_COLUMN]
    min_order_qty: int = item_data.loc[item_code, INVENTORY_DATA_MIN_ORDER_QTY_COLUMN]
    max_order_qty: int = item_data.loc[item_code, INVENTORY_DATA_MAX_ORDER_QTY_COLUMN]
    if max_order_qty == 0:
        max_order_qty = min_order_qty * 2

    return InventoryInfo(
        safety_stock=safety_stock,
        safety_lead_time=safety_lead_time,
        max_inventory=max_inventory,
        lead_time=lead_time,
        min_order_qty=min_order_qty,
        max_order_qty=max_order_qty
    )  # Return the inventory information tuple.


def determine_optimal_order_quantity(
        min_order_qty: int,
        max_order_qty: int,
        safety_stock: int
) -> int:
    """
    Determines the optimal order quantity based on minimum, maximum, and safety stock.

    This function calculates the optimal order quantity, especially when the maximum order
    quantity is zero, using a heuristic based on safety stock.

    Args:
        min_order_qty: The minimum order quantity.
        max_order_qty: The maximum order quantity.
        safety_stock: The safety stock quantity.

    Returns:
        The optimal order quantity.
    """
    if max_order_qty == 0:
        return max(min_order_qty, int(safety_stock * 1.5))  # Example heuristic: 1.5x safety stock
    return max_order_qty  # Return the maximum order quantity if not zero.


def aggregate_monthly_data(results_df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """
    Aggregates the results DataFrame by month and calculates the sum of lower, predicted, and upper bounds.

    It groups the data by month, sums the specified columns, sorts chronologically,
    rounds the sums up, and returns a dictionary mapping month strings ('MM-YYYY')
    to the rounded predicted order values.

    Args:
        results_df (pd.DataFrame): The DataFrame containing the prediction results.
                                   Must have a datetime column specified by PREDICTED_DATE_COLUMN
                                   and columns "Lower_Bound", "Predicted", "Upper_Bound".

    Returns:
        Dict[str, Union[int, float]]: A dictionary where keys are month strings (e.g., '04-2025')
                                      and values are the corresponding aggregated and rounded up
                                      predicted order quantities.
    """
    # Create a new column with the month and year string (e.g., '04-2025')
    results_df[REORDER_MONTH_COLUMN] = results_df[PREDICTED_DATE_COLUMN].dt.strftime(MONTH_FORMAT)

    # Group the DataFrame by the month string and aggregate the sums
    grouped: pd.DataFrame = results_df.groupby(REORDER_MONTH_COLUMN).agg({
        LOWER_BOUND_COLUMN: "sum",
        PREDICTED_COLUMN: "sum",
        UPPER_BOUND_COLUMN: "sum"
    })

    # Convert the month string index to a DatetimeIndex for proper sorting
    grouped.index = pd.to_datetime(grouped.index, format=MONTH_FORMAT)

    # Sort the DataFrame chronologically by the month index
    grouped.sort_index(ascending=True, inplace=True)

    # Round up all the aggregated values to the nearest integer (or float if original data was float)
    grouped = np.ceil(grouped)

    # Convert the DatetimeIndex back into a column (named after the original index name)
    grouped.reset_index(inplace=True)

    # Format the month column (which now contains datetime objects) back to the 'MM-YYYY' string format
    # This step assumes the column name after reset_index is REORDER_MONTH_COLUMN
    grouped[REORDER_MONTH_COLUMN] = grouped[REORDER_MONTH_COLUMN].dt.strftime(MONTH_FORMAT)

    # Rename the columns to the desired final names
    grouped.columns = [REORDER_MONTH_COLUMN, WORSE_CASE_ORDER_COLUMN, ORDER_COLUMN, BEST_CASE_ORDER_COLUMN]

    # Create a dictionary mapping month strings to the aggregated predicted order values
    # The values will be float if np.ceil was applied to float data, or int if applied to int data.
    lower_bound_demand_dict: Dict[str, Union[int, float]] = pd.Series(
        grouped[WORSE_CASE_ORDER_COLUMN].values,
        index=grouped[REORDER_MONTH_COLUMN]
    ).to_dict()

    demand_dict: Dict[str, Union[int, float]] = pd.Series(
        grouped[ORDER_COLUMN].values,
        index=grouped[REORDER_MONTH_COLUMN]
    ).to_dict()

    upper_bound_demand_dict: Dict[str, Union[int, float]] = pd.Series(
        grouped[BEST_CASE_ORDER_COLUMN].values,
        index=grouped[REORDER_MONTH_COLUMN]
    ).to_dict()

    # Return the dictionary of monthly demand
    return lower_bound_demand_dict, demand_dict, upper_bound_demand_dict


def add_months(date_str: str, months_to_add: int) -> str:
    """
    Adds a specified number of months to a date string in 'MM-YYYY' format.

    Uses dateutil.relativedelta for accurate month arithmetic (handles varying month lengths
    and leap years correctly).

    Args:
        date_str (str): The starting date string in the format specified by MONTH_FORMAT (e.g., '04-2025').
        months_to_add (int): The number of months to add (can be positive or negative).

    Returns:
        str: The resulting date string after adding the months, in the same 'MM-YYYY' format.
    """
    # Parse the input date string into a datetime object using the defined format
    dt: datetime = datetime.strptime(date_str, MONTH_FORMAT)

    # Add the specified number of months using relativedelta for accurate month arithmetic
    new_dt: datetime = dt + relativedelta(months=months_to_add)

    # Format the resulting datetime object back to the 'MM-YYYY' string format and return it
    return new_dt.strftime(MONTH_FORMAT)


def run_inventory_simulation(demand_dict: Dict[str, float], demand_type: str, inventory_info: InventoryInfo,
                             initial_stock: float) -> pd.DataFrame:
    """
    Simulates monthly inventory levels and reorder decisions.

    Args:
        demand_dict (Dict[str, float]): Monthly forecasted demand in format {'MM-YYYY': demand_value}.
        demand_type (str): Type of demand (e.g., 'lower', 'predicted', 'upper').
        inventory_info (InventoryInfo): Object with inventory parameters (safety_stock, min_order_qty, max_order_qty, lead_time).
        initial_stock (float): Starting stock on hand.

    Displays:
        DataFrame of inventory simulation across time.
    """

    # Unpack inventory configuration
    safety_stock = inventory_info.safety_stock
    min_order_qty = inventory_info.min_order_qty
    max_order_qty = inventory_info.max_order_qty
    lead_time_wd = int(inventory_info.lead_time.removesuffix('WD'))

    # guidelines = display_replenishment_guidelines(
    #     safety_stock=safety_stock,
    #     lead_time=inventory_info.lead_time,
    #     min_order_qty=min_order_qty,
    #     max_order_qty=max_order_qty,
    #     current_stock=initial_stock
    # )
    # print(guidelines)
    print(f'{demand_type} demand forecast:')

    # Map WD lead time to approximate months
    if lead_time_wd <= 20:
        arrival_delay_months = 1
        print(f"{LEAD_TIME} <= {inventory_info.lead_time} → Orders arrive NEXT month.")
    elif lead_time_wd <= 45:
        arrival_delay_months = 2
        print(f"{LEAD_TIME} <= {inventory_info.lead_time} → Orders arrive in 2 months.")
    else:
        arrival_delay_months = 3
        print(f"{LEAD_TIME} > {inventory_info.lead_time} → Orders arrive in 3 months (default assumption).")

    stock_on_hand: float = initial_stock
    stock_on_order: Dict[str, float] = {}  # ArrivalMonth → Quantity
    simulation_log = []

    # Sort months chronologically
    months = sorted(demand_dict.keys(), key=lambda d: datetime.strptime(d, '%m-%Y'))
    back_logged = outside_translate("functions.backlogged")
    for current_month in months:
        # --- STEP 1: Receive Arriving Stock ---
        # If any stock is scheduled to arrive this month, add it to stock on hand.
        arriving_stock = stock_on_order.pop(current_month, 0.0)
        stock_on_hand += arriving_stock

        # --- STEP 2: Record Initial State ---
        # Capture stock level before demand is processed (after arrivals).
        stock_start = stock_on_hand
        current_demand = demand_dict.get(current_month, 0.0)  # Use 0 if demand not defined for the month

        # --- STEP 3: Fulfill This Month's Demand ---
        # Fulfill as much demand as possible with current stock
        fulfilled = min(stock_on_hand, current_demand)
        shortfall = max(0.0, current_demand - fulfilled)  # If demand exceeds available stock
        stock_on_hand -= fulfilled  # Reduce stock by fulfilled amount

        # --- STEP 4: Calculate Reorder Point (ROP) ---
        # ROP = Total demand expected during lead time + safety stock
        demand_during_lead = 0.0
        for offset in range(arrival_delay_months):
            future_month = add_months(current_month, offset)
            demand_during_lead += demand_dict.get(future_month, 0.0)  # Forecast may not extend far enough, default to 0
        reorder_point = demand_during_lead + safety_stock

        # --- STEP 5: Compute Inventory Position ---
        # Inventory position = stock on hand + stock already ordered (in transit)
        total_stock_on_order = sum(stock_on_order.values())
        inventory_position = stock_on_hand + total_stock_on_order

        # Default order state: no order
        order_decision = outside_translate("functions.no")
        order_qty = 0.0
        expected_arrival_month = ""
        reasoning = f"{INVENTORY_LEVEL_COLUMN} ({inventory_position:.1f}) >= {REORDER_POINT_COLUMN} ({reorder_point:.1f})"

        # Track backlogged quantity from previous order shortages
        backlog_qty = 0.0

        # --- STEP 6: Decision to Place Order ---
        if inventory_position < reorder_point:
            order_decision = outside_translate("functions.yes")
            required_qty = math.ceil(reorder_point - inventory_position)  # Round up to avoid under-ordering

            # Enforce minimum and maximum order constraints
            # If the required quantity is less than the minimum order quantity, order the minimum
            if required_qty < min_order_qty:
                order_qty = min_order_qty
                reasoning = (
                    f"{INVENTORY_LEVEL_COLUMN} ({inventory_position:.1f}) < {REORDER_POINT_COLUMN} ({reorder_point:.1f}). "
                    f"{NEEDED}={required_qty}. {ORDERED} Min={min_order_qty:.0f}"
                )
            # If the required quantity is greater than the maximum order quantity, order the maximum
            elif required_qty > max_order_qty:
                backlog_qty += required_qty - max_order_qty  # Carry forward unmet portion
                order_qty = max_order_qty
                reasoning = (
                    f"{INVENTORY_LEVEL_COLUMN} ({inventory_position:.1f}) < {REORDER_POINT_COLUMN} ({reorder_point:.1f}). "
                    f"{NEEDED}={required_qty}. {ORDERED} Max={max_order_qty:.0f} {back_logged}={backlog_qty:.0f}"
                )
            else:
                order_qty = required_qty
                reasoning = (
                    f"{INVENTORY_LEVEL_COLUMN} ({inventory_position:.1f}) < {REORDER_POINT_COLUMN} ({reorder_point:.1f}). "
                    f"{ORDERED}={order_qty:.0f}"
                )
            if order_qty <= 0:
                print(f"[WARNING] Order qty calculated as zero — check constraints! Forcing min order.")
                order_qty = min_order_qty
            # Determine the month when this order will arrive
            expected_arrival_month = add_months(current_month, arrival_delay_months)

            # Add new order to the schedule (stack if other orders already arriving that month)
            stock_on_order[expected_arrival_month] = (
                    stock_on_order.get(expected_arrival_month, 0.0) + order_qty
            )
            # Update tracking of total in-transit stock
            total_stock_on_order += order_qty
            # Add backlog from previous underordering
            reorder_point += backlog_qty

        # 7. Log simulation state
        simulation_log.append({
            REORDER_MONTH_COLUMN: current_month,  # Month: The month the simulation step is for.
            STOCK_LEVEL_AT_MONTH_START_COLUMN: round(stock_start - arriving_stock, 1),
            # Stock @ Start: Stock on hand at the very beginning of the month before any scheduled orders arrive.
            ARRIVING_STOCK_COLUMN: round(arriving_stock, 1),
            # Arriving Stock: Quantity of stock from previous orders that arrived this month.
            STOCK_LEVEL_AFTER_ARRIVAL_COLUMN: round(stock_start, 1),
            # Stock After Arrival: Stock on hand after receiving arriving orders (this is the stock available to meet demand).
            EXPECTED_DEMAND_COLUMN: round(current_demand, 1),  # Demand: Normal predicted demand for this month.
            STOCK_LEVEL_AFTER_DEMAND_COLUMN: round(stock_on_hand, 1),
            # Stock After Demand: Stock on hand after fulfilling this month's demand.
            SHORTFALL_COLUMN: round(shortfall, 1),
            # Shortfall: How much demand could not be met if Stock After Arrival was less than Demand.
            REORDER_POINT_COLUMN: round(reorder_point, 1),
            # ROP: Calculated Reorder Point for this month (Demand During Lead Time + Safety Stock).
            f'{INVENTORY_LEVEL_COLUMN} @ End': round(inventory_position, 1),
            # Inventory Level (End): Inventory Position (Stock After Demand + Total Stock On Order) before making an ordering decision this month. This is compared against the ROP.
            ORDER_DECISION_COLUMN: order_decision,  # Order Decision: Yes/No if an order was placed this month.
            OREDER_QUANTITY_COLUMN: round(order_qty, 1),  # Order Qty: The quantity ordered (if decision was Yes).
            EXPECTED_STOCK_ARRIVAL_COLUMN: expected_arrival_month if expected_arrival_month != "" else outside_translate("functions.no_order_delivery"),
            # Expected Arrival: The month the order placed this month is expected to arrive.
            f'{STOCK_ON_ORDER_COLUMN} @ End': round(total_stock_on_order, 1),
            # Stock On Order (End): Total quantity of stock currently in transit (including any order placed this month) at the end of the month.
            REASONING_COLUMN: reasoning
            # Reasoning: Explanation for the order decision, showing the comparison between Inventory Position and ROP.
        })

    # Display formatted results
    results_df = pd.DataFrame(simulation_log)

    return results_df


def determine_replenishment(
        item_code: str,
        results_df: pd.DataFrame,
        safety_stock: int,
        lead_time: str,
        min_order_qty: int,
        max_order_qty: int,
        current_inventory: int
) -> List[Dict[str, str | int]]:
    """
    Determines a replenishment strategy based on predicted demand.

    This function calculates a replenishment plan based on predicted demand, safety stock,
    lead time, and order quantity constraints.

    Args:
        item_code: The item code.
        results_df: A DataFrame containing predicted demand.
        safety_stock: The safety stock quantity.
        lead_time: The lead time string.
        min_order_qty: The minimum order quantity.
        max_order_qty: The maximum order quantity.

    Returns:
        A list of dictionaries representing the replenishment plan.
    """
    replenishment_plan: List[Dict[str, str | int]] = []
    projected_stock: int = current_inventory  # Start with current stock
    lead_days: int = parse_lead_time(lead_time)
    last_order_date: pd.Timestamp | None = None

    max_order_qty = determine_optimal_order_quantity(min_order_qty, max_order_qty, safety_stock)
    earliest_date: pd.Timestamp = results_df[PREDICTED_DATE_COLUMN].min()

    total_worst = total_best = total_normal = 0  # Track total orders

    for index, row in results_df.iterrows():
        # projected_stock -= row[PREDICTED_COLUMN]  # Reduce stock by predicted demand
        predicted_demand = row[PREDICTED_COLUMN]
        worst_case_demand = row[LOWER_BOUND_COLUMN]
        best_case_demand = row[UPPER_BOUND_COLUMN]

        projected_stock -= predicted_demand  # Reduce stock by expected demand

        reorder_date: pd.Timestamp = max(row[PREDICTED_DATE_COLUMN] - timedelta(days=lead_days),
                                         earliest_date)  # Ensure reorder date is not before earliest data point

        if projected_stock < safety_stock:
            # Calculate required order quantity for each scenario
            required_qty_worst = max(safety_stock - (projected_stock - worst_case_demand), 0)
            required_qty_best = max(safety_stock - (projected_stock - best_case_demand), 0)
            required_qty_normal = max(safety_stock - (projected_stock - predicted_demand), 0)

            # Apply constraints and round up
            order_qty_worst = math.ceil(
                required_qty_worst)  # max(min_order_qty, min(math.ceil(required_qty_worst), max_order_qty))
            order_qty_best = math.ceil(
                required_qty_best)  # max(min_order_qty, min(math.ceil(required_qty_best), max_order_qty))
            order_qty_normal = math.ceil(
                required_qty_normal)  # max(min_order_qty, min(math.ceil(required_qty_normal), max_order_qty))

            # Update total sums
            total_worst += order_qty_worst
            total_best += order_qty_best
            total_normal += order_qty_normal

            if last_order_date and (reorder_date - last_order_date).days < lead_days:
                replenishment_plan[-1][WORSE_CASE_ORDER_COLUMN] += order_qty_worst
                replenishment_plan[-1][BEST_CASE_ORDER_COLUMN] += order_qty_best
                replenishment_plan[-1][ORDER_COLUMN] += order_qty_normal
            else:
                replenishment_plan.append({
                    REORDER_DATE_COLUMN: reorder_date.strftime(DATE_FORMAT),
                    WORSE_CASE_ORDER_COLUMN: order_qty_worst,
                    ORDER_COLUMN: order_qty_normal,
                    BEST_CASE_ORDER_COLUMN: order_qty_best
                })
                last_order_date = reorder_date

            projected_stock += order_qty_worst  # Stock increases based on worst-case order
            # projected_stock += order_qty  # Increase stock after ordering

    replenishment_plan_df: pd.DataFrame = pd.DataFrame(replenishment_plan)
    replenishment_plan_df[REORDER_DATE_COLUMN] = pd.to_datetime(replenishment_plan_df[REORDER_DATE_COLUMN])
    # replenishment_plan_df[REORDER_MONTH_COLUMN] = replenishment_plan_df[REORDER_DATE_COLUMN].dt.to_period("M").dt.to_timestamp()
    replenishment_plan_df[REORDER_MONTH_COLUMN] = replenishment_plan_df[REORDER_DATE_COLUMN].dt.strftime(MONTH_FORMAT)
    monthly_plan: pd.DataFrame = replenishment_plan_df.groupby(REORDER_MONTH_COLUMN).sum(
        numeric_only=True).reset_index()

    # Add total row
    total_row = {REORDER_MONTH_COLUMN: 'TOTAL', WORSE_CASE_ORDER_COLUMN: total_worst, ORDER_COLUMN: total_normal,
                 BEST_CASE_ORDER_COLUMN: total_best}
    monthly_plan = pd.concat([monthly_plan, pd.DataFrame([total_row])], ignore_index=True)

    # Tracking Variables
    carry_forward = 0
    orders = []
    extra_needed = []

    for index, row in replenishment_plan_df.iterrows():
        demand = row[ORDER_COLUMN] + carry_forward  # Include previous shortfall
        order = 0
        carry_forward = 0

        # If total demand is below the minimum order, roll it over
        if demand < min_order_qty:
            carry_forward = demand
            extra_needed.append(carry_forward)
        else:
            # Order within limits
            order = min(demand, max_order_qty)
            if demand > max_order_qty:
                carry_forward = demand - max_order_qty  # Roll over excess
            extra_needed.append(0)

        orders.append(order)

    # Add results to DataFrame
    replenishment_plan_df["Ordered"] = orders
    replenishment_plan_df["Shortfall (Carried)"] = [0] + extra_needed[:-1]  # Carry over starts from 2nd month
    replenishment_plan_df["Next Month Extra Need"] = extra_needed

    return monthly_plan, replenishment_plan_df  # Return the replenishment plan list of dictionaries for the item.


def is_forecast_feasible(item_code: str) -> bool:
    """
    Checks whether forecasting is feasible for a given item.
    Conditions:
    - At least 8 weeks of demand data
    - Non-zero and variable demand
    - No major weekly gaps in data
    - No stockouts during the current month
    """
    try:
        # === 1. Demand Check ===
        demand_df: pd.DataFrame = get_demand_data(item_code)
        if len(demand_df) < 8:
            print(f"⚠️ Not enough data to forecast for {item_code} (needs ≥ 8 weeks).")
            return False

        total_demand: float = demand_df[MENGE_COLUMN].sum()
        unique_demand_values: int = demand_df[MENGE_COLUMN].nunique()

        if total_demand == 0 or unique_demand_values <= 1:
            print(f"⚠️ Demand for {item_code} is zero or unchanging.")
            return False

        # === 2. Continuity Check ===
        expected_range: pd.DatetimeIndex = pd.date_range(
            start=demand_df[DATE_COLUMN].min(),
            end=demand_df[DATE_COLUMN].max(), freq='W'
        )
        data_continuity_ratio: float = len(demand_df) / len(expected_range)

        if data_continuity_ratio < 0.75:
            print(f"⚠️ {item_code} demand data has large weekly gaps.")
            return False

        # === 3. Stockout Check for current month ===
        stock_df: pd.DataFrame
        metadata: str
        stock_df, metadata = get_stock_data(item_code)

        # 🧪 Debugging print — make sure the column exists!
        print(f"Stock columns available: {stock_df.settings.tolist()}")
        print(f"Checking stock values in column: {MENGE_COLUMN}")

        if MENGE_COLUMN not in stock_df.columns:
            print(f"⚠️ Column '{MENGE_COLUMN}' not found in stock data for {item_code}.")
            return False

        recent_stock: pd.DataFrame = stock_df.tail(4)
        if (recent_stock[MENGE_COLUMN] >= 0).any():
            print(f"⚠️ {item_code} had recent stockouts. Demand may be censored.")
            return False

        return True

    except Exception as e:
        logging.error(f"🚨 Feasibility check failed for {item_code}: {e}")
        return False


def check_replenishment_violations(
        item_code: str,
        results_df: pd.DataFrame,
        safety_stock: int,
        lead_time: str,
        min_order_qty: int,
        max_order_qty: int
) -> List[Dict[str, Any]]:
    """
    Checks if forecasted demand violates the replenishment time period.

    A violation occurs when:
    - Forecasted demand causes stock to drop **below safety stock**
    - Before a new order **arrives** (lead time constraint)

    Returns:
        A list of violations, each with date, projected stock, and reorder recommendation.
    """
    violations: List[Dict[str, Any]] = []
    projected_stock: int = safety_stock
    lead_days: int = parse_lead_time(lead_time)

    for _, row in results_df.iterrows():
        date: pd.Timestamp = row[PREDICTED_DATE_COLUMN]
        forecasted_demand: int = row[PREDICTED_COLUMN]
        projected_stock -= forecasted_demand

        # If projected stock falls below safety stock, check replenishment feasibility
        if projected_stock < safety_stock:
            reorder_date: pd.Timestamp = date - timedelta(days=lead_days)
            required_qty: int = max(safety_stock - projected_stock, min_order_qty)
            order_qty: int = min(required_qty, max_order_qty)

            violations.append({
                "Date": date.strftime(DATE_FORMAT),
                "Forecasted Demand": forecasted_demand,
                "Projected Stock": projected_stock,
                "Safety Stock": safety_stock,
                "Reorder Date": reorder_date.strftime(DATE_FORMAT),
                "Recommended Order Qty": order_qty
            })

    return violations


def compute_storage_cost(stock_levels_df: pd.DataFrame, cost_per_unit: float) -> float:
    """
    Calculates storage cost based on stock levels.
    :param stock_levels_df: DataFrame with stock levels over time.
    :param cost_per_unit: Cost per unit per time step.
    :return: Total storage cost.
    """
    total_storage_cost: float = (stock_levels_df['stock_level'] * cost_per_unit).sum()
    return total_storage_cost


def compute_logistics_cost(demand_df: pd.DataFrame, cost_per_order: float) -> float:
    """
    Estimates logistics cost based on demand peaks.
    :param demand_df: DataFrame with demand data.
    :param cost_per_order: Cost per order.
    :return: Total logistics cost.
    """
    num_orders: int = len(demand_df)  # Assuming each demand entry represents an order
    total_logistics_cost: float = num_orders * cost_per_order
    return total_logistics_cost


def compute_opportunity_cost(stock_levels_df: pd.DataFrame, safety_stock: float, penalty_per_stockout: float) -> float:
    """
    Computes opportunity cost due to stockouts.
    :param stock_levels_df: DataFrame with stock levels.
    :param safety_stock: Safety stock threshold.
    :param penalty_per_stockout: Penalty per stockout event.
    :return: Total opportunity cost.
    """
    stockout_events: int = (stock_levels_df['stock_level'] < safety_stock).sum()
    total_opportunity_cost: float = stockout_events * penalty_per_stockout
    return total_opportunity_cost

def get_article_dropdown_options():
    """
    Reads the posten file and extracts all unique article numbers ('ArtikelNr')
    to generate a list of options suitable for a Dash dropdown.

    Returns:
        list: A list of dictionaries, where each dictionary has 'label'
              and 'value' keys, both set to a unique article number.
              Returns an empty list if an error occurs.
    """
    try:
        # Read the main data file using the path from your config
        df = load_posten_file()
        df = df[df[ARTICLE_NUMBER_COLUMN].str.startswith(ITEM_START, na=False)]
        # Get unique, non-null article numbers from the specified column
        unique_articles = df[ARTICLE_NUMBER_COLUMN].dropna().unique()

        # Sort the articles for a consistent and user-friendly dropdown
        sorted_articles = sorted(list(unique_articles))

        # Create the list of dictionaries in the format Dash requires
        dropdown_options = [{'label': str(article), 'value': str(article)} for article in sorted_articles]

        return dropdown_options

    except FileNotFoundError:
        logging.error(f"Error creating article dropdown: The file '{POSTEN_FILTERED_FILE}' was not found.")
        return []
    except KeyError:
        logging.error(f"Error creating article dropdown: Column '{ARTICLE_NUMBER_COLUMN}' not found in the file.")
        return []

def compute_total_cost(
        stock_levels_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        safety_stock: float,
        cost_per_unit: float = 0.5,
        cost_per_order: float = 10,
        penalty_per_stockout: float = 100,
) -> Tuple[float, float, float, float]:
    """
    Computes total cost including storage, logistics, and opportunity costs.
    :param stock_levels_df: DataFrame with stock levels over time.
    :param demand_df: DataFrame with demand data.
    :param safety_stock: Safety stock threshold.
    :param cost_per_unit: Storage cost per unit per time step.
    :param cost_per_order: Cost per order processed.
    :param penalty_per_stockout: Penalty per stockout event.
    :return: Tuple with total cost, storage cost, logistics cost, opportunity cost.
    """
    storage_cost: float = compute_storage_cost(stock_levels_df, cost_per_unit)
    logistics_cost: float = compute_logistics_cost(demand_df, cost_per_order)
    opportunity_cost: float = compute_opportunity_cost(stock_levels_df, safety_stock, penalty_per_stockout)

    total_cost: float = storage_cost + logistics_cost + opportunity_cost

    return {
        "Storage Cost": storage_cost,
        "Logistics Cost": logistics_cost,
        "Opportunity Cost": opportunity_cost,
        "Total Cost": total_cost
    }

def calculate_metrics(y_true,y_pred,lower_bound,upper_bound):
    rmse = root_mean_squared_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mpiw = np.mean(upper_bound - lower_bound)

    # PICP: percentage of true values within prediction interval
    within_interval = np.logical_and(y_pred >= lower_bound, y_pred <= upper_bound)
    picp = np.mean(within_interval)  # in percentage
    return round(rmse,2),round(mse,2),round(mae,2),round(mpiw,2),round(picp,2)


def generate_table_for_line_and_sunburst(starting_year: str, ending_year: str, itemNr: str) -> pd.DataFrame:
    """

    Loads the 'posten_filtered.csv' file and generates a source breakdown table between the specified years.
    :param starting_year: in 'YYYY' format
    :param ending_year: in 'YYYY' format
    :param itemNr: theARTICLE_NUMBER_COLUMN to filter on
    :return: a DataFrame aggregated by month showing total usage and breakdown by source
    """
    # This function filters the postings for CONSUMPTION_POSTEN_TYPE entries, confines them to the given year range,
    # sums quantities per source per date, then aggregates by month for the specified itemNr.
    # The result contains, for each month:
    #   - total_quantity: total used quantity for that item
    #   - source_pairs: dict mapping each source number to its subtotal that month
    #   - verbrauch: percentage of the grand total used in that month
    #   -ARTICLE_NUMBER_COLUMN: the item number (same for all rows)

    posten_df = load_posten_file()


    verbrauch_items: pd.DataFrame = posten_df[posten_df[POSTEN_TYPE_COLUMN] == CONSUMPTION_POSTEN_TYPE].copy()
    verbrauch_items[QUANTITY_COLUMN] = verbrauch_items[QUANTITY_COLUMN].abs()
    verbrauch_items[DATE_COLUMN] = pd.to_datetime(verbrauch_items[DATE_COLUMN])
    source_breakdown_table_df: pd.DataFrame = verbrauch_items.groupby(
        [DATE_COLUMN, ARTICLE_NUMBER_COLUMN, SOURCE_NUMBER_COLUMN]
    )[QUANTITY_COLUMN].sum().reset_index()
    source_breakdown_table_df.rename(columns={QUANTITY_COLUMN: 'TotalQuantityFromSource'}, inplace=True)

    starting_date = "01-01-" + starting_year
    ending_date = "12-12-" + ending_year
    source_breakdown_df = source_breakdown_table_df[
        (source_breakdown_table_df[DATE_COLUMN] >= starting_date) &
        (source_breakdown_table_df[DATE_COLUMN] <= ending_date)
        ]
    source_breakdown_table = source_breakdown_df

    source_breakdown_table = source_breakdown_table[source_breakdown_table[ARTICLE_NUMBER_COLUMN] == itemNr].copy()

    total_quantity = source_breakdown_table["TotalQuantityFromSource"].sum()
    source_breakdown_table["month"] = source_breakdown_table[DATE_COLUMN].dt.to_period("M").dt.to_timestamp()
    agg = (
        source_breakdown_table
        .groupby("month")
        .apply(lambda g: pd.Series({
            "total_quantity": g["TotalQuantityFromSource"].sum(),
            "source_pairs": g
            .groupby(SOURCE_NUMBER_COLUMN)["TotalQuantityFromSource"]
            .sum()
            .to_dict()
        }))
        .reset_index()
    )
    agg[CONSUMPTION_POSTEN_TYPE] = (agg["total_quantity"] / total_quantity) * 100
    agg[CONSUMPTION_POSTEN_TYPE] = agg[CONSUMPTION_POSTEN_TYPE].round(2)
    agg = agg[(agg != 0).all(axis=1)]
    agg[ARTICLE_NUMBER_COLUMN] = itemNr
    return agg


# --- Data Loading and Processing for Treemap ---
def create_treemap_figure(starting_year: str, ending_year: str) -> go.Figure:
    """
    Loads data, processes it, and generates a Plotly Express treemap figure.
    :param starting_year (str): The starting year for filtering the data (format 'YYYY').
    :param ending_year (str): The ending year for filtering the data (format 'YYYY').
    :return:
        go.Figure: A Plotly treemap where eachARTICLE_NUMBER_COLUMN is sized by its usage rate
        and colored according to its rank among the filtered items.
    """

    # This function:
    # 1) Reads the cleaned 'posten_filtered.csv' dataset (with fallbacks if not found).
    # 2) Filters entries by year range and by the relevant Postenart types.
    # 3) Aggregates usage and incoming quantities perARTICLE_NUMBER_COLUMN.
    # 4) Computes usage_rate = (usage / total_incoming) * 100, filters and ranks the top items.
    # 5) Maps eachARTICLE_NUMBER_COLUMN’s usage_rate rank to one of 40 predefined colors (blue→red).
    # 6) Builds and returns a Plotly treemap figure using those colors.

    posten_df = load_posten_file()

    verbrauch_items: pd.DataFrame = posten_df[posten_df[POSTEN_TYPE_COLUMN].isin([CONSUMPTION_POSTEN_TYPE])].copy()
    verbrauch_items = verbrauch_items[
        (verbrauch_items[DATE_COLUMN] >= starting_year) & (verbrauch_items[DATE_COLUMN] <= ending_year)]
    verbrauch_items[QUANTITY_COLUMN] = verbrauch_items[QUANTITY_COLUMN].abs()

    zugang_items: pd.DataFrame = posten_df[posten_df[POSTEN_TYPE_COLUMN].isin([ZUGANG_POSTEN_TYPE])].copy()
    zugang_items = zugang_items[
        (zugang_items[DATE_COLUMN] >= starting_year) & (zugang_items[DATE_COLUMN] <= ending_year)]
    einkauf_items: pd.DataFrame = posten_df[posten_df[POSTEN_TYPE_COLUMN].isin([EINKAUF_POSTEN_TYPE])].copy()
    einkauf_items = einkauf_items[
        (einkauf_items[DATE_COLUMN] >= starting_year) & (einkauf_items[DATE_COLUMN] <= ending_year)]
    istmeldung_items = posten_df[posten_df[POSTEN_TYPE_COLUMN].isin([ISTMELDUNG_POSTEN_TYPE])].copy()
    istmeldung_items = istmeldung_items[
        (istmeldung_items[DATE_COLUMN] >= starting_year) & (istmeldung_items[DATE_COLUMN] <= ending_year)]

    verbrauch_items[DATE_COLUMN] = pd.to_datetime(verbrauch_items[DATE_COLUMN])
    zugang_items[DATE_COLUMN] = pd.to_datetime(zugang_items[DATE_COLUMN])
    einkauf_items[DATE_COLUMN] = pd.to_datetime(einkauf_items[DATE_COLUMN])
    istmeldung_items[DATE_COLUMN] = pd.to_datetime(istmeldung_items[DATE_COLUMN])

    verbrauch_items = verbrauch_items.groupby(
        [ARTICLE_NUMBER_COLUMN]
    )[QUANTITY_COLUMN].sum().reset_index()
    verbrauch_items = verbrauch_items.rename(columns={QUANTITY_COLUMN: 'usage'})

    zugang_items = zugang_items.groupby(
        [ARTICLE_NUMBER_COLUMN]
    )[QUANTITY_COLUMN].sum().reset_index()
    zugang_items = zugang_items.rename(columns={QUANTITY_COLUMN: 'zugang'})

    einkauf_items = einkauf_items.groupby(
        [ARTICLE_NUMBER_COLUMN]
    )[QUANTITY_COLUMN].sum().reset_index()
    einkauf_items = einkauf_items.rename(columns={QUANTITY_COLUMN: 'einkauf'})

    istmeldung_items = istmeldung_items.groupby(
        [ARTICLE_NUMBER_COLUMN]
    )[QUANTITY_COLUMN].sum().reset_index()
    istmeldung_items = istmeldung_items.rename(columns={QUANTITY_COLUMN: 'istmeldung'})

    merged = pd.merge(einkauf_items, zugang_items, on=ARTICLE_NUMBER_COLUMN, how="outer", suffixes=('_df1', '_df2'))
    merged = pd.merge(merged, istmeldung_items, on=ARTICLE_NUMBER_COLUMN, how="outer")
    merged['Total_Value'] = (
        merged[['einkauf', 'zugang', 'istmeldung']].fillna(0)
        .sum(axis=1)
    )

    final = merged[[ARTICLE_NUMBER_COLUMN, "Total_Value"]]
    final.rename(columns={'Total_Value': 'zugang'}, inplace=True)

    df = verbrauch_items.merge(final, on=ARTICLE_NUMBER_COLUMN, how="inner")
    df = df[df["usage"] > 1000]
    df["usage_rate"] = (df["usage"] / df["zugang"]) * 100
    df = df[df["usage_rate"] < 100]
    df = df.sort_values('usage_rate', ascending=False).head(350)
    df["usage_rate"] = df["usage_rate"].round(3)
    df.drop(columns=["zugang", "usage"], inplace=True)
    table_len = len(df)
    df_treemap = df.reset_index(drop=True)
    df_treemap[ARTICLE_NUMBER_COLUMN] = df_treemap[ARTICLE_NUMBER_COLUMN].astype(str)
    hex_palette = [
        "#1500ff", "#005a9d", "#005d7d", "#004a4c", "#004647", "#004445", "#004443", "#004343", "#004342", "#004442",
        "#004442", "#004543", "#004745", "#004e4b", "#006064", "#006367", "#006666", "#006862", "#006b5c", "#006d55",
        "#00704d", "#007344", "#00753c", "#007833", "#007b27", "#007e0e", "#008000", "#188000", "#3e7f00", "#557d00",
        "#697b00", "#7b7800", "#8d7400", "#9f6f00", "#b16800", "#c35e00", "#d55100", "#e63f00", "#f32b00", "#ff0000"
    ]
    num_unique_rates = len(df_treemap["usage_rate"].unique())

    if num_unique_rates > 1:
        # Original calculation is safe when there are multiple unique rates
        df_treemap["color_index"] = (
                (df_treemap["usage_rate"].rank(method="min") - 1)
                / (num_unique_rates - 1) # Use the stored variable to avoid re-calculation
                * (len(hex_palette) - 1)
        ).round(0).astype(int)
    else:
        # Handle the edge case of 0 or 1 unique rates by assigning a default color index
        df_treemap["color_index"] = 0

    artikel_colors = {row[ARTICLE_NUMBER_COLUMN]: hex_palette[row["color_index"]] for _, row in df_treemap.iterrows()}

    template = translate(_language, "functions.treemap_title")
    title = template.format(
        count=table_len,
        start=starting_year,
        end=ending_year
    )
    article_var = translate(_language, "functions.treemapARTICLE_NUMBER_COLUMN")
    usage_rate_var = translate(_language, "functions.treemap_usage_rate")
    df_treemap = df_treemap.rename(columns={
        ARTICLE_NUMBER_COLUMN: article_var,
        "usage_rate": usage_rate_var
    })

    fig = px.treemap(
        df_treemap,
        path=[article_var],
        values=usage_rate_var,
        color=article_var,
        color_discrete_map=artikel_colors,
        title=title,
    )

    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25)
    )
    fig.update_traces(marker=dict(cornerradius=5))

    return fig


def create_demand_over_time_figure(itemNr: str, starting_year: str, ending_year: str) -> go.Figure:
    """
    Loads usage data for a singleARTICLE_NUMBER_COLUMN between two years and produces
    a time-series line chart of monthly percentage usage.

    This function:
      1. Calls generate_table_for_line_and_sunburst() to get per-month usage breakdown.
      2. Drops the raw source breakdown columns, keeping only the monthly percentage.
      3. Renames the percentage column for clarity.
      4. Builds a Plotly Express line chart with markers, formatting the x-axis as YYYY-MM.
      5. Returns the configured go.Figure for embedding in Dash or similar.

    :param itemNr (str):      TheARTICLE_NUMBER_COLUMN for which to plot usage over time.
    :param starting_year (str): The starting year (YYYY) of the time window.
    :param ending_year (str):   The ending year (YYYY) of the time window.
    :return:
        go.Figure: A Plotly Express line chart showing percentual usage by month.
    """

    agg = generate_table_for_line_and_sunburst(starting_year, ending_year, itemNr)
    if agg.empty:
        logging.info("agg empty")
    agg = agg.drop(columns=["source_pairs", "total_quantity"])
    agg.rename(columns={CONSUMPTION_POSTEN_TYPE: "Percentage"}, inplace=True)

    template = translate(_language, "functions.line_title")
    title = template.format(
        itemNr=itemNr,
        starting_year=starting_year,
        ending_year=ending_year
    )
    month_var = translate(_language, "functions.line_month")
    percentage_var = translate(_language, "functions.line_percentage")
    agg.rename(columns={"month": month_var, "Percentage": percentage_var}, inplace=True)
    fig = px.line(
        agg,
        x=month_var,
        y=percentage_var,
        markers=True,
        title=title,
        labels={'date': 'Date', QUANTITY_COLUMN: QUANTITY_COLUMN},
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        xaxis_tickformat='%Y-%m'
    )
    return fig


def create_sunburst_figure(itemNr: str, date: str, starting_year: str, ending_year: str) -> go.Figure:
    """
    Loads data, processes it, and generates a Plotly Express sunburst figure.
    :param itemNr:         TheARTICLE_NUMBER_COLUMN for which to build the sunburst.
    :param date:           The specific month to visualize, in 'YYYY-MM-DD' format.
    :param starting_year:  The starting year of the data window ('YYYY').
    :param ending_year:    The ending year of the data window ('YYYY').
    :return:               A Plotly Figure containing the sunburst chart.
    """

    # This function builds a Plotly Sunburst chart for a specificARTICLE_NUMBER_COLUMN on a given month.
    # It:
    # 1) Retrieves a monthly breakdown of total usage and per-source subtotals via
    #    generate_table_for_line_and_sunburst().
    # 2) Filters the aggregated data to the exact month (date) requested.
    # 3) Computes each source’s percentage share of that month’s total usage.
    # 4) Categorizes sources by their prefix (‘1-’, ‘2-’, etc.), grouping all others as “rest”.
    # 5) Builds and returns a sunburst figure with layers:
    #      • root =ARTICLE_NUMBER_COLUMN
    #      • middle = Category
    #      • leaves =SOURCE_NUMBER_COLUMN
    #    sized by percent usage and colored by Category.

    agg = generate_table_for_line_and_sunburst(starting_year, ending_year, itemNr)
    agg = agg[agg["month"] == date]
    total_quantity = agg["total_quantity"].sum()
    df_here: pd.DataFrame = pd.DataFrame(columns=[SOURCE_NUMBER_COLUMN, QUANTITY_COLUMN, "Category"])
    for s, q in agg["source_pairs"].iloc[0].items():

        c = s[:2]
        if c not in (PROD_START, '2-', '3-', '4-', '7-',PART_START):
            c = "rest"

        # Compute percentage of the grand total and build a single-row DataFrame
        pct = q / total_quantity * 100
        tmp_df = pd.DataFrame({
            SOURCE_NUMBER_COLUMN: [s],
            QUANTITY_COLUMN: [pct],
            "Category": [c],
        })

        # Append to df_here
        df_here = pd.concat([df_here, tmp_df], ignore_index=True)
    df_here[ARTICLE_NUMBER_COLUMN] = itemNr
    df_here[QUANTITY_COLUMN] = df_here[QUANTITY_COLUMN].round(0)
    df_here.rename(columns={QUANTITY_COLUMN: "Percentage"}, inplace=True)
    df_nonzero = df_here[df_here["Percentage"] > 0]

    template = translate(_language, "functions.sunburst_title")
    title = template.format(
        itemNr=itemNr,
        date = date[:7]
    )
    article_var = translate(_language, "functions.treemapARTICLE_NUMBER_COLUMN")
    category_var = translate(_language, "functions.category")
    percentage_var = translate(_language, "functions.line_percentage")

    df_nonzero = df_nonzero.rename(columns={
        "Category": category_var,
        ARTICLE_NUMBER_COLUMN: article_var,
        "Percentage": percentage_var,
    })
    fig = px.sunburst(
        df_nonzero,
        path=[article_var, category_var, SOURCE_NUMBER_COLUMN],
        values=percentage_var,
        color=category_var,
        color_continuous_scale="Cividis"
    )

    fig.update_layout(
        margin=dict(t=40, l=10, r=10, b=10),
        title=title,
    )

    return fig

def create_configuration_with_stats(posten: pd.DataFrame, prod: str, threshold: float = 0.8) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Creates a configuration of required quantities and separates standard vs. rare items
    based on their occurrence frequency in the usage history.

    Returns:
        - full_config: DictARTICLE_NUMBER_COLUMN, Required Quantity]
        - standard_config: DictARTICLE_NUMBER_COLUMN, Required Quantity] (appears >= threshold of the time)
        - rare_items_info: DictARTICLE_NUMBER_COLUMN, {'occurrence_ratio': float, 'required_quantity': float}]
    """
    # Filter and prepare data
    all_items = posten.copy()
    all_items = all_items[all_items[SOURCE_NUMBER_COLUMN] == prod]
    all_items = all_items[all_items[POSTEN_TYPE_COLUMN] == CONSUMPTION_POSTEN_TYPE]
    all_items[QUANTITY_COLUMN] = all_items[QUANTITY_COLUMN].abs()
    all_items = all_items[all_items[ARTICLE_NUMBER_COLUMN].str.startswith(ITEM_START, na=False)] #only look at 6- items
    all_items[DATE_COLUMN] = pd.to_datetime(all_items[DATE_COLUMN])

    # Group by Date andARTICLE_NUMBER_COLUMN
    artikel_quantity_per_date = all_items.groupby([DATE_COLUMN, ARTICLE_NUMBER_COLUMN])[QUANTITY_COLUMN].sum().reset_index()
    artikel_quantity_per_date.rename(columns={QUANTITY_COLUMN: 'TotalQuantity'}, inplace=True)

    # Pivot the table
    pivoted_table = artikel_quantity_per_date.pivot_table(
        index=DATE_COLUMN,
        columns=ARTICLE_NUMBER_COLUMN,
        values='TotalQuantity',
        fill_value=0
    )

    # Get total number of unique dates
    total_days = pivoted_table.shape[0]

    print(f"\nCreating configuration forSOURCE_NUMBER_COLUMN: {prod}...")

    full_config = {}
    standard_config = {}
    rare_items_info = {}

    for artikel_nr in pivoted_table.columns:
        quantities = pivoted_table[artikel_nr]
        occurrence_ratio = (quantities > 0).sum() / total_days  # How often it's used

        # Determine required_quantity based on occurrence and non-zero quantities
        if occurrence_ratio == 0:
            # If an item never occurred with quantity > 0, its required quantity is 0
            required_quantity = 0.0
        else:
            # Filter out zero quantities for mode calculation to get actual usage quantity
            # Ensure we only consider quantities that are truly positive.
            non_zero_quantities = quantities[quantities > 0]

            if not non_zero_quantities.empty:
                if non_zero_quantities.nunique() == 1:
                    # If there's only one unique non-zero quantity, use that
                    required_quantity = non_zero_quantities.iloc[0]
                else:
                    # Otherwise, find the mode of non-zero quantities
                    required_quantity = non_zero_quantities.mode().iloc[0]
            else:
                # This case should ideally not be reached if occurrence_ratio > 0,
                # but as a safeguard, if somehow no positive quantities are found,
                # set to 0.0
                required_quantity = 0.0

        full_config[artikel_nr] = required_quantity

        if occurrence_ratio >= threshold:
            standard_config[artikel_nr] = required_quantity
        else:
            rare_items_info[artikel_nr] = {
                'occurrence_ratio': round(occurrence_ratio, 2),
                'required_quantity': required_quantity
            }

    return full_config, standard_config, rare_items_info

def create_br_dict():
    """
    Creates a dictionary mapping unique series ("Baureihe") to their respective
    data from the `BR_MAPS` dataset. The dataset is filtered to include only entries
    whose "Nr." column values begin with PROD_START.

    The function reads a CSV file, filters specific rows, identifies unique train
    series, and constructs a dictionary where each unique series maps to a subset
    of the filtered dataset corresponding to that series.
    :return: A dictionary where keys are unique series (Baureihe) extracted
        from the dataset, and products are DataFrame subsets filtered by those train
        series. dict[braureihe:df[productno, baureihe]
    :rtype: dict
    """
    br = pd.read_csv(BR_MAPS)
    br = br.copy().loc[br['Nr.'].str.startswith(PROD_START, na=False)]
    br_list = br['Baureihe'].unique()
    br_list = [item for item in br_list if pd.notna(item)] # remove nan values
    br_dict = {}
    for i in br_list:
        br_dict[i] = br[br["Baureihe"] == i]
    return br_dict


def create_product_distribution():
    """
    :return:
    """
    df = load_posten_file()
    br_dict = create_br_dict()
    df = df[df[POSTEN_TYPE_COLUMN] == "Istmeldung"]
    df = df[df[ARTICLE_NUMBER_COLUMN].str.startswith(PROD_START, na=False)]
    df = df.groupby(ARTICLE_NUMBER_COLUMN)[QUANTITY_COLUMN].sum().reset_index()
    all_prods_with_br = pd.concat(list(br_dict.values()), ignore_index=True)
    merged_df = pd.merge(all_prods_with_br, df, left_on='Nr.', right_on=ARTICLE_NUMBER_COLUMN)
    merged_df.drop(columns=['Nr.'], inplace=True)
    merged_df["TotalQuantityBR"] = merged_df.groupby("Baureihe")[QUANTITY_COLUMN].transform("sum")
    merged_df['PctProduct'] = round(merged_df[QUANTITY_COLUMN] / merged_df['TotalQuantityBR'],4)
    merged_df.rename(columns={ARTICLE_NUMBER_COLUMN:"ProduktNr"}, inplace=True)
    return merged_df


def generate_baureihe_summary(baureihe_key: str, threshold: float = 0.5) -> pd.DataFrame:
    """
    Generate a summary table like the image for a specific Baureihe.

    Args:
        br_dict: Dict of {Baureihe: DataFrame of product Nrs}
        posten: Main usage DataFrame
        baureihe_key: Baureihe identifier (e.g., "0160.1")
        threshold: Frequency threshold for standard vs rare item classification

    Returns:
        Summary DataFrame with:
            - Item
            - Produkt
            - Wie oft eingebaut
            - Wahrscheinlichkeit wie oft sie eingebaut wurde in den N Produkten
    """
    posten = load_posten_file()
    br_dict = create_br_dict()
    product_df = br_dict.get(baureihe_key)
    if product_df is None:
        raise ValueError(f"Baureihe '{baureihe_key}' not found in br_dict.")

    product_ids = product_df["Nr."].tolist()
    total_products = len(product_ids)

    # Dict[(item, product)] = {'used_in': bool, 'qty': int}
    summary_data = []

    for prod in product_ids:
        full_config, _, rare_items_info = create_configuration_with_stats(posten, prod, threshold=threshold)

        # Only keepARTICLE_NUMBER_COLUMN starting with ITEM_START
        full_config = {k: v for k, v in full_config.items() if k.startswith(ITEM_START)}

        for artikel_nr, qty in full_config.items():
            summary_data.append((artikel_nr, prod, qty))

    # Now aggregate item usage across products
    result_rows = []
    from collections import Counter
    usage_counter = Counter()         # how many times each item appears (across all products)
    product_item_pairs = set()

    for artikel, produkt, qty in summary_data:
        usage_counter[artikel] += 1
        product_item_pairs.add((artikel, produkt))

    for artikel, produkt in product_item_pairs:
        # Filter entries for this item + product
        qty_list = [qty for (a, p, qty) in summary_data if a == artikel and p == produkt]
        # Mode of quantities or first if all same
        most_common_qty = Counter(qty_list).most_common(1)[0][0]
        usage_ratio = usage_counter[artikel] / total_products

        result_rows.append({
            "Product": produkt,
            "Component": artikel,
            f"Komponente ist Bestandteil des Produkts in ": f"{usage_ratio * 100:.0f}% ",
            "Gewöhnlicher Produktionskoeffizient": most_common_qty,
        })

    result_df = pd.DataFrame(result_rows)
    result_df = result_df.sort_values(by=["Product", "Komponente ist Bestandteil des Produkts in ", "Component"],ascending= [True,False,True]).reset_index(drop=True)

    return result_df


def generate_component_quantity_probabilities2(
        baureihe_key: str,
        threshold: float = 0.5,
        baureihe_usage_filter: Optional[List[float]] = None,
        menge_frequency_filter: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Generates probabilities for component quantities within a specified 'Baureihe' key, while accounting
    for additional filtering criteria on the usage and frequency of quantities.

    This function processes data from various sources, calculates probabilities for
    component quantities within a specified 'Baureihe', and optionally applies filters
    based on specified thresholds. It also includes functionality to compute weighted
    average quantities per component.

    :param baureihe_key: Specifies the 'Baureihe' for which the component quantity probabilities
        should be calculated.
    :param threshold: Represents the threshold value used during configuration generation.
    :param baureihe_usage_filter: A list containing two float values [min, max] that define the range
        for filtering based on 'Baureihe-Wide Usage (%)'.
    :param menge_frequency_filter: A list containing two float values [min, max] for filtering
        based on 'Probability of Menge (%)'.
    :return: A pandas DataFrame with calculated probabilities and additional statistical
        attributes for component quantities.
    """
    try:
        posten_df = load_posten_file()
        if posten_df.empty:
            logging.warning("Posten file is empty for component quantity probabilities.")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading posten file: {e}", exc_info=True)
        return pd.DataFrame()

    try:
        br_dict_data = create_br_dict()
    except Exception as e:
        logging.error(f"Error creating Baureihe dictionary: {e}", exc_info=True)
        return pd.DataFrame()

    product_df_for_baureihe = br_dict_data.get(baureihe_key)
    if product_df_for_baureihe is None or product_df_for_baureihe.empty:
        logging.warning(f"Baureihe '{baureihe_key}' not found or has no products.")
        return pd.DataFrame()

    product_ids_in_baureihe = product_df_for_baureihe["Nr."].tolist()
    total_products_in_baureihe = len(product_ids_in_baureihe)
    if total_products_in_baureihe == 0:
        logging.warning(f"No products in Baureihe '{baureihe_key}'.")
        return pd.DataFrame()

    component_baureihe_usage_counter = Counter()
    for prod_id_for_summary in product_ids_in_baureihe:
        full_config_for_summary, _, _ = create_configuration_with_stats(posten_df, prod_id_for_summary, threshold)
        # Ensure full_config_for_summary is a dictionary
        if not isinstance(full_config_for_summary, dict):
            logging.warning(f"Configuration for product {prod_id_for_summary} is not a dict, skipping.")
            continue
        filtered_config = {k: v for k, v in full_config_for_summary.items() if isinstance(k, str) and k.startswith(ITEM_START)}
        for component_key in filtered_config.keys():
            component_baureihe_usage_counter[component_key] += 1

    all_combinations = []

    for product_id in product_ids_in_baureihe:
        prod_usage_df = posten_df[
            (posten_df[SOURCE_NUMBER_COLUMN] == product_id) &
            (posten_df[POSTEN_TYPE_COLUMN] == CONSUMPTION_POSTEN_TYPE)
            ].copy()
        if prod_usage_df.empty: continue

        prod_usage_df[QUANTITY_COLUMN] = prod_usage_df[QUANTITY_COLUMN].abs()
        if ARTICLE_NUMBER_COLUMN not in prod_usage_df.columns: continue
        prod_usage_df = prod_usage_df[prod_usage_df[ARTICLE_NUMBER_COLUMN].str.startswith(ITEM_START, na=False)]
        if DATE_COLUMN not in prod_usage_df.columns: continue
        try:
            prod_usage_df[DATE_COLUMN] = pd.to_datetime(prod_usage_df[DATE_COLUMN])
        except Exception:
            logging.warning(f"Could not parse BuchungsDatum for product {product_id}, skipping.")
            continue

        daily_component_usage = prod_usage_df.groupby([DATE_COLUMN, ARTICLE_NUMBER_COLUMN])[QUANTITY_COLUMN].sum().reset_index()
        daily_component_usage = daily_component_usage[daily_component_usage[QUANTITY_COLUMN] > 0]
        if daily_component_usage.empty: continue

        component_quantities_lists = daily_component_usage.groupby(ARTICLE_NUMBER_COLUMN)[QUANTITY_COLUMN].apply(list).to_dict()

        for component, quantities_list in component_quantities_lists.items():
            if not quantities_list: continue
            quantity_counts = Counter(quantities_list)
            total_observations_for_component_in_product = len(quantities_list)

            overall_baureihe_usage_numeric = (
                    (component_baureihe_usage_counter[component] / total_products_in_baureihe) * 100
            ) if total_products_in_baureihe > 0 else 0.0

            for quantity_value, count in quantity_counts.items():
                probability_of_quantity_numeric = (count / total_observations_for_component_in_product) * 100
                all_combinations.append({
                    "Produkt": product_id,
                    "Komponente": component,
                    "Baureihe-Wide Usage (%)": round(overall_baureihe_usage_numeric, 2),
                    "Produktionskoeffizient (Menge)": round(quantity_value, 4),
                    "Probability of Menge (%)": round(probability_of_quantity_numeric, 2),
                })

    if not all_combinations:
        logging.info(f"No component usage data found for Baureihe {baureihe_key} to create initial DataFrame.")
        return pd.DataFrame()

    result_df = pd.DataFrame(all_combinations)

    # Apply filters if ranges are provided
    if baureihe_usage_filter and 'Baureihe-Wide Usage (%)' in result_df.columns:
        min_val, max_val = baureihe_usage_filter
        result_df = result_df[
            (result_df['Baureihe-Wide Usage (%)'] >= min_val) &
            (result_df['Baureihe-Wide Usage (%)'] <= max_val)
            ]

    if menge_frequency_filter and 'Probability of Menge (%)' in result_df.columns:
        min_val, max_val = menge_frequency_filter
        result_df = result_df[
            (result_df['Probability of Menge (%)'] >= min_val) &
            (result_df['Probability of Menge (%)'] <= max_val)
            ]

    if result_df.empty:
        logging.info(f"No component usage data found for Baureihe {baureihe_key} after filtering.")
        return pd.DataFrame()

    if not result_df.empty:
        # Create a temporary column for Menge * (Probability / 100)
        # Ensure the columns are numeric before this operation. They should be from earlier rounding.
        result_df['temp_Menge_x_Prob_Decimal'] = result_df['Produktionskoeffizient (Menge)'] * \
                                                 (result_df['Probability of Menge (%)'] / 100.0)

        # Group by 'Komponente' to sum the weighted Menges and the probabilities (as decimals)
        komponente_weighted_avg_data = result_df.groupby('Komponente', as_index=False).agg(
            Sum_Menge_x_Prob_Decimal=('temp_Menge_x_Prob_Decimal', 'sum'),
            Sum_Prob_Decimal=('Probability of Menge (%)', lambda x: (x / 100.0).sum())
        )

        # Calculate the weighted average Menge for each Komponente
        new_col_name = 'Gew. Mittel Produktionskoeff. (Komponente)'
        komponente_weighted_avg_data[new_col_name] = 0.0 # Initialize

        # Avoid division by zero
        non_zero_prob_mask = komponente_weighted_avg_data['Sum_Prob_Decimal'] != 0
        komponente_weighted_avg_data.loc[non_zero_prob_mask, new_col_name] = \
            komponente_weighted_avg_data.loc[non_zero_prob_mask, 'Sum_Menge_x_Prob_Decimal'] / \
            komponente_weighted_avg_data.loc[non_zero_prob_mask, 'Sum_Prob_Decimal']

        # Round the new column
        komponente_weighted_avg_data[new_col_name] = komponente_weighted_avg_data[new_col_name].round(4)

        # Merge the weighted average back into the main DataFrame
        result_df = pd.merge(
            result_df,
            komponente_weighted_avg_data[['Komponente', new_col_name]],
            on='Komponente',
            how='left'
        )

        # Drop the temporary column
        result_df = result_df.drop(columns=['temp_Menge_x_Prob_Decimal'])

    sort_columns = [
        "Produkt", "Komponente", "Baureihe-Wide Usage (%)",
        "Probability of Menge (%)", "Produktionskoeffizient (Menge)"
    ]
    # Ensure all sort columns exist before trying to sort
    sort_columns = [col for col in sort_columns if col in result_df.columns]

    if sort_columns: # only sort if there are columns to sort by
        result_df = result_df.sort_values(
            by=sort_columns,
            ascending=[True, True, False, False, True]
        ).reset_index(drop=True)
    elif not result_df.empty: # if no sort columns but df not empty, reset index
        result_df = result_df.reset_index(drop=True)


    if 'Baureihe-Wide Usage (%)' in result_df.columns:
        result_df["Komponente ist Bestandteil des Produktes in (%)"] = result_df["Baureihe-Wide Usage (%)"].apply(
            lambda x: f"{x:.0f}%")
    if 'Probability of Menge (%)' in result_df.columns:
        result_df["Anteil, bei dem die Komponente mit dem Koeffizient für das Produkt verwendet wird (%)"] = result_df[
            "Probability of Menge (%)"].apply(lambda x: f"{x:.2f}% ")

    result_df = result_df.rename(columns={
        "Produkt": outside_translate("series.col_produkt"),
        "Komponente": outside_translate("series.col_komponente"),
        "Baureihe-Wide Usage (%)": outside_translate("series.col_baureihe_wide_usage_numeric"),
        "Produktionskoeffizient (Menge)": outside_translate("series.col_prod_koeff_numeric"),
        "Probability of Menge (%)": outside_translate("series.col_prob_menge_numeric"),
        "Gew. Mittel Produktionskoeff. (Komponente)": outside_translate("series.col_weighted_avg_prod_koeff_numeric"),
        "Komponente ist Bestandteil des Produktes in (%)": outside_translate("series.display_col_baureihe_wide_usage"),
        "Anteil, bei dem die Komponente mit dem Koeffizient für das Produkt verwendet wird (%)": outside_translate("series.display_col_prob_menge")
    })
    return result_df
# translation Functions
def translate (lang: dict[str, str], key: str) -> str:
    """
    Translate a key to the specified language.

    :param language: The language code ('en' or 'de').
    :param key: The key to translate.
    :return: The translated string.
    """
    if isinstance(lang, dict):
        lang = lang.get('language', 'en')  # default to 'en' if missing
    return TRANSLATIONS.get(lang, {}).get(key, key)

def outside_translate(key: str) -> str:
    return translate(_language, key)

def change_language(new_language):
    global _language
    _language = new_language

def translate_column_explanations() :
    month = outside_translate("article.month")
    stock_level_start = outside_translate("simulation.stock_level_start")
    arriving_stock_quantity = outside_translate("simulation.arriving_stock_quantity")
    stock_after_arrival = outside_translate("simulation.stock_after_arrival")
    expected_demand_quantity = outside_translate("simulation.expected_demand_quantity")
    stock_after_demand_consumption = outside_translate("simulation.stock_after_demand_consumption")
    shortfall = outside_translate("simulation.shortfall")
    order_trigger_level = outside_translate("simulation.order_trigger_level")
    inventory_level_end = outside_translate("simulation.inventory_level_end")
    order_decision = outside_translate("simulation.order_decision")
    order_quantity = outside_translate("simulation.order_quantity")
    expected_stock_arrival = outside_translate("simulation.expected_stock_arrival")
    stock_on_order_end = outside_translate("simulation.stock_on_order_end")
    reasoning_for_order_decision = outside_translate("simulation.reasoning_for_order_decision")
    explanations = {
        month: outside_translate("functions.month"),
        stock_level_start: outside_translate("functions.stock_level_start"),
        arriving_stock_quantity: outside_translate("functions.arriving_stock_quantity"),
        stock_after_arrival: outside_translate("functions.stock_after_arrival"),
        expected_demand_quantity: outside_translate("functions.expected_demand_quantity"),
        stock_after_demand_consumption: outside_translate("functions.stock_after_demand_consumption"),
        shortfall: outside_translate("functions.shortfall"),
        order_trigger_level: outside_translate("functions.order_trigger_level"),
        inventory_level_end: outside_translate("functions.inventory_level_end"),
        order_decision: outside_translate("functions.order_decision"),
        order_quantity: outside_translate("functions.order_quantity"),
        expected_stock_arrival: outside_translate("functions.expected_stock_arrival"),
        stock_on_order_end: outside_translate("functions.stock_on_order_end"),
        reasoning_for_order_decision: outside_translate("functions.reasoning_for_order_decision")
    }
    inv_levl = outside_translate("functions.inventory_level")
    ord_lvl = outside_translate("functions.order_level")
    needed = outside_translate("functions.needed")
    ordered = outside_translate("functions.ordered")
    global COLUMN_EXPLANATIONS, INVENTORY_LEVEL_COLUMN, REORDER_POINT_COLUMN, NEEDED, ORDERED
    COLUMN_EXPLANATIONS = explanations
    INVENTORY_LEVEL_COLUMN = inv_levl
    REORDER_POINT_COLUMN = ord_lvl
    NEEDED = needed
    ORDERED = ordered

def get_origin(item_code: str) -> str:
    """
    Finds the origin region for an item, prioritizing 'Ursprungsland/-region'.
    Args:
        item_code: The item code ('Nr.') to look up.

    Returns:
        The origin country code as a string.
        Returns a message if the item code is not found or if both origin fields are null.
    """
    # Filter the DataFrame to the row with the matching item code
    dataframe = pd.read_csv(ARTIKEL_HERKUNFTSDATEN)
    item_data = dataframe[dataframe['Nr.'] == item_code]

    # Check if the item code exists in the DataFrame
    if item_data.empty:
        return f"Item {item_code} not found."

    # Extract the 'Ursprungsland/-region' value
    # .iloc[0] is used to select the first (and only) row from the filtered data
    origin_country = item_data['Ursprungsland/-region'].iloc[0]

    # Check if the origin_country is null (NaN)
    if pd.isna(origin_country):
        # If it is null, get the 'Herkunftsland/-region' instead
        origin_country = item_data['Herkunftsland/-region'].iloc[0]

    # Final check in case both fields are null
    if pd.isna(origin_country):
        return f"No origin data available for item {item_code}."

    return origin_country

# Helper function to generate recommendation and tradeoff text
def generate_recommendation_text(scenario_name: str,
                                 selected_scenario_metrics: Dict[str, Union[float, int]],
                                 all_scenario_metrics_df: pd.DataFrame,
                                 total_shortfall: float,
                                 origin_country: str,
                                 inventory_info,
                                 total_ordered) -> List[Union[html.P, html.Div, dcc.Markdown]]:
    """
    Generates the recommendation and tradeoff text for the selected scenario.
    """
    recommendations = []

    # Get metrics for the selected scenario
    avg_inventory = selected_scenario_metrics.get("Durchschnittlicher Lagerbestand", 0)
    est_shipments = selected_scenario_metrics.get("Geschätzte Lieferungen", 0)
    combine = False
    if est_shipments > 1:
        combine = True

    min_dem = outside_translate("simulation.min_demand")
    normal_dem = outside_translate("simulation.normal_demand")
    max_dem = outside_translate("simulation.max_demand")
    planned_dem = outside_translate("simulation.planned_demand")
    avg_dem = outside_translate("simulation.avg_calc_demand")
    title = ""

    if scenario_name == "Minimum Demand":
        title = min_dem
    elif scenario_name == "Normal Demand":
        title = normal_dem
    elif scenario_name == "Maximum Demand":
        title = max_dem
    elif scenario_name == "Planned Demand":
        title = planned_dem
    else:
        title = avg_dem
    # Basic Recommendation Statement
    recommendations.append(dcc.Markdown(outside_translate("functions.replenishment_recommendation_intro").format(scenario_name=title)))

    # Core Tradeoffs
    recommendations.append(dcc.Markdown(id={"type": "i18n", "key": "functions.understanding_scenario_tradeoffs"}, className="mt-3"))

    # Service Level / Stock-out Risk vs. Inventory Holding Costs & Green Impact
    if "Minimum Demand" in scenario_name:
        recommendations.append(html.Div([
            dcc.Markdown(id={"type": "i18n", "key": "functions.pros_lower_inventory"}),
            dcc.Markdown(f"""
                - {outside_translate("functions.reduced_holding_costs").format(avg_inventory=f"{avg_inventory:,.0f}")}
                - {outside_translate("functions.lower_environmental_footprint")}
                - {outside_translate("functions.optimized_shipments").format(est_shipments=est_shipments)}
            """),
            dcc.Markdown(id={"type": "i18n", "key": "functions.cons_higher_stockout_risk"}),
            dcc.Markdown(f"""
                - {outside_translate("functions.increased_stockout_risk")}
                - {outside_translate("functions.risk_of_rush_orders")}
            """)
        ]))
    elif "Maximum Demand" in scenario_name:
        recommendations.append(html.Div([
            dcc.Markdown(id={"type": "i18n", "key": "functions.pros_higher_service_level"}),
            dcc.Markdown(f"""
                - {outside_translate("functions.lower_stockout_risk")}
                - {outside_translate("functions.avoidance_rush_orders")}
            """),
            dcc.Markdown(id={"type": "i18n", "key": "functions.cons_higher_inventory"}),
            dcc.Markdown(f"""
                - {outside_translate("functions.higher_inventory_holding_costs").format(avg_inventory=f"{avg_inventory:,.0f}")}
                - {outside_translate("functions.increased_environmental_footprint")}
                - {outside_translate("functions.higher_estimated_shipments").format(est_shipments=est_shipments)}
            """)
        ]))
    elif "Normal Demand" in scenario_name or "Average Calculated Demand" in scenario_name:
        recommendations.append(html.Div([
            dcc.Markdown(id={"type": "i18n", "key": "functions.balanced_approach_heading"}),
            dcc.Markdown(f"""
                - {outside_translate("functions.optimized_balance")}
                - {outside_translate("functions.efficient_resource_use").format(est_shipments=est_shipments, avg_inventory=f"{avg_inventory:,.0f}")}
            """),
            dcc.Markdown(id={"type": "i18n", "key": "functions.considerations"}),
            dcc.Markdown(f"""
                - {outside_translate("functions.balanced_risk_warning")}
                - {outside_translate("functions.wise_considerations_note")}
            """)
        ]))
    elif "Planned Demand" in scenario_name:
        recommendations.append(html.Div([
            dcc.Markdown(id={"type": "i18n", "key": "functions.planned_heading"}),
            dcc.Markdown(f"""
                - {outside_translate("functions.manual_planned_demand_note")}
                - {outside_translate("functions.plan_evaluation_prompt").format(avg_inventory=f"{avg_inventory:,.0f}", est_shipments=est_shipments)}
                - {outside_translate("functions.comparison_table_note")}
            """)
        ]))
    if combine:
        recommendations.append(
            dcc.Markdown(id={"type": "i18n", "key": "functions.green_sustainability_insights_and_notes"}, className="mt-3"))
        recommendations.append(dcc.Markdown(f"""
                - {outside_translate("functions.waste_reduction")}
                - {outside_translate("functions.storage_energy")}
                - {outside_translate("functions.transportation_emissions").format(origin_country=origin_country, max_order_qty=inventory_info.max_order_qty, min_order_qty=inventory_info.min_order_qty, est_shipments=est_shipments)}
                - {outside_translate("functions.shortfall_impact").format(total_shortfall=f"{total_shortfall:,.0f}")}
            """))
    else:
        # Global Green/Sustainability Insights (Applicable to all scenarios but contextualized)
        recommendations.append(dcc.Markdown(id={"type": "i18n", "key": "functions.green_sustainability_insights_and_notes"}, className="mt-3"))
        recommendations.append(dcc.Markdown(f"""
            - {outside_translate("functions.waste_reduction")}
            - {outside_translate("functions.storage_energy")}
            - {outside_translate("functions.transportation_emissions_global").format(origin_country=origin_country, max_order_qty=inventory_info.max_order_qty, min_order_qty=inventory_info.min_order_qty)}
            - {outside_translate("functions.shortfall_impact").format(total_shortfall=f"{total_shortfall:,.0f}")}
        """))


    return recommendations



# --- Ecological Functions ---

def optimize_replenishment_ecological(
        components: Dict[str, dict],
        suppliers: Dict[str, dict],
        demand_scenarios: Dict[str, Dict[str, List[float]]],
        scenario_probs: Dict[str, float],
        periods: int,
        w_eco: float = 1.0,
        shortage_cost: float = 1000
) -> Dict[str, Dict[Union[Tuple[str, int], Tuple[str, int, str]], float]]:
    """
    Optimize replenishment planning under multiple demand scenarios considering ecological impact.

    Args:
        components: Mapping of component ID to component attributes.
        suppliers: Mapping of supplier ID to supplier attributes.
        demand_scenarios: Dictionary of demand scenarios.
                          Format: {scenario_name: {component_id: [demand_t0, demand_t1, ...]}}
        scenario_probs: Probability of each scenario occurring.
        periods: Number of time periods in the planning horizon.
        w_eco: Weight factor for ecological costs in objective function.
        shortage_cost: Penalty cost per unit of shortage.

    Returns:
        A dictionary with order quantities, flags, inventories, and shortages.
    """
    prob = pulp.LpProblem("ReplenishmentWithEcology", pulp.LpMinimize)
    I, q, y, z = {}, {}, {}, {}

    # Define variables
    for i in components:
        for t in range(periods):
            q[i, t] = pulp.LpVariable(f"q_{i}_{t}", lowBound=0)
            y[i, t] = pulp.LpVariable(f"y_{i}_{t}", cat="Binary")
            for omega in demand_scenarios:
                I[i, t, omega] = pulp.LpVariable(f"I_{i}_{t}_{omega}", lowBound=0)
                z[i, t, omega] = pulp.LpVariable(f"z_{i}_{t}_{omega}", lowBound=0)

    # Constraints
    for i, comp in components.items():
        lead = comp["lead_time"]
        for omega in demand_scenarios:
            for t in range(periods):
                demand = demand_scenarios[omega][i][t]
                prev_I = comp["init_inventory"] if t == 0 else I[i, t - 1, omega]
                q_lead = q[i, t - lead] if t - lead >= 0 else 0

                # Inventory balance equation
                prob += (
                        I[i, t, omega] == prev_I + q_lead - demand - z[i, t, omega]
                ), f"InventoryBalance_{i}_{t}_{omega}"

                # Minimum safety stock
                prob += I[i, t, omega] >= comp["safety_stock"], f"SafetyStock_{i}_{t}_{omega}"

        for t in range(periods):
            # Order quantity constraints
            prob += q[i, t] >= comp["min_order"] * y[i, t], f"MinOrder_{i}_{t}"
            prob += q[i, t] <= comp["max_order"] * y[i, t], f"MaxOrder_{i}_{t}"

    # Objective function: cost + weighted ecological impact + shortage penalty
    obj = 0
    for omega in demand_scenarios:
        pi = scenario_probs[omega]
        for i, comp in components.items():
            sup = suppliers[comp["supplier"]]
            for t in range(periods):
                obj += pi * (
                        sup["transport_cost"] * comp["distance"] * q[i, t]
                        + comp["holding_cost"] * I[i, t, omega]
                        + sup["order_cost"] * y[i, t]
                        + w_eco * (
                                sup["transport_emission"] * comp["distance"] * q[i, t]
                                + comp["storage_emission"] * I[i, t, omega]
                        )
                        + shortage_cost * z[i, t, omega]
                )
    prob += obj

    # Solve the problem
    prob.solve()

    # Extract and return solution
    return {
        "order_qty": {(i, t): q[i, t].varValue for i in components for t in range(periods)},
        "order_flag": {(i, t): y[i, t].varValue for i in components for t in range(periods)},
        "inventory": {(i, t, omega): I[i, t, omega].varValue for i in components for t in range(periods) for omega in
                      demand_scenarios},
        "shortage": {(i, t, omega): z[i, t, omega].varValue for i in components for t in range(periods) for omega in
                     demand_scenarios},
    }


def extract_lead_time(value: Union[str, int, float, None]) -> int:
    """
    Extract the lead time (in periods) from a given raw value.

    Args:
        value: A string like '5 WD' or number representing days.

    Returns:
        Estimated lead time in periods.
    """
    if pd.isna(value):
        return 1
    if isinstance(value, (int, float)):
        return int(value)

    match = re.search(r'\d+', str(value))
    if match:
        days = int(match.group())
        if 'WD' in value or 'Werktag' in value:
            return max(1, round(days / 5))  # Assuming 5 workdays per week
        else:
            return days
    return 1


def generate_monthly_tables(
        comp_id: str,
        start_month: str,
        scenario_plan_dfs: Dict[str, DataFrame],
        components_filtered: Dict[str, dict],
        suppliers: Dict[str, dict]
) -> Dict[str, DataFrame]:
    """
    Generate monthly cost and CO2 emission summary tables for multiple planning scenarios.

    Args:
        comp_id (str): Component identifier.
        start_month (str): Start month in 'YYYY-MM' format.
        scenario_plan_dfs (Dict[str, DataFrame]): A dictionary mapping scenario names
            ('conservative', 'normal', 'optimistic') to their corresponding order plan DataFrames.
        components_filtered (Dict[str, dict]): Dictionary of component details.
        suppliers (Dict[str, dict]): Dictionary of supplier details.

    Returns:
        Dict[str, DataFrame]: A dictionary mapping scenario names to monthly KPI summary DataFrames.
    """
    # Determine number of months to generate
    sample_df = next(iter(scenario_plan_dfs.values()))
    n_periods = len(sample_df)

    # Generate list of month labels
    month_labels = pd.date_range(start=start_month, periods=n_periods, freq="MS").strftime("%Y-%m")

    # Lookup component and supplier information
    comp = components_filtered[comp_id]
    sup = suppliers[comp["supplier"]]

    monthly_tables: Dict[str, DataFrame] = {}

    # Loop through each planning scenario
    for scenario, df in scenario_plan_dfs.items():
        df = df.copy()

        # Set index to 'Month Start' if needed
        if 'Month Start' in df.columns:
            df = df.set_index('Month Start')
        elif df.index.name != 'Month Start':
            df.index.name = 'Month Start'

        # Assign standardized month labels as index
        df.index = month_labels
        df.index.name = "Month"

        # Compute monthly KPIs
        monthly_df = pd.DataFrame({
            "Order Quantity": df["Order Quantity"],
            "Inventory": df["Stock Level After Arrival of Stocks"],
            "Shortage": df["Shortfall"],
            "Transport Cost": df["Order Quantity"] * comp["distance"] * sup["transport_cost"],
            "Holding Cost": df["Stock Level After Arrival of Stocks"] * comp["holding_cost"],
            "Order Cost": (df["Order Quantity"] > 0).astype(int) * sup["order_cost"],
            "Transport CO2": df["Order Quantity"] * comp["distance"] * sup["transport_emission"],
            "Storage CO2": df["Stock Level After Arrival of Stocks"] * comp["storage_emission"],
        }, index=month_labels)

        monthly_tables[scenario] = monthly_df

    return monthly_tables


def summarize_scenario_metrics(
        comp_id: str,
        scenario_plan_dfs: Dict[str, pd.DataFrame],
        components_filtered: Dict[str, dict],
        suppliers: Dict[str, dict]
):
    """
    Summarizes key metrics (cost, CO2, inventory, etc.) across planning scenarios for a given component.

    Args:
        comp_id: The component ID to analyze.
        scenario_plan_dfs: Mapping of scenario names to their corresponding DataFrames.
        components_filtered: Component metadata dictionary.
        suppliers: Supplier metadata dictionary.

    Returns:
        pd.DataFrame: A table summarizing key metrics for each scenario.
        comp: Component metadata dictionary.
        sup: Supplier metadata dictionary.
    """
    comp = components_filtered[comp_id]
    sup = suppliers[comp["supplier"]]

    rows = []

    for scenario, df in scenario_plan_dfs.items():
        oq = df["Order Quantity"].sum()
        inv = df["Stock Level After Arrival of Stocks"].sum()
        sh = df["Shortfall"].sum()
        hc = (df["Stock Level After Arrival of Stocks"] * comp["holding_cost"]).sum()
        sco2 = (df["Stock Level After Arrival of Stocks"] * comp["storage_emission"]).sum()
        tc = (df["Order Quantity"] * comp["distance"] * sup["transport_cost"]).sum()
        tco2 = (df["Order Quantity"] * comp["distance"] * sup["transport_emission"]).sum()
        oc = (df["Order Quantity"] > 0).sum() * sup["order_cost"]

        rows.append({
            "Scenario": scenario,
            "Total Order Qty": oq,
            "Total Orders Placed": (df["Order Quantity"] > 0).sum(),
            "Total Inventory": inv,
            "Total Shortage": sh,
            "Total Transport Cost": tc,
            "Total Holding Cost": hc,
            "Total Order Cost": oc,
            "Total Transport CO2": tco2,
            "Total Storage CO2": sco2,
        })

    return pd.DataFrame(rows), comp, sup


def build_components_and_suppliers(
        artikel_daten: pd.DataFrame,
        final_stock_level: int,
        safety_stock: int,
        DEFAULT_DISTANCE: int,
        DEFAULT_ORDER_COST: float,
        DEFAULT_TRANSPORT_COST: float,
        DEFAULT_TRANSPORT_EMISSION: float,
        DEFAULT_STORAGE_EMISSION: float,
        DEFAULT_SCRAP_EMISSION: float,
        extract_lead_time_fn
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Build components and suppliers dictionaries from raw article data.

    Args:
        artikel_daten: Input DataFrame with article metadata.
        final_stock_level: Default initial stock level for all components.
        safety_stock: Default safety stock value if missing.
        DEFAULT_DISTANCE: Fallback distance in km.
        DEFAULT_ORDER_COST: Fallback cost per order.
        DEFAULT_TRANSPORT_COST: Fallback transport cost per unit/km.
        DEFAULT_TRANSPORT_EMISSION: Fallback CO2 emissions per unit/km.
        DEFAULT_STORAGE_EMISSION: Emission per unit in storage.
        DEFAULT_SCRAP_EMISSION: Emission per unit scrapped.
        extract_lead_time_fn: Function to extract lead time from raw Beschaffungszeit.

    Returns:
        Tuple of two dictionaries: (components, suppliers)
    """
    components = {}
    suppliers = {}

    country_to_distance = {
        'DE': 100,
        'FR': 800,
        # Add more countries as needed
    }

    for idx, row in artikel_daten.iterrows():
        comp_id = row['Nr.']
        sup_id = row['Kreditorennr.']
        einstandspreis = row['Einstandspreis'] if not pd.isna(row['Einstandspreis']) else 1.0
        holding_cost = einstandspreis * 0.01

        # Estimate distance by country
        if 'Ursprungsland/-region' in row and pd.notna(row['Ursprungsland/-region']):
            country = row['Ursprungsland/-region']
            distance = country_to_distance.get(country, DEFAULT_DISTANCE)
        else:
            distance = DEFAULT_DISTANCE

        components[comp_id] = {
            "supplier": sup_id,
            "distance": distance,
            "lead_time": extract_lead_time_fn(row['Beschaffungszeit']),
            "init_inventory": final_stock_level,
            "safety_stock": int(row['Sicherheitsbestand']) if not pd.isna(row['Sicherheitsbestand']) else safety_stock,
            "min_order": int(row['Minimalbestand']) if not pd.isna(row['Minimalbestand']) else 1,
            "max_order": int(row['Maximalbestand']) if not pd.isna(row['Maximalbestand']) else 99999,
            "holding_cost": holding_cost,
            "storage_emission": DEFAULT_STORAGE_EMISSION,
            "scrap_emission": DEFAULT_SCRAP_EMISSION,
        }

        if sup_id not in suppliers:
            suppliers[sup_id] = {
                "order_cost": DEFAULT_ORDER_COST,
                "transport_cost": DEFAULT_TRANSPORT_COST,
                "transport_emission": DEFAULT_TRANSPORT_EMISSION,
            }

    return components, suppliers


def prepare_demand_scenarios_from_predictions(
        results_df: pd.DataFrame,
        comp_id: str,
        date_col: str = None,
        scenario_probs: Dict[str, float] = {"conservative": 0.3, "normal": 0.5, "optimistic": 0.2}
):
    """
    Aggregates forecast results into monthly demand scenarios.

    Args:
        results_df: DataFrame with index or column as dates and 'Predicted', 'Lower_Bound', 'Upper_Bound' settings.
        comp_id: Component ID for which to build scenarios.
        date_col: Optional column name if date is not in the index.
        scenario_probs: Probabilities assigned to each scenario (must include conservative, normal, optimistic).

    Returns:
        - demand_scenarios: Dict mapping scenario names to {comp_id: list of monthly demand}.
        - periods: Number of months aggregated.
        - months: List of month labels as strings ("YYYY-MM").
        - scenario_probs: Probabilities assigned to each scenario (must include conservative, normal, optimistic).
    """
    df = results_df.copy()

    # Ensure datetime index or column
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
    else:
        df.index = pd.to_datetime(df.index)

    # Generate month labels
    df['Month'] = df.index.to_series().dt.to_period('M').astype(str)

    # Aggregate monthly forecasts
    monthly_agg = df.groupby('Month').agg({
        'Predicted': 'sum',
        'Lower_Bound': 'sum',
        'Upper_Bound': 'sum'
    }).reset_index()

    # Extract values
    predicted_demand = monthly_agg['Predicted'].tolist()
    lower_demand = monthly_agg['Lower_Bound'].tolist()
    upper_demand = monthly_agg['Upper_Bound'].tolist()
    months = monthly_agg['Month'].tolist()
    periods = len(months)

    # Build demand scenarios
    demand_scenarios = {
        "conservative": {comp_id: lower_demand},
        "normal": {comp_id: predicted_demand},
        "optimistic": {comp_id: upper_demand}
    }

    return demand_scenarios, periods, months, scenario_probs
