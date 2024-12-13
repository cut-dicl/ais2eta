# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:12:22 2022

@author: Nicos Evmides & Brian
"""
import pandas as pd
import sqlalchemy
from pathlib import Path


def load_dataset(get_invalid_eta_as_null):
    #table_nm = 'my_dataset'
    #if get_invalid_eta_as_null == True:
    #    table_nm = 'my_xgb_dataset'
    #engine = sqlalchemy.create_engine(
    #    'mysql+pymysql://aisdb:aisdbpass1@localhost/aisdb')
    #dataset = pd.read_sql_table(table_nm, engine)

    curr_dir = Path(__file__).parents[2].absolute()
    input_files_dir = rf"{curr_dir}/dataset generation python code"
    dataset = pd.read_csv(rf"{input_files_dir}/all_data_final.csv")
    #dataset.received_timestamp = pd.to_datetime(dataset.received_timestamp)
    #print (dataset)
    print(len(dataset))
    return dataset


def get_eqvlnt_dict_vals_for_destination():
    ship_categ = {
        'BULK CARRIER': 0,
        'CONTAINER': 1,
        'GENERAL CARGO': 2,
        'TANKER': 3,
        'OFFSHORE SUPPORT': 4,
        'PASSENGER': 5,
        'OTHER': 6
    }
    dest = {'CYLMS': 0, 'CYLCA': 1, 'CYZYY': 2, 'CYMOI':4}
    return ship_categ, dest


def get_cols_to_drop(paper):

    # All CSV columns
    #received_timestamp,
    #mmsi,
    #longitude,
    #latitude,
    #sog,
    #cog,
    #rot,
    #nav_status,
    #true_heading,
    #destination,
    #draught,
    #eta_total,
    #breadth,
    #length,
    #vesselType,
    #ShipCategory,
    #arrival_time_exists,
    #arrival_time,
    #actual_eta_in_min,
    #agent_eta_in_min,
    #distance_to_cover

    if paper == "evmides":
    # ************************ OUR
    # Please note the bolow columns are the ones to be excluded
    # Our paper

        cols = [
            'vesselType', 'received_timestamp', 'mmsi', 'arrival_time_exists',
            'arrival_time', 'eta_total'
            ]

    elif paper == "kolley":
    # ************************ PAPER Kolley et al (P1)
    # Please note the bolow columns are the ones to be excluded
    # NN Paper
    #MMSI
    #LAT
    #LON
    #SOG
    #COG
    #Heading
    #Vessel type
    #Status
        cols = [
            'received_timestamp', 'rot', 'destination', 'draught', 'eta_total', 'breadth',
            'length', 'arrival_time_exists', 'arrival_time',
            'distance_to_cover', 'vesselType'
            ]

    elif paper == "flapper":
    # ************************ PAPER Flapper et al (P2)
    # Please note the bolow columns are the ones to be excluded
        cols = [
            'received_timestamp', 'mmsi', 'arrival_time_exists',
            'arrival_time', 'eta_total', 'sog', 'cog', 'true_heading', 'rot', 'destination', 'draught'
            ]

    elif paper == "parolas":
    # ************************ PAPER Parolas et al (P3)
    # Please note the bolow columns are the ones to be excluded

        cols = [
            'vesselType', 'received_timestamp', 'mmsi', 'arrival_time_exists',
            'arrival_time', 'eta_total', 'sog','cog', 'true_heading', 'rot', 'destination', 'draught'
            ]

    elif paper == "hajbabaie":
    # ************************ PAPER Hajbabaie et al (P4)
    # Please note the bolow columns are the ones to be excluded

        cols = [
            'vesselType', 'received_timestamp', 'mmsi', 'arrival_time_exists',
            'arrival_time', 'eta_total', 'rot', 'destination', 'draught'
            ]

    #if len(args) > 0:
    #    cols.extend(args)
    return cols


def load_and_prepare_dataset(get_invalid_eta_as_null, eta_in_hours, paper):
    try:
        # Step 1: Load dataset
        print("Loading dataset...")
        dataset = load_dataset(get_invalid_eta_as_null)
        if dataset is None or dataset.empty:
            raise ValueError("Dataset is empty or not loaded correctly.")
        print("Dataset loaded successfully.")

        # Step 2: Validate paper argument
        if not paper:
            raise ValueError("The 'paper' argument is missing or None.")
        print(f"Processing for paper: {paper}...")

        # Step 3: Drop columns based on the paper specified
        print(f"Dropping columns for paper: {paper}...")
        cols_to_drop = get_cols_to_drop(paper)
        if not set(cols_to_drop).issubset(dataset.columns):
            missing_cols = set(cols_to_drop) - set(dataset.columns)
            raise KeyError(f"Columns to drop not found in dataset: {missing_cols}")
        dataset.drop(columns=cols_to_drop, inplace=True)

        # Step 4: Replace ship categories and destinations with equivalent values
        print("Replacing ship category and destination values...")
        ship_categ_dict, dest_dict = get_eqvlnt_dict_vals_for_destination()
        dataset = dataset.replace({
            "ShipCategory": ship_categ_dict,
            "destination": dest_dict
        })

        # Step 5: Remove duplicates
        print("Removing duplicates from dataset...")
        dataset.drop_duplicates(inplace=True, ignore_index=True)
        print(f"Dataset after removing duplicates: {len(dataset)} rows remaining.")

        # Step 6: Prepare features (X) and target (y)
        print("Preparing features (X) and target (y)...")
        if 'actual_eta_in_min' not in dataset.columns:
            raise KeyError("'actual_eta_in_min' column not found in dataset.")
        x = dataset.drop(columns=['actual_eta_in_min'])
        y = dataset['actual_eta_in_min']

        # Step 7: Extract k-fold values
        print("Extracting k-fold values...")
        if 'k_fold' not in dataset.columns:
            raise KeyError("'k_fold' column not found in dataset.")
        k_folds = dataset['k_fold']

        print("Dataset prepared successfully.")
        return x, y, k_folds

    except KeyError as ke:
        print(f"KeyError: {ke}")
        raise
    except ValueError as ve:
        print(f"ValueError: {ve}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise



def load_and_prepare_dataset_old(get_invalid_eta_as_null, eta_in_hours, paper):
    dataset = load_dataset(get_invalid_eta_as_null)
    print (dataset)
    dataset.drop(columns=get_cols_to_drop(paper=paper), inplace=True)
    ship_categ_dict, dest_dict = get_eqvlnt_dict_vals_for_destination()
    dataset = dataset.replace({
        "ShipCategory": ship_categ_dict,
        "destination": dest_dict
    })
    dataset.drop_duplicates(inplace=True,ignore_index=True)
    x = dataset.drop(columns=['actual_eta_in_min'])
    y = dataset['actual_eta_in_min']
    k_folds = dataset['k_fold']
    return x, y , k_folds
