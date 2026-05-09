# INST 737 Final Project - Samyug Swarna

## Project Overview

This project analyzes gym demand and gym competition at the ZIP Code Tabulation Area level. The goal is to combine public health indicators with gym locations to identify areas that may have higher need for fitness services and then compare that need with the current number of gym locations.

The project uses a data science pipeline with extraction, transformation, modeling, and visualization stages. It has  health, geographic, and points-of-interest data to create an analytical dataset at the ZCTA level.

## Business Problem

Fitness businesses and planners may want to know where gym demand may be high and where current gym availability may be limited. This project uses obesity and physical inactivity as proxy indicators of fitness-related need, then compares those measures with gym counts from OpenStreetMap data. By joining these datasets at the ZCTA level, the project creates a way to explore areas where demand and competition may differ.

## Datasets Used

### 1. CDC PLACES ZCTA Data
This dataset provides ZIP Code Tabulation Area-level public health indicators. For this project, the main measures used are:

- Obesity
- Physical Inactivity

These variables are used as demand-related health indicators.

### 2. U.S. Census TIGER/Line ZCTA Shapefile
This dataset provides polygon boundaries for ZCTAs. It is used to spatially organize the project and to connect gym point locations to ZCTA areas through a spatial join.

### 3. OpenStreetMap Maryland POI Data from Geofabrik
This dataset provides points of interest and area-based POI features for Maryland. It is used to identify gym and fitness-related locations and count them by ZCTA.


## Project Structure


inst737-final-project-samyug-swarna/
├── data/
│   ├── raw/
│   │   ├── cdc/
│   │   ├── census/
│   │   └── osm/
│   ├── transformed/
│   ├── model_outputs/
│   ├── visualizations/
│   └── reference-tables/
├── etl/
│   ├── __init__.py
│   ├── extract.py
│   ├── transform.py
│   └── load.py
├── analysis/
│   ├── __init__.py
│   └── model_1.py
├── vis/
│   ├── __init__.py
│   └── visualizations.py
├── main.py
├── README.md
├── requirements.txt
└── .gitignore


## Final Project Updates

A stronger model evaluation, logging, and error handling. Now it saves additional evaluation outputs in `data/model_outputs/`, including model metrics, predictions, feature importance, an actual versus predicted plot, and a residuals plot.

Pipeline makes a log file at `logs/pipeline.log`. Added logging statements so that each major stage of the pipeline reports progress more clearly, also because any failures will be easier to see and debug. Additional `try/except` handling was also added so that some stage-specific errors can be logged without immediately breaking the full run.

