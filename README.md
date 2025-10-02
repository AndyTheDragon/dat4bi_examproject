# Customer Purchasing Behaviour: *Optimizing sales channels for Arhaan Thai.*

## Annotation

Our project addresses the challenge of understanding how customer purchasing behaviour differs across five sales channels (TakeawayWolt, TakeawayMealo, TakeawayCard, EatInCard and EatInSoftpay).
This challenge is important because optimizing operations, menus and marketing for each channel can significantly increase revenue and improve customer satisfaction.
We will use descriptive analytics and machine learning classification models to uncover key behavioural patterns and predict the sales channel of new orders.
The insights will benefit the restaurant manager by enabling data driven decisions that improve efficiency, boost sales, and enhance the dining experience for customers.

## Problem Statement

*A local restaurant wants to understand the distinct purchasing behaviors of its customers across its five sales channels (TakeawayWolt, TakeawayMealo, TakeawayCard, EatInCard, and EatInSoftpay). By identifying the key factors that differentiate these channels, the restaurant can optimize its marketing, menu offerings, and operational strategies to increase sales and customer satisfaction.*

## Context and Purpose

Arhaan Thai is a local restaurant that serves customers through five distinct sales channels: *TakeawayWolt, TakeawayMealo, TakeawayCard, EatInCard, and EatInSoftpay*.
 The restaurant collects historical order data, including timestamp, item count and channel, but has not yet systematically analyzed these data to understand customer behaviour. Gaining a deeper understanding of how purchases vary across channels can help the restaurant make informed decisions about marketing, menu offerings and operations.

The purpose is to analyze and predict customer behaviour per channel so that the restaurant can:

- Optimize operations
- Adjust menu offerings and marketing
- Increase revenue and customer satisfaction

## Research Questions

- *How do order characteristics differ across the five sales channels?*
- *Can we predict the sales channel of a new order with at least 80% accuracy?*
- *Which features are most important for differentiating sales channels, and how can these insights guide operations and marketing?*

## Hypotheses

- The average spend pr customer is higher for in-house customers as the probability for them ordering drinks is higher than people eating takeaway.
- The more main dishes ordered in house, the more beverages are ordered.

## Proposed Solution

We will build a descriptive analytics and machine learning project to achieve this goal. We will first perform an exploratory data analysis (EDA) to describe the differences in order characteristics across channels. We will then develop a predictive classification model to determine which sales channel an order belongs to based on its features. The model's insights, derived from feature importance and model coefficients, will serve as our primary source of understanding customer behavior.

## Data and Scope

The project will use historical order data. We will aggregate and engineer features from the order lines to form a dataset with the following columns:

- order_id (unique identifier)
- day_of_week
- order_total
- sales_channel (our target variable)
- number_of_maindishes
- number_of_snacks
- number_of_drinks
- number_of_soups
- number_of_extras
- is_takeaway (a binary variable)

### From Raw Data to Dataframe

In the data_engineering.ipynb we take our raw data and transforms it into the above columns. Orderlines have been aggregated into one row that represents the whole order, where before an order would be on several rows.
Text values have been encoded into numeric values as described in the tables below.

#### Understanding the Data

The following columns have been encoded into numeric values as displayed below;

#### day_of_week

| #  | Text       |
|----|------------|
| 1  | Monday     |
| 2  | Tuesday    |
| 3  | Wednesday  |
| 4  | Thursday   |
| 5  | Friday     |
| 6  | Saturday   |
| 7  | Sunday     |

#### payment_method

| #  | Text       |
|----|------------|
| 1  | Mealo      |
| 2  | Wolt       |
| 3  | In-House   |

#### is_takeaway

| #  | Text       |
|----|------------|
| 1  | Takeaway   |
| 2  | In-House   |

## Project Planning

- **Sprint Timeline**
  - Sprint 1: Problem Formulation
  - Sprint 2: Data Preparation
  - Sprint 3: Data Modelling
  - Sprint 4: Business Application
- **Tools and technologies**
  - StreamLit Application
  - Anaconda 3 Python environment and libraries
  - Jupyter Notebook

## Repository & Developtment setup

- Git repository: [https://github.com/AndyTheDragon/dat4bi_examproject](https://github.com/AndyTheDragon/dat4bi_examproject)
- Structure
  - Four Python packages called *DataExploration*, *ShowClassifcation*, *ShowClustering*, *ShowLinearRegression*
  - *data* folder containing raw and clean data
  - *Notebooks* used for experimenting before inserting into report
  - *Documents* containing files such as the decision tree pdf
  - A *report.ipynb* notebook containing our findings
  - *README.md*
  - *App_\*.py* files pertaining the streamlit application
- Software Requirements
  - Graphviz
  - Streamlit
