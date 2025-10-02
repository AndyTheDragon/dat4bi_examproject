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

- *"How do order characteristics differ across the five sales channels?"*
- *"Can we predict the sales channel of a new order with at least 80% accuracy?"*
- *"Which features are most important for differentiating sales channels, and how can these insights guide operations and marketing?"*

## Hypotheses

- Orders placed through a takeaway platform will have a higher order total compared to in house dining
- The more main dishes ordered in house, the more beverages is ordered.
- The weekdays with the lowest order-totals will have more take-away orders.

## Proposed Solution

We will build a descriptive analytics and machine learning project to achieve this goal. We will first perform an exploratory data analysis (EDA) to describe the differences in order characteristics across channels. We will then develop a predictive classification model to determine which sales channel an order belongs to based on its features. The model's insights, derived from feature importance and model coefficients, will serve as our primary source of understanding customer behavior.

## Data and Scope

The project will use historical order data. We will aggregate and engineer features from the order lines to form a dataset with the following columns:

- order_id (unique identifier)
- sales_channel (our target variable)
- number_of_maindishes
- number_of_snacks
- number_of_drinks
- order_total
- is_takeaway (a binary variable)
- time_of_day
- day_of_week

#### Data

We have a data_engineering.ipynb file where we load the dataset in the format that we receive it. We decide to format it by updating names for product categories, ensuring numeric values and removing any irrelevant information etc. 

#### Report

In our report we have used the formatted dataset which we worked out from data_engineering.ipynb. This is where we begin our actual analysis, exploring and cleaning the data, preparing the data for clustering, boxplots, histograms, decision tree and heatmap.

## Project Planning

- **Sprint Timeline**
  - Sprint 1: Problem Formulation
  - Sprint 2: Data Preparation
  - Sprint 3: Data Modelling
  - Sprint 4: Business Application
- Tools and technologies
- Teamwork setup

## Repository & Developtment setup

Git repository: [https://github.com/AndyTheDragon/dat4bi_examproject](https://github.com/AndyTheDragon/dat4bi_examproject)

- Structure
- Software Requirements
