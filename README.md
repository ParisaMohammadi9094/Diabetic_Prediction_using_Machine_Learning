# Predicting the Success of Startups

This project aims to predict the success of startups based on various financial and administrative features using machine learning techniques.

## Project Overview

In this project, we use a dataset containing information about various startups, including their spending on R&D, Administration, and Marketing, as well as their state location and profit. The goal is to build a machine learning model that can predict the profit of a startup based on these features.

## Dataset

The dataset used in this project contains the following features:
- **R&D Spend**: Amount of money spent on research and development
- **Administration**: Amount of money spent on administration
- **Marketing Spend**: Amount of money spent on marketing
- **State**: The state in which the startup is located
- **Profit**: The profit earned by the startup (target variable)

The dataset can be found in the `data/` directory.

## Project Structure

- `data/`: Contains the dataset file(s).
  - `startup_data.csv`: The raw dataset
  - `preprocessed_startup_data.csv`: The preprocessed dataset
- `notebooks/`: Jupyter notebooks for data preprocessing and model training.
  - `data_preprocessing.ipynb`: Notebook for data preprocessing
  - `model_training.ipynb`: Notebook for model training and evaluation
- `src/`: Source code for data preprocessing and model training.
  - `data_preprocessing.py`: Script for data preprocessing
  - `model_training.py`: Script for model training and evaluation
- `README.md`: Project overview and instructions
- `requirements.txt`: List of required Python packages
- `LICENSE`: License information

## Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
