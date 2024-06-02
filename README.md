# Predicting Stock Prices using LSTM Neural Networks

## Project Description
This project utilizes Long Short-Term Memory (LSTM) neural networks to predict stock prices. It leverages historical stock price data, including technical indicators like RSI and MACD, to forecast future prices. The main goal is to demonstrate the application of deep learning techniques in financial market analysis.

### Key Features:
- Data collection from Yahoo Finance API
- Calculation of technical indicators (RSI, MACD)
- Data preprocessing and normalization
- LSTM model training and evaluation
- Visualization of predictions and performance metrics

### Technologies Used:
- Python
- Pandas
- NumPy
- Yahoo Finance API
- Scikit-learn
- TensorFlow/Keras
- Matplotlib

## Introduction
This project aims to predict stock prices using Long Short-Term Memory (LSTM) neural networks. LSTM networks are a type of recurrent neural network (RNN) capable of learning long-term dependencies, making them suitable for time-series forecasting tasks like stock price prediction.

## Dataset
The dataset comprises historical stock price data from various companies. We obtained the data using the Yahoo Finance API, which includes features such as closing prices, volume, and technical indicators like RSI and MACD.

## Installation
To run this project, you'll need Python and the following libraries:
- pandas
- numpy
- yfinance
- scikit-learn
- matplotlib
- tensorflow

You can install the dependencies using pip:
```bash
pip install pandas numpy yfinance scikit-learn matplotlib tensorflow
```
Usage

    Clone the repository:

    bash

git clone https://github.com/yourusername/StockPredictor-LSTM.git
cd StockPredictor-LSTM

Navigate to the notebooks directory:

bash

cd notebooks

Open the Jupyter notebook:

bash

jupyter notebook StockPredictor.ipynb

Run the notebook:

    Load the data: The notebook includes code to download historical stock price data from Yahoo Finance.
    Preprocess the data: The notebook preprocesses the data by calculating technical indicators (RSI, MACD), scaling the data, and preparing it for the LSTM model.
    Train the model: The notebook trains an LSTM model on the processed data.
    Make predictions: The notebook uses the trained model to predict future stock prices and visualize the results.
