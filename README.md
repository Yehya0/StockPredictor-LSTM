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

# Usage

Clone the repository:

    git clone https://github.com/yourusername/StockPredictor-LSTM.git
    cd StockPredictor-LSTM

Navigate to the notebooks directory:

    cd notebooks

Open the Jupyter notebook:

    jupyter notebook StockPredictor.ipynb

Run the notebook:

    Load the data: The notebook includes code to download historical stock price data from Yahoo Finance.
    Preprocess the data: The notebook preprocesses the data by calculating technical indicators (RSI, MACD), scaling the data, and preparing it for the LSTM model.
    Train the model: The notebook trains an LSTM model on the processed data.
    Make predictions: The notebook uses the trained model to predict future stock prices and visualize the results.
Sure, here’s a more professionally formatted section for the "Results of NVIDIA":

---

## Results of NVIDIA

### Historical Close Price
The chart below shows the historical closing prices of NVIDIA stock over the selected period. This data serves as the basis for further analysis and predictions.

![Historical Close Price](https://github.com/Yehya0/StockPredictor-LSTM/assets/89547515/982ac1ee-07e7-4693-8d2e-c6283f92c55c)

### MACD and Signal Line
The MACD (Moving Average Convergence Divergence) and Signal Line are plotted to identify potential buy and sell signals. The intersections of these lines indicate market trends and momentum changes.

![MACD and Signal Line](https://github.com/Yehya0/StockPredictor-LSTM/assets/89547515/bd67c683-e347-4cc7-aca1-437c2fd7cd14)

### Relative Strength Index (RSI)
The RSI chart helps to determine the momentum of NVIDIA's stock price. It identifies overbought or oversold conditions which could indicate potential reversal points.

![Relative Strength Index](https://github.com/Yehya0/StockPredictor-LSTM/assets/89547515/c287d74a-98d3-40bc-be72-7922d43132ff)

### Future Stock Price Prediction
Using the LSTM (Long Short-Term Memory) model, the following chart presents the predicted future stock prices of NVIDIA. This predictive model helps in making informed investment decisions.

![Future Stock Price Prediction](https://github.com/Yehya0/StockPredictor-LSTM/assets/89547515/2093f665-2513-4409-876a-51ca78f6e064)

---

