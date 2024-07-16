# For Lottifies
import json 
from streamlit_lottie import st_lottie 

# Import NumPy for numerical operations
import numpy as np

# Import pandas for data manipulation and analysis
import pandas as pd

# Import yfinance for downloading stock market data
import yfinance as yf

import tensorflow as tf
from tensorflow.keras.models import load_model

# Import Streamlit for creating web applications
import streamlit as st

# Import Matplotlib for data visualization
import matplotlib.pyplot as plt

# Import necessary libraries
from datetime import datetime  # Import datetime module for date and time operations
import yfinance as yf  # Import yfinance for downloading stock data


# Now we are going to load our model
model = load_model('Stock_Price_Prediction.h5')

# Yha Heading Likhna Hai

# Navigation Bar
# Define the navigation options
nav_options = {
    "Home": "Home",
    "Analyze Stocks": "Analyze Stocks",
    "Trend Trackers" : "Trend Trackers",
    "Indicator Insights": "Indicator Insights",
    "About": "About"
}

#For Lootie Aimation
def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)
    
# Create a sidebar with a selectbox for navigation
selected_page = st.sidebar.selectbox("Navigation", list(nav_options.keys()))

# Display content based on the selected page
if selected_page == "Home":
    # Title Of The Website
    st.markdown("""
        <style>
            .title {
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Stock Sage</h1>", unsafe_allow_html=True)
    st.markdown("""
### Overview

StockSage is your ultimate companion for navigating the stock market with confidence and clarity. Whether you're an experienced investor or just getting started, StockSage provides powerful tools and insights to help you make informed decisions.

### Features

1. **Stock Analysis**
   - Enter the ticker symbol of any stock to analyze its performance metrics, historical trends, and analyst ratings.

2. **Predictive Analytics**
   - Utilize advanced machine learning models to forecast potential price movements based on thorough historical data analysis.

3. **Technical Indicators**
   - Learn about and interpret key technical indicators such as moving averages (MA/EMA), stochastic oscillators, MACD, Bollinger Bands, RSI, Fibonacci retracement, Ichimoku Cloud, standard deviation, and ADX.

Explore StockSage's features to enhance your investment strategy and stay ahead in the dynamic world of finance!

                """)

  # Add content for about page

elif selected_page == "Analyze Stocks":
    # Define start date for fetching stock data
    start = '2012-01-01'

    # Get today's date and format it as 'YYYY-MM-DD' using strftime
    end = datetime.today().strftime('%Y-%m-%d')

    # Yha input le rhe hai stock name and default GOOG Rahega
    # Title Of The Website
    st.markdown("""
        <style>
            .title {
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Analyze Stocks</h1>", unsafe_allow_html=True)
    stock = st.text_input('Ticker Symbol', 'GOOG')


    # Download stock data using yfinance library from 'start' to 'end' date
    data = yf.download(stock, start, end)

    # 'data' now contains the stock data from 'start' to today's date ('end')


    # Display a subheader for Stock Data using Streamlit
    st.subheader('Stock Data')

    # Display the entire dataset using Streamlit
    st.write(data)

    # Create a DataFrame for training data containing the first 80% of the 'Close' prices
    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])

    # Create a DataFrame for testing data containing the remaining 20% of the 'Close' prices
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])


    # Import MinMaxScaler from sklearn for scaling data
    from sklearn.preprocessing import MinMaxScaler

    # Initialize MinMaxScaler to scale data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0,1))

    # Select the last 100 days of data from data_train for continuity
    pas_100_days = data_train.tail(100)

    # Concatenate the selected 100 days with data_test to ensure sequence continuity
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)

    # Scale the combined data_test using the MinMaxScaler
    data_test_scale = scaler.fit_transform(data_test)
    
    
    # Define a function to display MA50 and MA100 plot
    def display_ma100():
        st.subheader('Price vs MA100')
        
        # Calculate the 100-day moving averages (MA50 and MA100) of the 'Close' prices
        ma_100_days = data.Close.rolling(100).mean()
        
        # Create a new figure for plotting with a specific size
        fig = plt.figure(figsize=(8, 6))
        
        # Plot the MA100 in red, original 'Close' prices in green
        plt.plot(ma_100_days, 'r', label='MA100')
        plt.plot(data.Close, 'g', label='Close Price')
        
        # Label axes
        plt.xlabel('Time')
        plt.ylabel('Price')
        
        # Show the plot using Matplotlib
        plt.show()
        
        # Show legend
        plt.legend()

        
        # Display the Matplotlib plot in Streamlit
        st.pyplot(fig)
        
        # Display introductory Markdown text
    # Define a function to display 50-day EMA plot
    def display_ema_50_days():
        # Calculate the 50-day exponential moving average (EMA)
        ema_50_days = data['Close'].ewm(span=50, adjust=False).mean()

        # Create a new figure for plotting with a specific size
        fig2 = plt.figure(figsize=(8, 6))

        # Plot the 50-day EMA data in red ('r')
        plt.plot(ema_50_days, 'r', label='50-Day EMA')

        # Plot the 'Close' price data in green ('g')
        plt.plot(data.Close, 'g', label='Close Price')

        # Label axes
        plt.xlabel('Time')
        plt.ylabel('Price')

        # Show legend
        plt.legend()

        # Display the plot using Matplotlib
        plt.show()

        # Display the Matplotlib plot in Streamlit
        st.pyplot(fig2)
    
    st.markdown("# Long-Term Stock Analysis")
    st.markdown("Click the button below to analyze long-term trends:")

    # Create a button in Streamlit
    if st.button('Long-Term Analysis'):
        # Call function to display MA50 and MA100 plot
        display_ma100()
        
    
    st.markdown("# Sort-Term Stock Analysis")
    st.markdown("Click the button below to analyze short-term trends:")
    
    # Create a button in Streamlit
    if st.button('Short-Term Analysis'):
        # Call function to display MA50 and MA100 plot
        display_ema_50_days()
        
    st.markdown("""
                ## Market Prediction Based on 100-Day Moving Average

Explore the market prediction using the 100-day moving average:

1. **Understanding the Prediction**:
   - The 100-day moving average (MA100) is a widely used indicator in technical analysis.
   - It helps in smoothing out price data to identify trends and potential reversals.

2. **How It Works**:
   - Click the button below to predict market trends based on the MA100.
                """)


    # Initialize empty lists x and y to store sequences and labels
    x = []
    y = []

    # Iterate through data_test_scale starting from index 100 to the end
    for i in range(100, data_test_scale.shape[0]):
        # Append sequences of 100 consecutive data points to x
        x.append(data_test_scale[i-100:i])
        # Append the corresponding label (next data point) to y
        y.append(data_test_scale[i, 0])

    # Convert lists x and y into NumPy arrays for model prediction
    x, y = np.array(x), np.array(y)

    # Use the trained model to predict y values based on input sequences x
    predict = model.predict(x)

    # Calculate the inverse scaling factor to revert scaled predictions and actual values back to their original scale
    scale = 1 / scaler.scale_

    # Scale the predicted values and actual values back to their original scale
    predict = predict * scale
    y = y * scale

    # Display a subheader for 'Original Price vs Predicted Price' using Streamlit
    st.subheader('Original Price vs Predicted Price')
    
    def predicted_Based_On_MA():
         # Create a new figure for plotting with a specific size
        fig4 = plt.figure(figsize=(8, 6))

        # Plot the predicted prices in red ('r') with the label 'Predicted Price'
        plt.plot(predict, 'r', label='Predicted Price')

        # Plot the actual prices (y) in green ('g') with the label 'Original Price'
        plt.plot(y, 'g', label='Original Price')

        # Label the x-axis as 'Time' and the y-axis as 'Price'
        plt.xlabel('Time')
        plt.ylabel('Price')

        # Show the plot using Matplotlib
        plt.show()

        # Display the Matplotlib plot in Streamlit
        st.pyplot(fig4)
        
    if st.button('Predict'):
        # call the function
        with st.spinner('Please wait...'):
            predicted_Based_On_MA()
   
      
    
    
    
    
    
 
 
 # Footer
    
elif selected_page == "Trend Trackers":
    # Display the Markdown content in Streamlit
    # Title Of The Website
    st.markdown("""
        <style>
            .title {
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Trend Trackers</h1>", unsafe_allow_html=True)
    st.markdown("""

    ## Understanding Moving Averages

    Moving averages are essential tools in technical analysis, helping traders identify trends and potential reversals by smoothing out price data. In our approach, we primarily use the Simple Moving Average (SMA) to develop our trading model due to its effectiveness in identifying long-term trends.

    ### Simple Moving Average (SMA)
    The Simple Moving Average (SMA) calculates the average of a selected range of prices, usually closing prices, by the number of periods in that range. 

    **Advantages of SMA**:
    - **Simplicity**: Easy to calculate and understand.
    - **Long-term trends**: Ideal for identifying long-term trends due to its smooth nature.

    **Disadvantages of SMA**:
    - **Lag**: Can be slower to respond to recent price changes compared to other averages.

    ### Exponential Moving Average (EMA)
    The Exponential Moving Average (EMA) gives more weight to recent prices, making it more responsive to new information. 

    **Advantages of EMA**:
    - **Sensitivity**: More sensitive to recent price changes, making it suitable for short-term trading.
    - **Timely signals**: Provides quicker signals for potential market reversals.

    **Disadvantages of EMA**:
    - **Noise**: Can be more prone to false signals in choppy markets.

    ## When to Use SMA vs. EMA

    ### Long-term Trading with SMA
    SMAs are beneficial for long-term trading strategies as they smooth out price movements and are less affected by short-term volatility.

    ### Short-term Trading with EMA
    EMAs are more suitable for short-term trading strategies due to their sensitivity to recent price movements, allowing for quicker reaction times.

    ## How to Place Trades Based on Moving Averages

    ### Using SMA for Long-term Trades
    1. **Identify the Trend**: Look for the direction of the SMA (e.g., 200-day SMA) to determine the long-term trend.
    2. **Entry Signal**: Enter a trade when the price crosses above the SMA for a bullish signal or below the SMA for a bearish signal.
    3. **Exit Signal**: Exit the trade when the price crosses back in the opposite direction of your entry signal.

    ### Using EMA for Short-term Trades
    1. **Identify the Trend**: Use a shorter period EMA (e.g., 50-day EMA) to identify the short-term trend.
    2. **Entry Signal**: Enter a trade when the price crosses above the EMA for a bullish signal or below the EMA for a bearish signal.
    3. **Exit Signal**: Exit the trade when the price crosses back in the opposite direction of your entry signal.

    ---

    """)
    st.image('images/movingAverage.png', caption='Trade Placement Based on Moving Averages')
    ("""

    By understanding and utilizing these moving averages, traders can develop effective strategies to navigate various market conditions and time their trades more effectively.

                    """)

elif selected_page== "Indicator Insights":
    # Title Of The Website
    st.markdown("""
        <style>
            .title {
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Indicator Insights</h1>", unsafe_allow_html=True)
    st.markdown("""

                ## 1. Stochastic Oscillator

                The Stochastic Oscillator is a momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time. It is used to generate overbought and oversold trading signals, utilizing a 0-100 bounded range of values.

                - **Overbought Condition**: Above 80
                - **Oversold Condition**: Below 20

                ### How to Use:
                1. **Identify Overbought/Oversold Conditions**: Monitor the oscillator values relative to 80 (overbought) and 20 (oversold).
                2. **Entry/Exit Signals**: Enter a sell position when the oscillator falls below 80 from above. Enter a buy position when the oscillator rises above 20 from below.

                """)
    st.image('images/1.png', caption='Stochastic oscillator')
    
    st.markdown("""
                ## 2. Moving Average Convergence Divergence (MACD)

The MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price. It consists of the MACD line, signal line, and histogram.

- **MACD Line**: Difference between the 12-day EMA and 26-day EMA
- **Signal Line**: 9-day EMA of the MACD line
- **Histogram**: Difference between the MACD line and the Signal line

### How to Use:
1. **Crossovers**: Buy when the MACD line crosses above the signal line; sell when the MACD line crosses below the signal line.
2. **Divergence**: Identify potential reversals by spotting divergences between MACD and price movements.

                """)
    st.image('images/2.png', caption='Moving Average Convergence Divergence')
    
    st.markdown("""
                ## 3. Bollinger Bands

Bollinger Bands are volatility bands placed above and below a moving average. Volatility is based on the standard deviation, which changes as volatility increases and decreases.

- **Upper Band**: Moving average plus 2 standard deviations
- **Lower Band**: Moving average minus 2 standard deviations

### How to Use:
1. **Identify Volatility**: Bands expand with high volatility and contract with low volatility.
2. **Overbought/Oversold Conditions**: Prices close to the upper band indicate overbought conditions; prices near the lower band indicate oversold conditions.

                """)
    st.image('images/3.png', caption='Bollinger Bands')
    
    st.markdown("""
                ## 4. Relative Strength Index (RSI)

The RSI is a momentum oscillator that measures the speed and change of price movements. It oscillates between 0 and 100.

- **Overbought Condition**: Above 70
- **Oversold Condition**: Below 30

### How to Use:
1. **Identify Overbought/Oversold Conditions**: Monitor the RSI values relative to 70 (overbought) and 30 (oversold).
2. **Entry/Exit Signals**: Buy when RSI moves above 30 from below; sell when RSI moves below 70 from above.

                """)
    st.image('images/4.png', caption='Relative Strength Index ')
    
    st.markdown("""
                ## 5. Fibonacci Retracement

Fibonacci Retracement levels are horizontal lines that indicate where support and resistance are likely to occur. They are based on Fibonacci numbers and are expressed as percentages: 23.6%, 38.2%, 50%, 61.8%, and 100%.

### How to Use:
1. **Identify Key Levels**: Use the retracement levels to identify potential reversal levels during a trend.
2. **Entry/Exit Points**: Enter trades at retracement levels in the direction of the trend; exit trades at levels indicating potential reversals.

                """)
    st.image('images/5.png', caption='Fibonacci Retracement')
    
    st.markdown("""
                ## 6. Ichimoku Cloud

The Ichimoku Cloud is a comprehensive indicator that defines support and resistance, identifies trend direction, gauges momentum, and provides trading signals.

### How to Use:
1. **Trend Identification**: Prices above the cloud indicate an uptrend; prices below the cloud indicate a downtrend.
2. **Signal Generation**: Buy when the conversion line crosses above the baseline; sell when the conversion line crosses below the baseline.

                """)
    st.image('images/6.png', caption='Ichimoku Cloud')
    
    st.markdown("""
               ## 7. Standard Deviation

Standard Deviation is a statistical measure of market volatility. It calculates how much prices deviate from the mean or average price.

### How to Use:
1. **Measure Volatility**: High standard deviation indicates high volatility; low standard deviation indicates low volatility.
2. **Risk Assessment**: Use standard deviation to assess the risk associated with a security.

                """)
    st.image('images/7.png', caption='Standard Deviation')
    
    st.markdown("""
               ## 8. Average Directional Index (ADX)

The ADX is a trend strength indicator that ranges from 0 to 100. It helps investors determine the strength of a trend, not the direction.

- **Strong Trend**: ADX above 25
- **Weak Trend**: ADX below 20

### How to Use:
1. **Identify Trend Strength**: Use ADX to determine if the market is trending strongly or weakly.
2. **Entry/Exit Signals**: Combine with other indicators to confirm trends and identify entry/exit points.
                """)
    st.image('images/8.png', caption='Average Directional Index')

elif selected_page== "About" :
    st.markdown("""
        <style>
            .title {
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<h1 class='title'>About</h1>", unsafe_allow_html=True)
    
    st.markdown("""
StockSage is your ultimate companion for navigating the stock market with confidence and clarity. Whether you're a seasoned investor or just starting out, StockSage equips you with the tools and insights needed to make informed decisions.

## Our Mission

At StockSage, we believe in democratizing access to financial insights. Our platform offers:
- **Stock Analysis:** Enter any stock ticker symbol to dive deep into its performance metrics, historical trends, and analyst ratings.
- **Predictive Analytics:** Leverage advanced machine learning models to forecast potential price movements based on robust historical data analysis.
- **Technical Indicators:** Explore and understand key technical indicators such as moving averages (MA/EMA), stochastic oscillators, MACD, Bollinger Bands, RSI, Fibonacci retracement, Ichimoku Cloud, standard deviation, and ADX.

## Disclaimer

While StockSage provides powerful tools and insights, it's important to note that we are not registered with financial regulatory bodies. The information provided is for educational and informational purposes only. Always consult with a certified financial advisor or broker before making any investment decisions.

## About the Creator

I am **Aditya Sarkar**, the creator of StockSage. My passion for combining technology and finance led to the development of this project. I aim to make stock market analysis accessible and insightful for everyone.

Join StockSage today and embark on a journey to elevate your investment strategy with data-driven precision and confidence!
""")



st.markdown("""
    <style>
        .social-icons {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px; /* Adjust the gap between icons if needed */
        }
        .social-icons .icon {
            margin: 0 10px;
        }
    </style>
    <div class="social-icons">
        <a href="https://www.linkedin.com/in/aditya-sarkar-a7a321206/" target="_blank" class="icon">
            <img src="https://img.icons8.com/color/48/000000/linkedin.png"/>
        </a>
        <a href="https://www.instagram.com/adi_jong_un/" target="_blank" class="icon">
            <img src="https://img.icons8.com/color/48/000000/instagram-new.png"/>
        </a>
        <a href="https://github.com/SarkariKill" target="_blank" class="icon">
            <img src="https://img.icons8.com/material-rounded/48/000000/github.png"/>
        </a>
    </div>
    """, unsafe_allow_html=True)         
st.markdown("""           
---



Thank you for choosing StockSage.
Empowering your investment journey with data-driven insights and informed decisions!




                """)   