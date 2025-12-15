# GPT2-Stock-Prediction

 Predicting stock trends using the time series model GPT-2.



<div align="center">


This is the official repository for **GPT2-Stock-Prediction**.

![made-for-VSCode](https://img.shields.io/badge/Made%20for-VSCode-1f425f.svg)

</div>

</div>





##  ✅ Overview of Time Series Forecasting Methods 

* **Autoregressive Models**
   Autoregressive models are among the simplest and most direct approaches to time series forecasting. Examples include AR, MA, ARMA, and ARIMA models. These methods forecast future values based on past observations and are best suited for stable and linear time series data.

* **Exponential Smoothing Models**
   Exponential smoothing models predict future values by applying weighted averages to historical data. They perform well on series with trends and seasonality. The Holt–Winters model is a classic example in this category.

* **Neural Network Models**
   Deep learning models such as Recurrent Neural Networks (RNN), Long Short-Term Memory networks (LSTM), and Gated Recurrent Units (GRU) have shown strong performance in time series prediction tasks. These models are particularly effective at capturing long-term dependencies and nonlinear patterns.

* **Convolutional Neural Network Models**
   Convolutional Neural Networks (CNNs) can be applied to time series forecasting to capture local patterns and features in the data. CNN-based models are especially useful when there are strong local correlations or spatial relationships within sequences.

* **Transformer Models**
   Originally developed for natural language processing, Transformer architectures have recently been applied to time series forecasting. Their ability to capture global dependencies makes them well suited for modeling long sequences.

* **Ensemble Models**
   Ensemble forecasting combines multiple base models, such as Random Forests and Gradient Boosting Trees, to produce more accurate and robust predictions than any single model.







## :bulb: Train based on GPT2

* **Successful implementation**: Transfer the model used for predicting text to predict floating-point numbers!

![LoRA for Qwen3](img/1.png)




##  :hourglass: Environment

* We use python==3.11 and pytorch == 2.3.0  with CUDA version 11.8
* Training with NVIDIA GeForce 4090 GPU is sufficient.
* Create environment:

```python
conda create --name lora python=3.11
conda activate gpt2
```

* Install the corresponding version pytorch:

```python
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

* Install other dependency packages:

```bash
Download GPT-2 Model's pth from huggingface into folder /model
```







##  ✅   Dataset Field Description

- **trade_date**: Trading date, the date on which the trade occurred.  
- **ts_code**: Stock code, uniquely identifies a stock.

---

- **open**: Opening price, the price at the start of trading for the day.  
- **high**: Highest price, the maximum trading price during the day.  
- **low**: Lowest price, the minimum trading price during the day.  
- **close**: Closing price, the price at the end of trading for the day.  
- **pre_close** (derivable): Previous close price, the closing price of the previous trading day.  
- **change** (possibly redundant but meaningful): Price change, the difference between the closing price and the previous close price.  
- **pct_chg** (derivable): Percentage price change, the ratio of price change to the previous close price, typically expressed as a percentage.  
- **vol**: Volume, the total number of shares traded during the day.  
- **amount**: Turnover, the total trading value during the day.  
- **turnover_rate**: Turnover rate, the ratio of volume to float shares, reflecting stock liquidity.  
- **volume_ratio** (redundant): Volume ratio, the ratio of the current day’s volume to the average volume of recent trading days, used to measure trading activity.





## 📧 Connecting with Us



If you have any questions, please feel free to send email to `hzcheng@chd.edu.cn`



## 📜 Acknowledgment



 This project is inspired by *Digital Video and HD: Algorithms and Interfaces*, *Deep Learning*( The Flower Book ), *Linear Algebra Done Right*, etc. 