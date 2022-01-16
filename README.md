# NFT Floor Price Prediction
Models to predict the daily floor price of NFT collection

This notebook shows how to use Covalent API to acquire historic data to build a floor price prediction model for Bored Ape Yacht Club (BAYC) collection. The model can be also used to predict other NFT collections by simply changing the NFT contract address.

## 1. Introduction

There are generally two types of price prediction that buyers are interested in:

1.   **Collection Floor Price Prediction:** To predict the lowest price of the entire collection on any given day.
2.   **Token Price Prediction:** To predict the average (or floor) price of a specific token wihtin one NFT collection. i.e. within the Chromie Squiggle collection, based on the different traits and features and historic sale price of the specific token, what is the price on any given day.

This notebook will only cover the floor price prediction model. The token price prediction can also be performed by acquiring token level data, such as traits, token level transactions and prices. 

## 2. Data Preparation

The historic NFT data can be downloaded from Covalent API by specifying the NFT contract address.

![image](https://user-images.githubusercontent.com/36990254/149682359-a5100483-6f9c-44f5-8381-374e45a67d1c.png)

The daily data of Bored Ape Yacht Club (BAYC) shows the full history of the NFT from April 30th 2020 to January 15th 2021, which is the time this data was downloaded from Covalent. 

You can also input a specific token ID to check out the traits and see the image.

![image](https://user-images.githubusercontent.com/36990254/149682376-2bf513d6-51eb-48cb-9bed-bbe2681f150f.png)

![image](https://user-images.githubusercontent.com/36990254/149682387-4994e0f5-ec6f-4347-bab4-fd6d44140456.png)

## 3. Floor Price Prediction Models
### 3.1 Model Options
The dependent variable to predict is the floor price of a given NFT collection. Since the past sale price is usually a good indicator of the future price, this can be interpreted as the prediction of a time-dependent event:
```
Y(c|tn) = f(w,x,Y(c|t0,...,tn-1))
```
where Y is the floor price of collection c at time tn and x represents the time dependent independent variables;  Y from t0 to tn-1 represent the past prices that can be used to model the price now. 

We know the prices of a lot NFTs have sky-rocketed since last year and there is surely a trend in the price. Given most of the time-series models require stationarity (no trend) in the data, generally Y needs to be transformed into a difference in order to remove trending in the price:
```
dY(c|t) = f(w,x,dY(c|t0,...,tn-1))
```
where dY is the percentage change in price of the collection (or token) c from month t-1 to t. 

Some of the most commonly used time-series models are:

- [Autoregressive model](https://en.wikipedia.org/wiki/Autoregressive_model)
- [ARMA](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model)
- [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)

Since ARIMA model incorporates the differencing terms dY(c|t), the lag of the difference dY(c|t-1) and the lag of the error terms e(c|t), as an example for demonstration, the last option ARIMA is chosen to predict the floor price in the below sections.


### 3.2 ARIMA
ARIMA models are generally the most general class of models for forecasting a time series which can be made to be “stationary” by differencing (see more details [here](https://people.duke.edu/~rnau/411arim.htm)). The model form ARIMA(p,d,q) comes with three components: 
- p is the number of autoregressive terms,
- d is the number of nonseasonal differences needed for stationarity, and
- q is the number of lagged forecast errors in the prediction equation. 

#### 3.2.1 Stationary

In order to decide what values of p, d and q to use in the ARIMA model, it's useful to have a look at floor price data itself, its first difference (or second or higher differences) along with its autocorrelation function [ACF](https://en.wikipedia.org/wiki/Autocorrelation) and partial autocorrelation function [PACF](https://en.wikipedia.org/wiki/Partial_autocorrelation_function).

The plots below show that the original daily floor price time-sereis is trending (non-stationary). Once the first difference is used, the series becomes stationary as shown in the ACF and PACF plots. The second difference is over-differencing as shown by the over-shooting lag from ACF and PACF. 

![image](https://user-images.githubusercontent.com/36990254/149682445-58be6de8-206a-4e33-a462-695ae47a0b11.png)

The stationarity Augmented Dickey–Fuller [ADF](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test) tests below also show the same statistically, where p value becomes small after first differencing. So we have an idea only first differencing (d=1) is needed to turn the trending data into stationary.

#### 3.2.2 Training, testing data split

In order to test better how the ARIMA model performs, the whole dataset is split into 80% for training and 20% for testing.

#### 3.2.3 Finding the right parameters

Since there are three different parameters p, d, q in the ARIMA model and the combinations of the parameter choices could get very large. Fortunately the 'pmdarima' package provides an automated ARIMA that uses a stepwise approach to search multiple combinations of p,d,q parameters and chooses the best model.

![image](https://user-images.githubusercontent.com/36990254/149682472-fcaddaf5-345b-46a1-8dba-4e491e911654.png)

### 3.3 Final Model

According to the automatic runs, the best model is ARIMA(1,1,0)(0,0,0), which is a model with first difference and one lag of the first difference. There is no seasonality or constant term (intercept) in the model.

#### 3.3.1 Residuals check
Once the final model is run, it's essential to also check the residuals to see if all the assumptions of ARIMA models are met. Four components of the residuals are checked:

Standardised residual: The errors fluctuate around a mean of zero and have a constant uniform variance.

Density: The empirical density and the kernal density estimation (KDE) of the floor price series suggest it has a mean of zero but has a thinner and more pointy shape than a normal distribution.

QQ Plot: The QQ-plot of the floor price against a normal distribution shows not all the dots are around the red line. The deviations towards the negative values imply the distribution is skewed, the same conclusion from the density plot.

Correlogram: The correlogram (or ACF) plot shows most of the points are within the confidence interval, so the residual errors are not autocorrelated. However, there is one point that pushes far to the negative side, which indicates there might be some pattern in the residuals that are not explained in the model. Adding more predictors (explanatory variables) might help improve the model.

![image](https://user-images.githubusercontent.com/36990254/149682501-3dc54d53-9464-4d8e-a656-9900236e3c69.png)

#### 3.3.2 Forecast
Using the optimal model ARIMA(1,1,0) trained from historic data from April 30th 2021 to November 24th 2021, you can predict the floor price from November 25th 2021 to January 15th 2022.

![image](https://user-images.githubusercontent.com/36990254/149682520-ca4f94ba-c8d8-433e-94c1-307202e898cf.png)

#### 3.3.3 Peformance
The model performance metrics compare the predicted and the actual floor price during the testing period from November 25th 2021 to January 15th 2022. The mean absolute percentage error of 6.2% means the model is approximately 93.8% accurate in predicting the next 52 days.

![image](https://user-images.githubusercontent.com/36990254/149682538-d00e90fd-2a36-4173-b4c5-9177d2e41360.png)


## 4. Conclusion

This notebook shows an example of how to use Covalent API to download historic daily NFT floor price; and build a simple time-series ARIMA model. Although the model performance is not bad, there are a few limitations and improvments that can be made:

- The residuals from the ARIMA (1,1,0) model are not exactly normally distributed, indicating there might be room to improve the model by adding some more exogenous variables such as sale volumes or even traits of the collection.
- Only a short history of daily floor price is used to predict BAYC in the example. There probably hasn't been enough cycles (ups and downs in trend) in the history for the model to be trained to predict well in the future.
- ARIMA is a generic and simple time-series model. There could be other models and also other exogenous variables that can be used to build a better model.


```
