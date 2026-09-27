# Blinkit Demand Forecasting (ARIMA vs SARIMAX vs Prophet)

Daily demand forecasting for a quick-commerce (Blinkit) dataset, comparing a classical statistical approach (SARIMAX) against a modern automated forecasting tool (Prophet).

## Overview

This project forecasts daily total order value (`order_total`) for a quick-commerce platform. It follows the full time series pipeline - checking stationarity, picking the right model settings, building three different forecasting models, and honestly comparing how well each one actually performs on real, unseen data.

## Dataset

- **Source:** blinkit_orders.csv (order-level transaction data)
- **Date range:** March 2023 - November 2024 (~600 days)
- **Target:** Daily total order value, aggregated from individual orders
- **Split:** 480 days train / 120 days test (chronological - no shuffling, since order matters in time series)

## What was done, step by step

1. **Cleaned the data** - converted dates properly, confirmed there were no missing values
2. **Resampled to daily totals** - summed all orders within each day into one number per day
3. **Split into train/test BEFORE anything else** - to make sure the model is only ever trained on the past, and tested on data it has never seen, avoiding a common mistake (data leakage)
4. **Checked for stationarity** - tested with both ADF and KPSS; both agreed the data was stationary, so no differencing was needed
5. **Found the right model settings (p, d, q)** - read this off ACF and PACF plots, landing on p=1, q=1
6. **Built three models and compared them honestly:**
   - **ARIMA(1,0,1)** - the simplest model, using no seasonal component
   - **SARIMAX(1,0,1)(1,0,1,7)** - same as ARIMA, but with a weekly seasonal pattern added in
   - **Prophet** - Meta's more automated forecasting tool, needing far less manual tuning

## Results

| Model | RMSE | MAE | Roughly how far off (% of daily average) |
|---|---|---|---|
| ARIMA(1,0,1) | High AIC, flat forecast | - | Not usable - just predicts a flat average, ignores real patterns |
| **SARIMAX(1,0,1)(1,0,1,7)** | **8,083.49** | **6,529.99** | **~37%** |
| Prophet | 8,522.68 | 6,897.93 | ~39% |

**What these numbers mean in plain words:** on a typical day, both SARIMAX and Prophet's forecasts were off by roughly six and a half to seven thousand rupees, against an average day of about seventeen thousand rupees in sales. That is a meaningful miss - useful as a rough guide, but not precise enough to fully automate stocking decisions on its own.

**Forecast Bias:** +405 (about 2% of the daily average) - this tells us the model is not consistently guessing too high or too low. The error is more about day-to-day imprecision than the model leaning in the wrong direction.

## ARIMA vs SARIMAX vs Prophet - what actually happened

- **ARIMA failed outright.** With no seasonal component, it basically gave up and predicted close to the same flat number every day - not useful for a business that clearly has weekly demand swings.
- **SARIMAX won on accuracy.** Every coefficient in the model (including the seasonal ones) was statistically significant, confirming Blinkit's demand genuinely has a real 7-day weekly pattern. It slightly beat Prophet on both RMSE and MAE.
- **Prophet came a close second, with far less manual work.** No stationarity testing, no reading ACF/PACF plots by hand - Prophet figured out the seasonality mostly on its own, and even gives a built-in confidence range around each prediction, which SARIMAX doesn't provide out of the box.

## Should this be deployed?

Not as-is, at least not fully automated. Both SARIMAX and Prophet's ~37-39% error is above the roughly 10-20% error range that is generally considered "trustworthy" for automated inventory decisions in real demand planning. That said, the forecast is not useless - it correctly captures the real weekly pattern in demand, and can genuinely support planning decisions if paired with:

- A safety stock buffer sized around the model's own error margin
- A human planner reviewing the forecast before it becomes a final order
- A shorter forecast horizon (this project forecasted 120 days ahead in one go, which is a long stretch - accuracy is usually much better closer in)

**If choosing between the two for a real deployment:** SARIMAX for a single product where accuracy matters most and there is time to tune it properly. Prophet if this needed to scale across hundreds of different products quickly, where manually tuning a separate SARIMAX model for each one just isn't realistic.

## Tech Stack

- Python, Pandas, NumPy
- Statsmodels (ADF, KPSS, ACF/PACF, ARIMA, SARIMAX)
- Prophet
- Scikit-learn (RMSE, MAE)
- Matplotlib

## How to run

```bash
pip install pandas numpy statsmodels scikit-learn matplotlib prophet
jupyter notebook forecasting.ipynb
```

## Next steps

- Calculate MAPE/WMAPE directly for a precise, business-ready accuracy percentage
- Try a shorter forecast horizon (7-30 days) to see how much accuracy improves
- Add real external features (promotions, holidays) as inputs
- Try XGBoost with lag/rolling features as a third comparison point
- This dataset is also being used separately for a late-delivery classification project, a customer/product segmentation project, and a regression project on order value drivers
