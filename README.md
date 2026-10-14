# Biogas Time Series ANN

This project focuses on forecasting biogas production using time-series data and artificial neural networks (ANNs) with TSMixer architecture. The goal is to model historical patterns in biogas output and use them to predict future values, supporting operational planning and process optimization.

## Overview

Biogas production is influenced by many time-dependent factors such as feedstock input, temperature, retention time, and seasonal variation. This project uses an ANN-based approach to learn from past observations and generate short- or medium-term predictions.

The repository is intended for:
- time-series preprocessing and feature engineering
- evaluating prediction quality
- visualizing forecasts against observed values

## Tech Stack (except standard libraries)

- u8darts[torch]
- matplotlib
- ipykernel

## Usage and application

The project is designed around `ANN_Driver.ipynb` notebook. Use it and follow instructions in cells. Raw model files are in `TSMixer_models.dep`. Raw test data in `rest_dayaset.pkl`

The model recognizes relationships in the biogas production signal and provide useful forecasts to support planning and process control. Depending on the data, the model can be tuned for different horizons and feedstocks.

## Notes

- Time-series forecasting requires careful handling of seasonality and temporal leakage.
- The choice of lookback window and feature set can significantly affect performance.
- Monitoring model performance on unseen future periods is essential for reliable deployment.

## License

See `LICENSE.MD`