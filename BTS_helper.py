import os
import tempfile
import zipfile

import torch
from darts.models import TSMixerModel

#Function to generate continuous time series forecast
def historical_forecasts(model, dataset, forecast_horizon, scaler):
    """Generates probabilistic historical forecasts for each transformer
    and returns the inverse transformed results.

    Each forecast covers 24h (forecast_horizon). The time between two forecasts
    (stride) is also 24 hours.

    Combines multiple forecast chunks into a continuous time series.
    The historical forecasts are generated with strides, so we need to be careful about ordering.
    """
    # Do Forecast
    hfc = model.historical_forecasts(
        series=[dataset],
        forecast_horizon=forecast_horizon,
        stride=forecast_horizon,
        last_points_only=False,
        retrain=False,
        verbose=True,
    )
    hfc = scaler.inverse_transform(hfc)
    # Initialize the final continuous series with the first forecast
    continuous_series = hfc[0][0]
    # Append the remaining forecasts to the continuous series
    for series in hfc[0][1:]:
        continuous_series = continuous_series.append(series)
    return continuous_series

# Main model definitions
class ScaledLoss(torch.nn.Module):
    def __init__(self, scale_factor=1.0):
        super(ScaledLoss, self).__init__()
        self.scale_factor = scale_factor
        self.raw_loss = torch.nn.MSELoss()

    def forward(self, input, target):
        loss = self.raw_loss(input, target)
        return loss * self.scale_factor

def load_all_models(zip_path: str, step_intervals: list):
    models = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        for interval in step_intervals:
            file_path = os.path.join(temp_dir, f"tsmixer_model_{interval}.pt")

            # Przypisujemy model bezpośrednio do klucza będącego liczbą (np. models[8])
            models[interval] = TSMixerModel.load(file_path)

    return models