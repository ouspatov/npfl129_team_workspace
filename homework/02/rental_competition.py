#!/usr/bin/env python3

import argparse
import lzma
import os
import pickle
import sys
import urllib.request
from typing import Optional

import numpy as np
import numpy.typing as npt
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str)
parser.add_argument("--recodex", default=False, action="store_true")
parser.add_argument("--seed", default=42, type=int)
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument(
    "--model_path",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rental_competition.model"),
    type=str,
)

class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rented bikes in the given hour.
    """
    def __init__(
        self,
        name="rental_competition.train.npz",
        url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/",
    ):
        if not os.path.exists(name):
            print(f"Downloading dataset {name}...", file=sys.stderr)
            urllib.request.urlretrieve(
                url + name, filename=name + ".tmp"
            )
            os.rename(name + ".tmp", name)

        # Load the dataset and return the data and targets.
        with np.load(name) as dataset:
            for key, value in dataset.items():
                setattr(self, key, value)


def one_hot(values: np.ndarray, n: int) -> np.ndarray:
    return np.eye(n)[np.clip(values.astype(int), 0, n - 1)]


SPLINES = sklearn.preprocessing.SplineTransformer(
    n_knots=5,
    degree=3,
).fit(np.array([[0.0] * 4, [1.0] * 4]))


def features(data: np.ndarray, groups: tuple) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)

    season = data[:, 0] - 1
    year = data[:, 1]
    month = data[:, 2] - 1
    hour = data[:, 3].astype(int)
    holiday = data[:, 4]
    weekday = data[:, 5]
    work = data[:, 6]
    weather = data[:, 7] - 1

    numeric = data[:, 8:12]
    temp = numeric[:, [0]]
    humidity = numeric[:, [2]]

    weather_oh = one_hot(np.minimum(weather, 2), 3)
    phase = one_hot(
        np.digitize(hour, [6, 10, 15, 20]) + 5 * work,
        10,
    )

    parts = [
        one_hot(season, 4),
        one_hot(month, 12),
        one_hot(hour, 24),
        one_hot(weekday, 7),
        weather_oh,
        one_hot(hour + 24 * work, 48),
        year[:, None],
        holiday[:, None],
        work[:, None],
    ]

    if "poly" in groups:
        parts += [numeric, numeric ** 2, temp * humidity]

    if "spline" in groups:
        parts += [SPLINES.transform(np.clip(numeric, 0, 1))]

    if "phase_weather" in groups:
        parts += [
            (phase[:, :, None] * weather_oh[:, None, :])
            .reshape(len(data), -1)
        ]

    if "phase_temp" in groups:
        parts += [phase * temp, phase * humidity]

    if "phase_year" in groups:
        parts += [phase * year[:, None]]

    if "weather_temp" in groups:
        parts += [weather_oh * humidity, weather_oh * temp]

    return np.concatenate(parts, axis=1)


MEMBERS = [
    (("spline", "phase_weather", "phase_temp"), 4.0),
    (("poly", "phase_weather", "phase_temp"), 0.1),
    (("spline", "phase_temp", "weather_temp"), 3.0),
    (
        ("spline", "phase_weather", "phase_temp", "phase_year"),
        10.0,
    ),
]


def fit_members(data, target):
    members = []
    for groups, alpha in MEMBERS:
        pipeline = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(),
            sklearn.linear_model.PoissonRegressor(
                alpha=alpha,
                max_iter=3000,
            ),
        )
        pipeline.fit(features(data, groups), target)
        members.append((groups, pipeline))
    return members


def predict_members(members, data):
    predictions = [
        pipeline.predict(features(data, groups))
        for groups, pipeline in members
    ]
    return np.maximum(np.mean(predictions, axis=0), 0)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()

        members = fit_members(train.data, train.target)

        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(members, model_file)

        return None

    else:
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            members = pickle.load(model_file)

        return predict_members(members, test.data)


if __name__ == "__main__":
    main_args = parser.parse_args(
        [] if "__file__" not in globals() else None
    )
    main(main_args)
# e45adcd4-55ef-41de-ac3f-e2cd9863b321
# b5121daf-294b-4cbe-9d6c-23ba00ffecf8

