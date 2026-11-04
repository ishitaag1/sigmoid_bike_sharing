"""
Helpers for the bike demand case study.
The notebook and the demand assistant both use these, so the cleaning and
the model are the same in both places.

Model: linear regression on log(rentals) with one-hot features and an
hour x day-type interaction (see notebook section 5 for why).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- Assumptions (not in the data, ops should replace them) ---
AVG_TRIP_MINUTES = 20     # typical bike-share trip length
MAX_UTILIZATION = 0.70    # keep ~30% of bikes docked so riders can find one

# Planning buffer: in ~90% of training hours, actual demand was below
# (prediction x 1.5). Checked on the test months in notebook section 5.
PLANNING_BUFFER = 1.50

CATEGORICAL = ["hr_day", "mnth", "season", "weekday", "weathersit"]
NUMERIC = ["temp_c", "hum_pct", "wind_kmh", "yr", "holiday"]


def load_and_clean(path="hour.csv"):
    """Read hour.csv, fix the data issues found in QC, add readable columns."""
    df = pd.read_csv(path, parse_dates=["dteday"])

    # Normalised weather back to real units (divisors from the Readme)
    df["temp_c"] = df["temp"] * 41
    df["feels_like_c"] = df["atemp"] * 50
    df["hum_pct"] = df["hum"] * 100
    df["wind_kmh"] = df["windspeed"] * 67

    # 2011-03-10: humidity = 0 all day (impossible, it rained) -> March 2011 median
    bad = df["hum_pct"] == 0
    march_2011 = (df["yr"] == 0) & (df["mnth"] == 3) & ~bad
    df.loc[bad, "hum_pct"] = df.loc[march_2011, "hum_pct"].median()

    # Weather type 4 appears only 3 times -> merge into type 3 (bad weather)
    df["weathersit"] = df["weathersit"].replace(4, 3)

    df["datetime"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    df["day_type"] = np.where(df["workingday"] == 1, "Working day", "Weekend/Holiday")
    # The interaction: hour 8 on a working day and hour 8 on a weekend are
    # different categories, so the model can learn two separate daily shapes.
    df["hr_day"] = df["hr"].astype(str) + "_" + df["workingday"].astype(str)
    return df


def make_X(df, columns=None):
    """One-hot encode. Pass the training columns so new data lines up with them."""
    X = pd.get_dummies(df[NUMERIC + CATEGORICAL].astype({c: str for c in CATEGORICAL}),
                       columns=CATEGORICAL, dtype=float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def train_model(df):
    X = make_X(df)
    model = LinearRegression().fit(X, np.log1p(df["cnt"]))
    model.columns_ = list(X.columns)   # remember the column order
    return model


def predict(model, df):
    """Expected rentals per hour."""
    X = make_X(df, model.columns_)
    return np.expm1(model.predict(X)).clip(0)


def bikes_on_road(rentals_per_hour, trip_minutes=AVG_TRIP_MINUTES):
    """Little's Law: bikes in use at once = rentals per hour x trip length in hours."""
    return rentals_per_hour * trip_minutes / 60


def bikes_needed(rentals_per_hour, trip_minutes=AVG_TRIP_MINUTES, max_util=MAX_UTILIZATION):
    """Bikes to have available system-wide so riders can still find one."""
    return bikes_on_road(rentals_per_hour, trip_minutes) / max_util
