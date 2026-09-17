"""
AgriNav model training — NASA-data-only, training code only.

Every feature value is pulled live from the real NASA POWER API
(https://power.larc.nasa.gov/api/temporal/daily/point) — no synthetic or
third-party (e.g. Kaggle) data. Nothing is generated locally except the two
target labels below, which POWER doesn't publish directly and which are
computed from the real NASA values via documented threshold rules.

1. Risk model:
   POWER has no ground-truth "risk_label" field, so labels are derived with
   domain-threshold rules on rainfall/temperature (documented in
   `label_risk_row`) — standard practice when no historical incident-labeled
   dataset exists.
   Input:  precipitation, temperature, humidity, solar_radiation, wind_speed,
           precip_7d, temp_7d_avg
   Output: risk_label (0=low, 1=medium, 2=high)

2. Crop-suitability model:
   Same NASA POWER features, matched against documented agro-climatic
   requirement ranges for common Bangladesh crops (`CROP_REQUIREMENTS`) to
   derive a "best_crop" label per day — also rule-based, since POWER has no
   crop-suitability field either.
   Input:  temperature, humidity, precip_7d, temp_7d_avg
   Output: best_crop (Aman Rice, Boro Rice, Wheat, Maize, Jute, Potato, Mustard)

Usage:
    python train_model.py --task risk                      # fetch real NASA POWER data, train
    python train_model.py --task risk --data data/risk_training_data.csv   # reuse a previous real export
    python train_model.py --task crop                       # NASA POWER data, crop-suitability labels
    python train_model.py --task both
    python train_model.py --task both --refresh              # force a fresh pull from NASA POWER
"""

import argparse
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

# NASA POWER's daily dataset has a processing lag of a few days, so "today"
# is never actually available yet — default the end date to 5 days back.
DEFAULT_START = "20200101"
DEFAULT_END = (datetime.today() - timedelta(days=5)).strftime("%Y%m%d")

RISK_FEATURES = [
    "precipitation",
    "temperature",
    "humidity",
    "solar_radiation",
    "wind_speed",
    "precip_7d",
    "temp_7d_avg",
]
RISK_TARGET = "risk_label"
RISK_CLASSES = ["low", "medium", "high"]

CROP_FEATURES = ["temperature", "humidity", "precip_7d", "temp_7d_avg"]
CROP_TARGET = "best_crop"

# Agro-climatic requirement ranges for common Bangladesh crops.
# (temp_min, temp_max) in Celsius, (rain_min, rain_max) = ideal 7-day
# cumulative rainfall in mm. Source: standard agronomy references (BARI/BRRI
# crop calendars) — used here as rule-based labeling, not a NASA field.
CROP_REQUIREMENTS = {
    "Aman Rice": {"temp": (20, 35), "rain_7d": (30, 120)},
    "Boro Rice": {"temp": (18, 33), "rain_7d": (0, 30)},
    "Wheat": {"temp": (10, 25), "rain_7d": (0, 20)},
    "Maize": {"temp": (18, 32), "rain_7d": (10, 60)},
    "Jute": {"temp": (24, 37), "rain_7d": (40, 150)},
    "Potato": {"temp": (15, 25), "rain_7d": (0, 20)},
    "Mustard": {"temp": (10, 25), "rain_7d": (0, 15)},
}

DATA_DIR = "data"
RAW_CSV = os.path.join(DATA_DIR, "nasa_power_raw.csv")
RISK_CSV = os.path.join(DATA_DIR, "risk_training_data.csv")
CROP_CSV = os.path.join(DATA_DIR, "crop_training_data.csv")

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMS = ["T2M", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN", "WS2M"]
POWER_COLUMN_MAP = {
    "T2M": "temperature",
    "PRECTOTCORR": "precipitation",
    "RH2M": "humidity",
    "ALLSKY_SFC_SW_DWN": "solar_radiation",
    "WS2M": "wind_speed",
}

# All 64 Bangladesh districts (name, lat, lon) — one point per district so the
# training data covers the country's full climate variation, not just a few
# major cities. Coordinates are each district's headquarters town.
BD_DISTRICTS = [
    # Dhaka division
    ("Dhaka", 23.8103, 90.4125),
    ("Faridpur", 23.6070, 89.8429),
    ("Gazipur", 23.9999, 90.4203),
    ("Gopalganj", 23.0050, 89.8266),
    ("Kishoreganj", 24.4449, 90.7766),
    ("Madaripur", 23.1641, 90.1897),
    ("Manikganj", 23.8644, 90.0047),
    ("Munshiganj", 23.5422, 90.5305),
    ("Narayanganj", 23.6238, 90.5000),
    ("Narsingdi", 23.9322, 90.7150),
    ("Rajbari", 23.7574, 89.6444),
    ("Shariatpur", 23.2423, 90.4348),
    ("Tangail", 24.2513, 89.9167),
    # Mymensingh division
    ("Mymensingh", 24.7471, 90.4203),
    ("Jamalpur", 24.9375, 89.9375),
    ("Netrokona", 24.8709, 90.7276),
    ("Sherpur", 25.0205, 90.0153),
    # Chattogram division
    ("Chattogram", 22.3569, 91.7832),
    ("Cox's Bazar", 21.4272, 92.0058),
    ("Cumilla", 23.4607, 91.1809),
    ("Brahmanbaria", 23.9571, 91.1119),
    ("Chandpur", 23.2333, 90.6667),
    ("Feni", 23.0159, 91.3976),
    ("Khagrachhari", 23.1193, 91.9847),
    ("Lakshmipur", 22.9447, 90.8282),
    ("Noakhali", 22.8696, 91.0995),
    ("Rangamati", 22.7324, 92.1936),
    ("Bandarban", 22.1953, 92.2184),
    # Rajshahi division
    ("Rajshahi", 24.3745, 88.6042),
    ("Bogura", 24.8465, 89.3773),
    ("Joypurhat", 25.0968, 89.0227),
    ("Naogaon", 24.7936, 88.9318),
    ("Natore", 24.4206, 88.9873),
    ("Chapainawabganj", 24.5965, 88.2775),
    ("Pabna", 24.0064, 89.2372),
    ("Sirajganj", 24.4534, 89.7010),
    # Khulna division
    ("Khulna", 22.8456, 89.5403),
    ("Bagerhat", 22.6602, 89.7895),
    ("Chuadanga", 23.6402, 88.8410),
    ("Jashore", 23.1667, 89.2167),
    ("Jhenaidah", 23.5450, 89.1539),
    ("Kushtia", 23.9013, 89.1200),
    ("Magura", 23.4873, 89.4198),
    ("Meherpur", 23.7622, 88.6318),
    ("Narail", 23.1725, 89.5126),
    ("Satkhira", 22.7185, 89.0705),
    # Barishal division
    ("Barishal", 22.7010, 90.3535),
    ("Barguna", 22.1591, 90.1120),
    ("Bhola", 22.6859, 90.6482),
    ("Jhalokati", 22.6406, 90.1987),
    ("Patuakhali", 22.3596, 90.3296),
    ("Pirojpur", 22.5841, 89.9720),
    # Sylhet division
    ("Sylhet", 24.8949, 91.8687),
    ("Habiganj", 24.3745, 91.4155),
    ("Moulvibazar", 24.4829, 91.7774),
    ("Sunamganj", 25.0658, 91.3950),
    # Rangpur division
    ("Rangpur", 25.7439, 89.2752),
    ("Dinajpur", 25.6217, 88.6354),
    ("Gaibandha", 25.3288, 89.5286),
    ("Kurigram", 25.8072, 89.6293),
    ("Lalmonirhat", 25.9923, 89.2847),
    ("Nilphamari", 25.9317, 88.8560),
    ("Panchagarh", 26.3411, 88.5541),
    ("Thakurgaon", 26.0336, 88.4616),
]


def fetch_power_data(lat, lon, start, end):
    """Pulls daily meteorological data for one point from the NASA POWER API.
    No auth needed, it's a public endpoint. Returns a DataFrame indexed by date.
    """
    params = {
        "parameters": ",".join(POWER_PARAMS),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
    }
    resp = requests.get(POWER_URL, params=params, timeout=60)
    resp.raise_for_status()
    parameter_data = resp.json()["properties"]["parameter"]

    df = pd.DataFrame(parameter_data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.rename(columns=POWER_COLUMN_MAP)
    df = df.replace(-999, np.nan).dropna()
    return df.sort_index()


def label_risk_row(row):
    """Domain-threshold labeling: high risk if the land is drying out
    (low 7-day rainfall) or under sustained heat stress; medium risk on
    borderline values; low risk otherwise."""
    if row["precip_7d"] < 5 or row["temp_7d_avg"] > 35:
        return 2  # high
    if row["precip_7d"] < 20 or row["temp_7d_avg"] > 32:
        return 1  # medium
    return 0  # low


def crop_fit_score(row, requirements):
    """Higher is better: 0 when temp/rain sit at the range center, more
    negative the further outside the crop's ideal range they fall."""
    t_lo, t_hi = requirements["temp"]
    r_lo, r_hi = requirements["rain_7d"]
    t_mid, t_half = (t_lo + t_hi) / 2, (t_hi - t_lo) / 2
    r_mid, r_half = (r_lo + r_hi) / 2, (r_hi - r_lo) / 2

    t_dev = (row["temp_7d_avg"] - t_mid) / t_half
    r_dev = (row["precip_7d"] - r_mid) / r_half
    return -(t_dev**2 + r_dev**2)


def label_best_crop(row):
    scores = {crop: crop_fit_score(row, reqs) for crop, reqs in CROP_REQUIREMENTS.items()}
    return max(scores, key=scores.get)


def build_power_dataset(locations=BD_DISTRICTS, start=DEFAULT_START, end=DEFAULT_END, refresh=False):
    """Fetches real NASA POWER daily data for each location and engineers
    the rolling features shared by both models. Saved incrementally to
    data/nasa_power_raw.csv — one district at a time, flushed to disk and
    printed immediately, so progress is visible on disk even mid-run.
    Re-running resumes from whichever districts are already in the CSV;
    pass refresh=True to wipe the cache and re-fetch everything."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if refresh and os.path.exists(RAW_CSV):
        os.remove(RAW_CSV)

    done = set()
    if os.path.exists(RAW_CSV):
        done = set(pd.read_csv(RAW_CSV, usecols=["location"])["location"].unique())
        print(f"Resuming: {len(done)}/{len(locations)} districts already in {RAW_CSV}", flush=True)

    remaining = [loc for loc in locations if loc[0] not in done]
    for name, lat, lon in remaining:
        print(f"Fetching NASA POWER data for {name} ({lat}, {lon})...", flush=True)
        df = fetch_power_data(lat, lon, start, end)
        df["precip_7d"] = df["precipitation"].rolling(7, min_periods=1).sum()
        df["temp_7d_avg"] = df["temperature"].rolling(7, min_periods=1).mean()
        df["location"] = name
        df = df.reset_index()
        df.to_csv(RAW_CSV, mode="a", header=not os.path.exists(RAW_CSV), index=False)
        print(f"  -> saved {len(df)} rows for {name} ({len(done) + 1}/{len(locations)})", flush=True)
        done.add(name)

    full = pd.read_csv(RAW_CSV, parse_dates=["date"])
    print(
        f"NASA POWER raw dataset complete: {len(full)} rows, "
        f"{full['location'].nunique()} districts -> {RAW_CSV}",
        flush=True,
    )
    return full


def build_power_risk_dataset(locations=BD_DISTRICTS, start=DEFAULT_START, end=DEFAULT_END, refresh=False):
    full = build_power_dataset(locations, start, end, refresh=refresh)
    full[RISK_TARGET] = full.apply(label_risk_row, axis=1)
    os.makedirs(DATA_DIR, exist_ok=True)
    full.to_csv(RISK_CSV, index=False)
    print(f"Saved risk training data -> {RISK_CSV}")
    return full


def build_power_crop_dataset(locations=BD_DISTRICTS, start=DEFAULT_START, end=DEFAULT_END, refresh=False):
    full = build_power_dataset(locations, start, end, refresh=refresh)
    full[CROP_TARGET] = full.apply(label_best_crop, axis=1)
    os.makedirs(DATA_DIR, exist_ok=True)
    full.to_csv(CROP_CSV, index=False)
    print(f"Saved crop training data -> {CROP_CSV}")
    return full


def chronological_split(df: pd.DataFrame, test_frac=0.2):
    """Splits by date instead of randomly. A random row split would leak
    information between adjacent days at the same location (precip_7d and
    temp_7d_avg are rolling windows, so neighboring rows are correlated) and
    make accuracy look better than real-world generalization. Holding out the
    most recent slice of dates for every location is the honest test: the
    model has never seen those dates for any location during training."""
    dates = np.sort(df["date"].unique())
    cutoff = dates[int(len(dates) * (1 - test_frac))]
    train_df = df[df["date"] < cutoff]
    test_df = df[df["date"] >= cutoff]
    print(f"Train dates: {train_df['date'].min().date()} -> {train_df['date'].max().date()}")
    print(f"Test dates:  {test_df['date'].min().date()} -> {test_df['date'].max().date()}")
    return train_df, test_df


def train_risk_model(df: pd.DataFrame, out_path: str):
    train_df, test_df = chronological_split(df)
    X_train, y_train = train_df[RISK_FEATURES], train_df[RISK_TARGET]
    X_test, y_test = test_df[RISK_FEATURES], test_df[RISK_TARGET]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n=== Risk model (NASA POWER) ===")
    print(f"5-fold CV accuracy on training data: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print("Held-out future dates:")
    print(classification_report(y_test, y_pred, target_names=RISK_CLASSES))
    print(confusion_matrix(y_test, y_pred))
    print(
        pd.Series(model.feature_importances_, index=RISK_FEATURES)
        .sort_values(ascending=False)
        .rename("importance")
    )

    joblib.dump(model, out_path)
    print(f"Saved risk model -> {out_path}")


def train_crop_model(df: pd.DataFrame, out_path: str):
    train_df, test_df = chronological_split(df)
    X_train, y_train = train_df[CROP_FEATURES], train_df[CROP_TARGET]
    X_test, y_test = test_df[CROP_FEATURES], test_df[CROP_TARGET]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
    )
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n=== Crop suitability model (NASA POWER) ===")
    print(f"5-fold CV accuracy on training data: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print("Held-out future dates:")
    print(classification_report(y_test, y_pred))
    print(
        pd.Series(model.feature_importances_, index=CROP_FEATURES)
        .sort_values(ascending=False)
        .rename("importance")
    )

    joblib.dump(model, out_path)
    print(f"Saved crop model -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["risk", "crop", "both"], default="risk")
    parser.add_argument("--start", type=str, default=DEFAULT_START, help="POWER start date YYYYMMDD")
    parser.add_argument("--end", type=str, default=DEFAULT_END, help="POWER end date YYYYMMDD")
    parser.add_argument("--data", type=str, default=None, help="Use this real CSV (e.g. an earlier data/*.csv export) instead of fetching from NASA POWER")
    parser.add_argument("--out", type=str, default=None, help="Output path for --task risk or crop")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch from NASA POWER instead of using data/nasa_power_raw.csv cache")
    args = parser.parse_args()

    if args.task in ("risk", "both"):
        if args.data:
            risk_df = pd.read_csv(args.data, parse_dates=["date"])
        else:
            risk_df = build_power_risk_dataset(start=args.start, end=args.end, refresh=args.refresh)
        train_risk_model(risk_df, args.out or "crop_risk_model.pkl")

    if args.task in ("crop", "both"):
        crop_df = build_power_crop_dataset(start=args.start, end=args.end, refresh=args.refresh)
        train_crop_model(crop_df, args.out or "crop_suitability_model.pkl")


if __name__ == "__main__":
    main()
