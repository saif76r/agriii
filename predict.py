"""
AgriNav prediction — uses the two models trained by train_model.py.
Reuses train_model.py's NASA POWER fetch code, district list, and feature
definitions so there is one source of truth for both training and inference.

Eight things this script can do:

1. Predict risk + best crop for one district on one *past/present* date
   (uses that day's actual NASA POWER readings)
       python predict.py predict --district Dhaka --date 20260601

2. National snapshot — risk + best crop for all 64 districts on one date
       python predict.py snapshot --date 20260601

3. Crop calendar — for every district, the most-recommended crop per month,
   built from the full 2020-2026 training data already in data/
       python predict.py calendar

4. Outlook — risk + best crop for a *future* date (next month, next season,
   even next year), where NASA POWER obviously has no actual reading yet.
   Instead of forecasting weather with a separate model, this uses NASA
   POWER's own multi-year (2020-2026) climatology: the average of the real
   NASA values recorded on that same calendar day (+/- 10 days) across every
   year in the cache. This is the standard "climatology as forecast"
   approach — still 100% NASA-sourced data, no synthetic values, no
   third-party weather API.
       python predict.py outlook --district Dhaka --date 20271215

5. Map — PNG of all 64 districts colored by risk (green/yellow/red) for one date
       python predict.py map --date 20260601

6. Alert — plain-language high-risk warnings for one date, all districts
   (falls back to climatology automatically if the date is in the future)
       python predict.py alert --date 20271215

7. Compare — one district's *actual* NASA-recorded values on the same
   calendar day across several real years, to see year-to-year change
       python predict.py compare --district Rangpur --years 2021,2022,2023,2024,2025 --month-day 0715

8. Rank — all 64 districts ranked by how well-suited they are to one crop
   on one date
       python predict.py rank --crop "Aman Rice" --date 20260701

Dates already covered by data/nasa_power_raw.csv (2020-01-01 to ~today-5d)
are read straight from that cache for `predict`/`snapshot`. A near-future
date outside that range triggers a live NASA POWER fetch for just the 7
days needed to compute the rolling features — still real NASA data, never
generated. Far-future dates use `outlook` (climatology) instead.
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd

import train_model as tm

RISK_MODEL_PATH = "crop_risk_model.pkl"
CROP_MODEL_PATH = "crop_suitability_model.pkl"

DISTRICT_COORDS = {name: (lat, lon) for name, lat, lon in tm.BD_DISTRICTS}


def load_models():
    if not (os.path.exists(RISK_MODEL_PATH) and os.path.exists(CROP_MODEL_PATH)):
        raise FileNotFoundError(
            "Model files not found. Run `python train_model.py --task both` first."
        )
    return joblib.load(RISK_MODEL_PATH), joblib.load(CROP_MODEL_PATH)


def get_features_for(district, date_str, cache=None):
    """Returns a dict of the 7 raw+rolling features for one district on one
    date. Uses the cached NASA POWER dataset when the date is already in it;
    otherwise fetches the last 7 days from NASA POWER live."""
    if district not in DISTRICT_COORDS:
        raise ValueError(f"Unknown district '{district}'. See train_model.BD_DISTRICTS.")

    target_date = pd.to_datetime(date_str, format="%Y%m%d")

    if cache is None and os.path.exists(tm.RAW_CSV):
        cache = pd.read_csv(tm.RAW_CSV, parse_dates=["date"])

    if cache is not None:
        row = cache[(cache["location"] == district) & (cache["date"] == target_date)]
        if not row.empty:
            return row.iloc[0].to_dict()

    lat, lon = DISTRICT_COORDS[district]
    start = (target_date - pd.Timedelta(days=7)).strftime("%Y%m%d")
    end = target_date.strftime("%Y%m%d")
    df = tm.fetch_power_data(lat, lon, start, end)
    if target_date not in df.index:
        raise ValueError(
            f"NASA POWER has no data for {district} on {target_date.date()} "
            "(too recent — POWER has a few days' processing lag, or too far in the future)."
        )
    df["precip_7d"] = df["precipitation"].rolling(7, min_periods=1).sum()
    df["temp_7d_avg"] = df["temperature"].rolling(7, min_periods=1).mean()
    return df.loc[target_date].to_dict()


def climatology_features(cache, district, target_date, window_days=10):
    """Average of the real NASA POWER values recorded on this same calendar
    day (+/- window_days) across every year already in the cache. This is
    what lets `outlook` answer questions about dates NASA hasn't measured
    yet, without inventing any data or bolting on a separate forecasting
    model — it's the standard climatology-as-forecast technique."""
    sub = cache[cache["location"] == district].copy()
    if sub.empty:
        raise ValueError(f"No cached NASA POWER data for '{district}'.")

    target_doy = target_date.dayofyear
    doy = sub["date"].dt.dayofyear
    diff = (doy - target_doy).abs()
    diff = np.minimum(diff, 365 - diff)  # wrap around the Dec/Jan boundary
    match = sub[diff <= window_days]
    if match.empty:
        raise ValueError(f"Not enough historical NASA data near day-of-year {target_doy} for '{district}'.")

    cols = ["precipitation", "temperature", "humidity", "solar_radiation", "wind_speed", "precip_7d", "temp_7d_avg"]
    features = match[cols].mean().to_dict()
    years = sorted(match["date"].dt.year.unique().tolist())
    features["years_averaged"] = len(years)
    features["years_list"] = ",".join(str(y) for y in years)
    features["samples_averaged"] = int(len(match))
    return features


def get_features_any(district, date_str, cache):
    """Actual NASA reading if the date is in the cache (or fetchable live);
    falls back to NASA POWER climatology for dates too far in the future."""
    try:
        return get_features_for(district, date_str, cache=cache), "actual"
    except ValueError:
        target_date = pd.to_datetime(date_str, format="%Y%m%d")
        return climatology_features(cache, district, target_date), "climatology"


def score_crop_fit(features, crop):
    return tm.crop_fit_score(features, tm.CROP_REQUIREMENTS[crop])


def predict_one(district, date_str, risk_model, crop_model, cache=None):
    features = get_features_for(district, date_str, cache=cache)
    risk_x = pd.DataFrame([{k: features[k] for k in tm.RISK_FEATURES}])
    crop_x = pd.DataFrame([{k: features[k] for k in tm.CROP_FEATURES}])

    risk_idx = risk_model.predict(risk_x)[0]
    risk_proba = risk_model.predict_proba(risk_x)[0]
    best_crop = crop_model.predict(crop_x)[0]

    return {
        "district": district,
        "date": date_str,
        "risk": tm.RISK_CLASSES[risk_idx],
        "risk_confidence": round(float(risk_proba.max()), 3),
        "best_crop": best_crop,
        "precipitation": round(float(features["precipitation"]), 2),
        "temperature": round(float(features["temperature"]), 2),
        "precip_7d": round(float(features["precip_7d"]), 2),
        "temp_7d_avg": round(float(features["temp_7d_avg"]), 2),
    }


def cmd_predict(args):
    risk_model, crop_model = load_models()
    result = predict_one(args.district, args.date, risk_model, crop_model)
    for k, v in result.items():
        print(f"{k:16s}: {v}")


def cmd_snapshot(args):
    risk_model, crop_model = load_models()
    cache = pd.read_csv(tm.RAW_CSV, parse_dates=["date"]) if os.path.exists(tm.RAW_CSV) else None

    rows = []
    for district in DISTRICT_COORDS:
        try:
            rows.append(predict_one(district, args.date, risk_model, crop_model, cache=cache))
        except ValueError as e:
            print(f"Skipping {district}: {e}")

    snapshot = pd.DataFrame(rows).sort_values("risk", ascending=False)
    os.makedirs(tm.DATA_DIR, exist_ok=True)
    out_path = os.path.join(tm.DATA_DIR, f"national_risk_snapshot_{args.date}.csv")
    snapshot.to_csv(out_path, index=False)

    print(snapshot.to_string(index=False))
    print(f"\nHigh-risk districts: {(snapshot['risk'] == 'high').sum()} / {len(snapshot)}")
    print(f"Saved -> {out_path}")


def outlook_one(district, target_date, cache, risk_model, crop_model):
    features = climatology_features(cache, district, target_date)
    risk_x = pd.DataFrame([{k: features[k] for k in tm.RISK_FEATURES}])
    crop_x = pd.DataFrame([{k: features[k] for k in tm.CROP_FEATURES}])

    risk_idx = risk_model.predict(risk_x)[0]
    risk_proba = risk_model.predict_proba(risk_x)[0]
    best_crop = crop_model.predict(crop_x)[0]

    return {
        "date": target_date.date(),
        "risk": tm.RISK_CLASSES[risk_idx],
        "risk_confidence": round(float(risk_proba.max()), 3),
        "best_crop": best_crop,
        "avg_precip_7d": round(float(features["precip_7d"]), 2),
        "avg_temp_7d_avg": round(float(features["temp_7d_avg"]), 2),
        "years_averaged": features["years_averaged"],
        "source_years": features["years_list"],
    }


def cmd_outlook(args):
    if not os.path.exists(tm.RAW_CSV):
        raise FileNotFoundError(f"{tm.RAW_CSV} not found. Run train_model.py --task risk first.")
    if args.district not in DISTRICT_COORDS:
        raise ValueError(f"Unknown district '{args.district}'. See train_model.BD_DISTRICTS.")

    risk_model, crop_model = load_models()
    cache = pd.read_csv(tm.RAW_CSV, parse_dates=["date"])
    target_date = pd.to_datetime(args.date, format="%Y%m%d")

    result = outlook_one(args.district, target_date, cache, risk_model, crop_model)
    print(f"Climatology outlook for {args.district} around {result['date']}")
    print(f"(This is the average of REAL NASA POWER readings from {result['source_years']} "
          f"on this same calendar day +/- 10 days — no invented values.)")
    for k, v in result.items():
        print(f"{k:16s}: {v}")


def cmd_season(args):
    if not os.path.exists(tm.RAW_CSV):
        raise FileNotFoundError(f"{tm.RAW_CSV} not found. Run train_model.py --task risk first.")
    if args.district not in DISTRICT_COORDS:
        raise ValueError(f"Unknown district '{args.district}'. See train_model.BD_DISTRICTS.")

    risk_model, crop_model = load_models()
    cache = pd.read_csv(tm.RAW_CSV, parse_dates=["date"])
    year = int(args.year)

    rows = []
    for month in range(1, 13):
        target_date = pd.Timestamp(year=year, month=month, day=15)
        result = outlook_one(args.district, target_date, cache, risk_model, crop_model)
        result["month"] = target_date.strftime("%B")
        rows.append(result)

    table = pd.DataFrame(rows)[["month", "avg_precip_7d", "avg_temp_7d_avg", "risk", "best_crop", "source_years"]]

    note = {
        "low": "normal",
        "medium": "moderate risk — keep watch",
        "high": " drought/heat risk",
    }
    table["note"] = table["risk"].map(note)

    print(f"Year-round climatology outlook for {args.district}, {year}")
    print("Every number below is the average of REAL NASA POWER readings from the")
    print("'source_years' column on that same calendar day +/- 10 days — nothing invented.")
    print(table.to_string(index=False))

    os.makedirs(tm.DATA_DIR, exist_ok=True)
    out_path = os.path.join(tm.DATA_DIR, f"season_{args.district}_{year}.csv")
    table.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


def cmd_calendar(args):
    if not os.path.exists(tm.CROP_CSV):
        raise FileNotFoundError(f"{tm.CROP_CSV} not found. Run train_model.py --task crop first.")

    df = pd.read_csv(tm.CROP_CSV, parse_dates=["date"])
    df["month"] = df["date"].dt.strftime("%b")

    calendar = (
        df.groupby(["location", "month"])[tm.CROP_TARGET]
        .agg(lambda s: s.mode().iloc[0])
        .unstack("month")
    )
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    calendar = calendar[[m for m in month_order if m in calendar.columns]]

    out_path = os.path.join(tm.DATA_DIR, "crop_calendar.csv")
    calendar.to_csv(out_path)
    print(calendar.to_string())
    print(f"\nSaved -> {out_path}")


def cmd_map(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    risk_model, crop_model = load_models()
    cache = pd.read_csv(tm.RAW_CSV, parse_dates=["date"]) if os.path.exists(tm.RAW_CSV) else None

    rows = []
    for district in DISTRICT_COORDS:
        try:
            rows.append(predict_one(district, args.date, risk_model, crop_model, cache=cache))
        except ValueError as e:
            print(f"Skipping {district}: {e}")

    snapshot = pd.DataFrame(rows)
    color_map = {"low": "#2ca02c", "medium": "#f0ad4e", "high": "#d62728"}
    colors = snapshot["risk"].map(color_map)
    lats = [DISTRICT_COORDS[d][0] for d in snapshot["district"]]
    lons = [DISTRICT_COORDS[d][1] for d in snapshot["district"]]

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.scatter(lons, lats, c=colors, s=110, edgecolors="black", linewidths=0.5, zorder=3)
    for lon, lat, name in zip(lons, lats, snapshot["district"]):
        ax.annotate(name, (lon, lat), fontsize=5.5, ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
    ax.set_title(f"AgriNav — drought/heat risk by district ({args.date})\nNASA POWER data")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=lbl)
               for lbl, c in color_map.items()]
    ax.legend(handles=handles, title="Risk", loc="lower right")
    fig.tight_layout()

    os.makedirs(tm.DATA_DIR, exist_ok=True)
    out_path = os.path.join(tm.DATA_DIR, f"risk_map_{args.date}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")
    print(snapshot["risk"].value_counts().to_string())


def cmd_alert(args):
    if not os.path.exists(tm.RAW_CSV):
        raise FileNotFoundError(f"{tm.RAW_CSV} not found. Run train_model.py --task risk first.")

    risk_model, crop_model = load_models()
    cache = pd.read_csv(tm.RAW_CSV, parse_dates=["date"])
    target_date = pd.to_datetime(args.date, format="%Y%m%d")

    alerts = []
    for district in DISTRICT_COORDS:
        try:
            features, source = get_features_any(district, args.date, cache)
        except ValueError as e:
            print(f"Skipping {district}: {e}")
            continue

        risk_x = pd.DataFrame([{k: features[k] for k in tm.RISK_FEATURES}])
        crop_x = pd.DataFrame([{k: features[k] for k in tm.CROP_FEATURES}])
        risk_idx = risk_model.predict(risk_x)[0]
        risk_label = tm.RISK_CLASSES[risk_idx]
        best_crop = crop_model.predict(crop_x)[0]

        if risk_label == "high":
            reason = "low rainfall / drought risk" if features["precip_7d"] < 5 else "excess heat"
            provenance = (
                f"NASA POWER real years averaged: {features['years_list']}"
                if source == "climatology"
                else f"NASA POWER actual reading for {target_date.date()}"
            )
            alerts.append(
                f"\U0001F6A8 {district} — {target_date.date()} ({source}): HIGH risk ({reason}). "
                f"Recommended crop: {best_crop}. avg_precip_7d={features['precip_7d']:.1f}mm, "
                f"avg_temp_7d={features['temp_7d_avg']:.1f}C  [{provenance}]"
            )

    os.makedirs(tm.DATA_DIR, exist_ok=True)
    out_path = os.path.join(tm.DATA_DIR, f"alerts_{args.date}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(alerts) if alerts else "No high-risk districts for this date.\n")

    print(f"{len(alerts)} high-risk district(s) out of {len(DISTRICT_COORDS)}")
    for a in alerts:
        print(a)
    print(f"\nSaved -> {out_path}")


def cmd_compare(args):
    if not os.path.exists(tm.RAW_CSV):
        raise FileNotFoundError(f"{tm.RAW_CSV} not found. Run train_model.py --task risk first.")
    if args.district not in DISTRICT_COORDS:
        raise ValueError(f"Unknown district '{args.district}'. See train_model.BD_DISTRICTS.")

    risk_model, crop_model = load_models()
    cache = pd.read_csv(tm.RAW_CSV, parse_dates=["date"])

    rows = []
    for year in args.years.split(","):
        date_str = f"{year.strip()}{args.month_day}"
        try:
            result = predict_one(args.district, date_str, risk_model, crop_model, cache=cache)
        except ValueError as e:
            print(f"Skipping {year}: {e}")
            continue
        rows.append(result)

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    out_path = os.path.join(tm.DATA_DIR, f"compare_{args.district}_{args.month_day}.csv")
    table.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


def cmd_rank(args):
    if args.crop not in tm.CROP_REQUIREMENTS:
        raise ValueError(f"Unknown crop '{args.crop}'. Choices: {list(tm.CROP_REQUIREMENTS)}")
    if not os.path.exists(tm.RAW_CSV):
        raise FileNotFoundError(f"{tm.RAW_CSV} not found. Run train_model.py --task risk first.")

    cache = pd.read_csv(tm.RAW_CSV, parse_dates=["date"])

    rows = []
    for district in DISTRICT_COORDS:
        try:
            features, source = get_features_any(district, args.date, cache)
        except ValueError as e:
            print(f"Skipping {district}: {e}")
            continue
        rows.append({
            "district": district,
            "fit_score": round(score_crop_fit(features, args.crop), 3),
            "source": source,
            "precip_7d": round(float(features["precip_7d"]), 2),
            "temp_7d_avg": round(float(features["temp_7d_avg"]), 2),
        })

    table = pd.DataFrame(rows).sort_values("fit_score", ascending=False)
    print(f"District ranking for {args.crop} on {args.date} (0 = ideal, more negative = worse fit)")
    print(table.to_string(index=False))
    out_path = os.path.join(tm.DATA_DIR, f"rank_{args.crop.replace(' ', '_')}_{args.date}.csv")
    table.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_predict = sub.add_parser("predict", help="Predict risk + best crop for one district/date")
    p_predict.add_argument("--district", required=True)
    p_predict.add_argument("--date", required=True, help="YYYYMMDD")
    p_predict.set_defaults(func=cmd_predict)

    p_snapshot = sub.add_parser("snapshot", help="Risk + best crop for all 64 districts on one date")
    p_snapshot.add_argument("--date", required=True, help="YYYYMMDD")
    p_snapshot.set_defaults(func=cmd_snapshot)

    p_calendar = sub.add_parser("calendar", help="Most-recommended crop per district per month")
    p_calendar.set_defaults(func=cmd_calendar)

    p_outlook = sub.add_parser("outlook", help="Climatology-based risk + crop outlook for a future date")
    p_outlook.add_argument("--district", required=True)
    p_outlook.add_argument("--date", required=True, help="YYYYMMDD, any year (uses NASA climatology for that calendar day)")
    p_outlook.set_defaults(func=cmd_outlook)

    p_season = sub.add_parser("season", help="Full-year, month-by-month climatology outlook for one district (when rain, when drought)")
    p_season.add_argument("--district", required=True)
    p_season.add_argument("--year", required=True, help="Any year, e.g. 2027")
    p_season.set_defaults(func=cmd_season)

    p_map = sub.add_parser("map", help="PNG map of all 64 districts colored by risk on one date")
    p_map.add_argument("--date", required=True, help="YYYYMMDD")
    p_map.set_defaults(func=cmd_map)

    p_alert = sub.add_parser("alert", help="High-risk-district alert messages for one date (uses climatology if the date is in the future)")
    p_alert.add_argument("--date", required=True, help="YYYYMMDD")
    p_alert.set_defaults(func=cmd_alert)

    p_compare = sub.add_parser("compare", help="Compare one district's actual NASA data on the same calendar day across multiple years")
    p_compare.add_argument("--district", required=True)
    p_compare.add_argument("--years", required=True, help="Comma-separated years, e.g. 2021,2022,2023,2024,2025")
    p_compare.add_argument("--month-day", required=True, dest="month_day", help="MMDD, e.g. 0715")
    p_compare.set_defaults(func=cmd_compare)

    p_rank = sub.add_parser("rank", help="Rank all 64 districts by suitability for one crop on one date")
    p_rank.add_argument("--crop", required=True, choices=list(tm.CROP_REQUIREMENTS))
    p_rank.add_argument("--date", required=True, help="YYYYMMDD")
    p_rank.set_defaults(func=cmd_rank)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
