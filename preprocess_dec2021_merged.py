"""
preprocess_dec2021_merged.py

Output format (z-score normalized):
  frame_id, vessel_id, LON, LAT, SOG, Heading

Split (6:2:2 ratio by day):
  Train: days 1-19  → dataset/noaa_dec2021_1min/train/
  Val:   days 20-25 → dataset/noaa_dec2021_1min/val/
  Test:  days 26-31 → dataset/noaa_dec2021_1min/test/

Expects:
  data/raw/2021_12/AIS_2021_12_01.zip ... AIS_2021_12_31.zip
"""

import os
import io
import glob
import re
import json
import zipfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Configuration (matching SMCHN paper exactly)

RAW_DIR    = "data/2021_12"
OUT_BASE   = "dataset/noaa_dec2021_1min"

# SMCHN paper geographic bounds
LAT_MIN, LAT_MAX =  30.0,  35.0
LON_MIN, LON_MAX = -120.0, -115.0

# SMCHN paper filters
SOG_MIN     = 1.0    # knots
SOG_MAX     = 22.0   # knots
HDG_MIN     = 0.0
HDG_MAX     = 360.0
MIN_VESSELS = 3      # minimum vessels per timestamp (paper value)

# Resampling
RESAMPLE_FREQ = "1min"
INTERP_LIMIT  = 5     # max consecutive NaN minutes to fill
MAX_GAP_MIN   = 10    # gaps > 10 min: do NOT interpolate (split into segments)

# Train/Val/Test split by day
TRAIN_DAYS = list(range(1, 20))   # days 1-19
VAL_DAYS   = list(range(20, 26))  # days 20-25
TEST_DAYS  = list(range(26, 32))  # days 26-31

# Features to normalize
FEATURE_COLS = ["LON", "LAT", "SOG", "Heading"]


# Step 1: Load one zip file

def load_zip(zip_path: str) -> pd.DataFrame:
    """Load first CSV inside a NOAA AIS zip file."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV in {zip_path}")
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(
                io.TextIOWrapper(f, encoding="utf-8"),
                usecols=["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "Heading"],
                low_memory=False,
            )
    return df


# Step 2: Clean one day's raw data

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Apply SMCHN paper cleaning rules."""
    df = df.copy()

    # Parse timestamp
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], errors="coerce")
    df = df.dropna(subset=["BaseDateTime", "MMSI", "LAT", "LON", "SOG", "Heading"])

    # MMSI must be 9 digits
    df["MMSI"] = df["MMSI"].astype(str).str.strip()
    df = df[df["MMSI"].str.match(r"^\d{9}$")]

    # Geographic bounds
    df = df[
        (df["LAT"] >= LAT_MIN) & (df["LAT"] <= LAT_MAX) &
        (df["LON"] >= LON_MIN) & (df["LON"] <= LON_MAX)
    ]

    # SOG and Heading range (keep invalid heading=511 for now, handle per-vessel)
    df = df[
        (df["SOG"] >= 0.0) & (df["SOG"] <= SOG_MAX) &
        (df["Heading"] >= 0.0) & (df["Heading"] <= 511.0)
    ]

    # Replace invalid heading
    df = df[(df["Heading"] >= HDG_MIN) & (df["Heading"] <= HDG_MAX)]

    return df.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)

# Step 3: Per-vessel resampling (on FULL monthly trajectory)

def resample_vessel(vd: pd.DataFrame) -> pd.DataFrame:

    #Resample one vessel's FULL monthly trajectory to 1-minute intervals.
    #Handles large gaps (> MAX_GAP_MIN) by NOT interpolating across them
    #segments before and after the gap are processed independently preventing fake interpolation during port calls or data outages.

    vd = vd.sort_values("BaseDateTime").copy()
    vd["BaseDateTime"] = vd["BaseDateTime"].dt.ceil(RESAMPLE_FREQ)
    vd = vd.drop_duplicates("BaseDateTime")
    vd = vd.set_index("BaseDateTime")[["LON", "LAT", "SOG", "Heading"]]

    # Identify large gaps and split into segments
    diffs = vd.index.to_series().diff()
    gap_mask = diffs > pd.Timedelta(minutes=MAX_GAP_MIN)
    segment_ids = gap_mask.cumsum()

    segments = []
    for _, seg in vd.groupby(segment_ids):
        if len(seg) < 2:
            continue

        # Resample and interpolate LON/LAT linearly
        resampled = seg.resample(RESAMPLE_FREQ).interpolate(
            method="linear", limit=INTERP_LIMIT, limit_direction="forward"
        )

        # SOG/Heading: use average (SMCHN paper) = forward fill then interpolate
        resampled["SOG"]     = resampled["SOG"].ffill(limit=INTERP_LIMIT)
        resampled["Heading"] = resampled["Heading"].interpolate(
            method="linear", limit=INTERP_LIMIT).ffill(limit=INTERP_LIMIT).bfill(limit=INTERP_LIMIT)

        resampled = resampled.dropna(subset=["LAT", "LON", "SOG"])
        if len(resampled) >= 2:
            segments.append(resampled)

    if not segments:
        return pd.DataFrame()

    return pd.concat(segments).reset_index().rename(columns={"index": "BaseDateTime"})

# Step 4: Filter moored/anchored vessels


def keep_moving(df: pd.DataFrame) -> pd.DataFrame:
    #Keep only vessels that have SOG >= 1 knot at some point in their trajectory.
    return df.groupby("MMSI").filter(lambda x: x["SOG"].max() >= SOG_MIN)



# Step 5: Filter timestamps with < MIN_VESSELS concurrent vessels

def filter_min_vessels(df: pd.DataFrame) -> pd.DataFrame:
    #Remove timestamps where fewer than MIN_VESSELS vessels are present.
    counts = df.groupby("BaseDateTime")["MMSI"].transform("count")
    return df[counts > MIN_VESSELS].copy()



# Step 6: Assign frame_id and vessel_id, convert to frame format

def to_frame_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert from (MMSI, BaseDateTime, LON, LAT, SOG, Heading)
    to    (frame_id, vessel_id, LON, LAT, SOG, Heading)
    where frame_id is a sequential integer per unique timestamp.
    """
    timestamps = sorted(df["BaseDateTime"].unique())
    ts_to_id = {ts: i for i, ts in enumerate(timestamps)}

    mmsis = sorted(df["MMSI"].unique())
    mmsi_to_id = {m: i for i, m in enumerate(mmsis)}

    df = df.copy()
    df["frame_id"]  = df["BaseDateTime"].map(ts_to_id)
    df["vessel_id"] = df["MMSI"].map(mmsi_to_id)
    df["day"]       = df["BaseDateTime"].dt.day   # keep for splitting

    return df[["frame_id", "vessel_id", "LON", "LAT", "SOG", "Heading", "day"]]


# Step 7: Z-score normalization

def compute_stats(df: pd.DataFrame, cols: list) -> dict:
    """Compute mean and std for each feature column."""
    stats = {}
    for col in cols:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std":  float(df[col].std()),
        }
        if stats[col]["std"] == 0 or not np.isfinite(stats[col]["std"]):
            stats[col]["std"] = 1.0
    return stats


def apply_zscore(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Apply z-score normalization using precomputed statistics."""
    df = df.copy()
    for col, s in stats.items():
        df[col] = (df[col] - s["mean"]) / s["std"]
    return df

# Main

def main():
    zip_files = sorted(glob.glob(os.path.join(RAW_DIR, "AIS_2021_12_*.zip")))
    if not zip_files:
        print(f"ERROR: No zip files found in {RAW_DIR}")
        return

    print(f"Found {len(zip_files)} zip files\n")

    # STEP A: Load and clean ALL days together
    print("=" * 60)
    print("STEP A: Loading and cleaning all 31 days...")
    print("=" * 60)

    all_frames = []
    for i, zp in enumerate(zip_files, 1):
        day = int(re.search(r"AIS_2021_12_(\d{2})\.zip", os.path.basename(zp)).group(1))
        print(f"  [{i:02d}/31] Day {day:02d} loading...", end="\r")
        try:
            raw  = load_zip(zp)
            clean = clean_raw(raw)
            clean["_day"] = day
            all_frames.append(clean[["MMSI", "BaseDateTime", "LON", "LAT", "SOG", "Heading", "_day"]])
            del raw, clean
        except Exception as e:
            print(f"\n  ERROR day {day}: {e}")

    print(f"\n  Merging all days...")
    df_all = pd.concat(all_frames, ignore_index=True)
    del all_frames

    print(f"  Total records after cleaning: {len(df_all):,}")
    print(f"  Unique vessels (raw):          {df_all['MMSI'].nunique():,}")

    #  STEP B: Per-vessel resampling on FULL monthly trajectory
    print("\n" + "=" * 60)
    print("STEP B: Resampling each vessel's full monthly trajectory...")
    print("=" * 60)

    vessels = df_all["MMSI"].unique()
    print(f"  Processing {len(vessels):,} vessels...")

    resampled_list = []
    for i, mmsi in enumerate(vessels):
        if i % 100 == 0:
            print(f"  Vessel {i:,}/{len(vessels):,}...", end="\r")

        vd = df_all[df_all["MMSI"] == mmsi].copy()
        result = resample_vessel(vd)
        if result.empty:
            continue
        result["MMSI"] = mmsi
        resampled_list.append(result)

    print(f"\n  Resampled {len(resampled_list):,} vessels successfully")

    df_resampled = pd.concat(resampled_list, ignore_index=True)
    del resampled_list, df_all

    #STEP C: Remove moored vessels
    print("\n" + "=" * 60)
    print("STEP C: Removing moored/anchored vessels (SOG < 1 knot always)...")
    print("=" * 60)

    before = df_resampled["MMSI"].nunique()
    df_resampled = keep_moving(df_resampled)
    after = df_resampled["MMSI"].nunique()
    print(f"  Vessels: {before:,} → {after:,} (removed {before-after:,} moored)")

    # STEP D: Filter timestamps with < MIN_VESSELS vessels
    print("\n" + "=" * 60)
    print(f"STEP D: Filtering timestamps with < {MIN_VESSELS} concurrent vessels...")
    print("=" * 60)

    before = len(df_resampled)
    df_resampled = filter_min_vessels(df_resampled)
    after = len(df_resampled)
    print(f"  Rows: {before:,} → {after:,}")
    print(f"  Final unique vessels: {df_resampled['MMSI'].nunique():,}")

    # Add day column
    df_resampled["day"] = df_resampled["BaseDateTime"].dt.day

    # STEP E: Compute z-score stats from TRAIN data only
    print("\n" + "=" * 60)
    print("STEP E: Computing z-score statistics from TRAIN data only...")
    print("=" * 60)

    df_train_raw = df_resampled[df_resampled["day"].isin(TRAIN_DAYS)]
    stats = compute_stats(df_train_raw, FEATURE_COLS)

    print("  Global statistics (from train days 1-19):")
    for col, s in stats.items():
        print(f"    {col}: mean={s['mean']:.6f}, std={s['std']:.6f}")

    # STEP F: Apply z-score and convert to frame format
    print("\n" + "=" * 60)
    print("STEP F: Applying z-score normalization and converting to frame format...")
    print("=" * 60)

    df_normalized = apply_zscore(df_resampled, stats)

    # Convert to frame format (frame_id per timestamp, vessel_id per MMSI)
    timestamps = sorted(df_normalized["BaseDateTime"].unique())
    mmsis      = sorted(df_normalized["MMSI"].unique())
    ts_to_id   = {ts: i for i, ts in enumerate(timestamps)}
    mmsi_to_id = {m: i for i, m in enumerate(mmsis)}

    df_normalized["frame_id"]  = df_normalized["BaseDateTime"].map(ts_to_id)
    df_normalized["vessel_id"] = df_normalized["MMSI"].map(mmsi_to_id)

    #STEP G: Split by day and save
    print("\n" + "=" * 60)
    print("STEP G: Splitting by day and saving CSVs...")
    print("=" * 60)

    split_map = {}
    for day in TRAIN_DAYS: split_map[day] = "train"
    for day in VAL_DAYS:   split_map[day] = "val"
    for day in TEST_DAYS:  split_map[day] = "test"

    out_cols = ["frame_id", "vessel_id", "LON", "LAT", "SOG", "Heading"]

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUT_BASE, split), exist_ok=True)

    total_rows = {"train": 0, "val": 0, "test": 0}

    for day in range(1, 32):
        split = split_map.get(day)
        if split is None:
            continue

        day_df = df_normalized[df_normalized["day"] == day][out_cols]
        if day_df.empty:
            print(f"  Day {day:02d}: empty, skipping")
            continue

        out_path = os.path.join(OUT_BASE, split, f"day_2021_12_{day:02d}.csv")
        day_df.to_csv(out_path, index=False)
        total_rows[split] += len(day_df)
        print(f"  Day {day:02d} ({split:5s}): {len(day_df):>8,} rows → {out_path}")

    # STEP H: Save global stats
    stats_path = os.path.join(OUT_BASE, "global_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"  Output:         {OUT_BASE}/")
    print(f"  Train rows:     {total_rows['train']:,}")
    print(f"  Val rows:       {total_rows['val']:,}")
    print(f"  Test rows:      {total_rows['test']:,}")
    print(f"  Unique vessels: {df_normalized['MMSI'].nunique():,}")
    print(f"  Global stats:   {stats_path}")
    print(f"\n  LAT range: {df_resampled['LAT'].min():.3f}° → {df_resampled['LAT'].max():.3f}°")
    print(f"  LON range: {df_resampled['LON'].min():.3f}° → {df_resampled['LON'].max():.3f}°")
    print(f"\nNext step: implement DataLoader and models")
    print("=" * 60)


if __name__ == "__main__":
    main()
