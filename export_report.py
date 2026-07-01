"""
export_report.py
-----------------
Combines all daily attendance CSV files from attendance/ into a single
summary report (CSV) showing each person's attendance count and dates.

Usage:
    python export_report.py
"""

import os
import pandas as pd

ATTENDANCE_DIR = "attendance"
OUTPUT_FILE = "attendance_summary.csv"


def build_summary():
    if not os.path.isdir(ATTENDANCE_DIR):
        print(f"No '{ATTENDANCE_DIR}' folder found. Run main.py first.")
        return

    csv_files = [f for f in os.listdir(ATTENDANCE_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("No attendance records found yet.")
        return

    all_records = []
    for filename in csv_files:
        path = os.path.join(ATTENDANCE_DIR, filename)
        df = pd.read_csv(path)
        all_records.append(df)

    combined = pd.concat(all_records, ignore_index=True)

    # Summary: attendance count + list of dates per person
    summary = combined.groupby("Name").agg(
        Days_Present=("Date", "nunique"),
        Dates=("Date", lambda x: ", ".join(sorted(set(x))))
    ).reset_index()

    combined.to_csv("attendance_full_log.csv", index=False)
    summary.to_csv(OUTPUT_FILE, index=False)

    print(f"Full log saved to attendance_full_log.csv ({len(combined)} entries)")
    print(f"Summary saved to {OUTPUT_FILE}:\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    build_summary()
