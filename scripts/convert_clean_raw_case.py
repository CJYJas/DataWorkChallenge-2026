import argparse
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = str(REPO_ROOT / "raw_data")
CASES_OUTPUT_DIR = str(REPO_ROOT / "data" / "cases")
RAW_DATA_OUTPUT_DIR = str(REPO_ROOT / "data" / "raw_data")

GRID_COL_MAP = {
    "I Index": "I",
    "J Index": "J",
    "K Index": "K",
    "X Coordinate": "X",
    "Y Coordinate": "Y",
    "Z Coordinate": "Z"
}

def standardize_case_columns(columns):
    new_cols = []
    for col in columns:
        col_clean = col.strip()
        if col_clean in GRID_COL_MAP:
            new_cols.append(GRID_COL_MAP[col_clean])
        elif "(" in col_clean and ")" in col_clean:
            # Extract standard simulation acronym in parentheses
            acronym = col_clean.split("(")[1].split(")")[0].strip()
            new_cols.append(acronym)
        else:
            new_cols.append(col_clean)
    return new_cols

def process_case_sheet(excel_file, sheet_name, out_path):
    print(f"  Reading sheet: {sheet_name} ...", end=" ", flush=True)
    t0 = time.time()

    df = pd.read_excel(
        excel_file,
        sheet_name=sheet_name,
        engine="calamine"
    )

    original_rows = len(df)

    # Find columns before standardizing their names
    def find_col(acronym):
        matches = [
            c for c in df.columns
            if acronym in str(c).upper()
        ]
        if not matches:
            raise ValueError(
                f"{acronym} column not found in {sheet_name}"
            )
        return matches[0]

    porv_col = find_col("PORV")

    # Required reservoir properties for this cleaning rule
    required_acronyms = [
        "PORO", "PERMX", "PERMY", "TRANX", "TRANY"
    ]
    required_cols = {
        acronym: find_col(acronym)
        for acronym in required_acronyms
    }

    # Convert required properties to numeric
    numeric_cols = [porv_col] + list(required_cols.values())
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Recognized invalid sentinel values
    sentinel_values = [-999.25, -999.0, -999]

    porv = df[porv_col]
    properties = df[list(required_cols.values())]

    # A required property is invalid if it is a sentinel or missing
    invalid_property = (
        properties.isin(sentinel_values) | properties.isna()
    )

    # Inactive cell with no usable required property values
    inactive_no_properties = (
        porv.eq(0) & invalid_property.all(axis=1)
    )

    # Active cell missing at least one required property
    active_missing_property = (
        porv.gt(0) & invalid_property.any(axis=1)
    )

    # Invalid/missing PORV: investigate rather than silently keeping
    invalid_porv = (
        porv.isin(sentinel_values) | porv.isna() | porv.lt(0)
    )

    drop_mask = (
        inactive_no_properties
        | active_missing_property
        | invalid_porv
    )

    removed_inactive = int(inactive_no_properties.sum())
    removed_active = int(active_missing_property.sum())
    removed_porv = int(
        (invalid_porv & ~inactive_no_properties
         & ~active_missing_property).sum()
    )

    df = df.loc[~drop_mask].copy()

    # Standardize column names
    df.columns = standardize_case_columns(df.columns)

    # Convert remaining data to float32
    df = df.astype(np.float32)

    # Save Parquet
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(
        out_path,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )

    print(
        f"Saved {os.path.basename(out_path)} | "
        f"Original: {original_rows} | "
        f"Inactive removed: {removed_inactive} | "
        f"Active missing properties removed: {removed_active} | "
        f"Invalid PORV removed: {removed_porv} | "
        f"Remaining: {len(df)} | "
        f"Time: {time.time() - t0:.2f}s"
    )

def process_all_cases(overwrite=False):
    os.makedirs(CASES_OUTPUT_DIR, exist_ok=True)

    case_workbooks = [
        "03 Train Cases 1.xlsx",
        "04 Train Cases 2.xlsx",
        "05 Train Cases 3.xlsx",
        "06 Train Cases 4.xlsx",
        "07  Validation Cases.xlsx",
        "08 Test Cases.xlsx"
    ]

    total_cases = 0
    skipped_cases = 0
    processed_cases = 0

    for wb_name in case_workbooks:
        wb_path = os.path.join(RAW_DATA_DIR, wb_name)

        if not os.path.exists(wb_path):
            print(f"Warning: Workbook not found: {wb_path}")
            continue

        print(f"\nProcessing workbook: {wb_name}")
        excel = pd.ExcelFile(wb_path, engine="calamine")

        for sheet_name in excel.sheet_names:
            total_cases += 1

            clean_name = sheet_name.strip().replace(" ", "_")
            out_path = os.path.join(
                CASES_OUTPUT_DIR,
                f"{clean_name}.parquet"
            )

            if os.path.exists(out_path) and not overwrite:
                print(f"  Skipping {sheet_name}: file already exists")
                skipped_cases += 1
                continue

            process_case_sheet(excel, sheet_name, out_path)
            processed_cases += 1

    print("\n--- Case Processing Complete ---")
    print(
        f"Total cases: {total_cases}, "
        f"Processed: {processed_cases}, "
        f"Skipped: {skipped_cases}"
    )

def process_other_raw_data():
    os.makedirs(RAW_DATA_OUTPUT_DIR, exist_ok=True)
    
    # 1. 01 Introduction.xlsx
    intro_path = os.path.join(RAW_DATA_DIR, "01 Introduction.xlsx")
    if os.path.exists(intro_path):
        out_sub = os.path.join(RAW_DATA_OUTPUT_DIR, "01_Introduction")
        os.makedirs(out_sub, exist_ok=True)
        excel = pd.ExcelFile(intro_path, engine="calamine")
        for s in excel.sheet_names:
            out_file = os.path.join(out_sub, f"{s.strip().replace(' ', '_')}.parquet")
            if os.path.exists(out_file):
                print(f"Skipping 01 Introduction - {s} (already exists)")
                continue
            df = pd.read_excel(excel, sheet_name=s)
            df.columns = [str(c).strip() for c in df.columns]
            # Convert string columns properly
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str)
            df.to_parquet(out_file, index=False, engine="pyarrow")
            print(f"Saved 01 Introduction: {s} -> {out_file}")

    # 2. 02 Field Production.xlsx
    field_path = os.path.join(RAW_DATA_DIR, "02 Field Production.xlsx")
    if os.path.exists(field_path):
        out_sub = os.path.join(RAW_DATA_OUTPUT_DIR, "02_Field_Production")
        os.makedirs(out_sub, exist_ok=True)
        excel = pd.ExcelFile(field_path, engine="calamine")
        for s in excel.sheet_names:
            if s == "Plots":
                continue
            out_file = os.path.join(out_sub, f"{s.strip().replace(' ', '_')}.parquet")
            if os.path.exists(out_file):
                print(f"Skipping 02 Field Production - {s} (already exists)")
                continue
            
            # Check if this sheet has multi-row headers (rate and cumulative sheets)
            if "cumulative" in s.lower() or "rate" in s.lower():
                df_raw = pd.read_excel(excel, sheet_name=s, header=None)
                cols = [str(c).strip() for c in df_raw.iloc[1].tolist()]
                df = df_raw.iloc[3:].copy()
                df.columns = cols
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                for c in cols:
                    if c != "Date":
                        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
            else:
                df = pd.read_excel(excel, sheet_name=s)
                df.columns = [str(c).strip() for c in df.columns]
                for c in df.columns:
                    if df[c].dtype == object and not pd.api.types.is_datetime64_any_dtype(df[c]):
                        # Check if all numeric
                        try:
                            df[c] = pd.to_numeric(df[c]).astype(np.float32)
                        except Exception:
                            df[c] = df[c].astype(str)
            
            df.to_parquet(out_file, index=False, engine="pyarrow")
            print(f"Saved 02 Field Production: {s} -> {out_file}")

    # 3. 09 Template Deliverable.xlsx
    template_path = os.path.join(RAW_DATA_DIR, "09 Template Deliverable.xlsx")
    if os.path.exists(template_path):
        out_sub = os.path.join(RAW_DATA_OUTPUT_DIR, "09_Template_Deliverable")
        os.makedirs(out_sub, exist_ok=True)
        excel = pd.ExcelFile(template_path, engine="calamine")
        for s in excel.sheet_names:
            out_file = os.path.join(out_sub, f"{s.strip().replace(' ', '_')}.parquet")
            if os.path.exists(out_file):
                print(f"Skipping 09 Template Deliverable - {s} (already exists)")
                continue
            df = pd.read_excel(excel, sheet_name=s)
            df.columns = [str(c).strip() for c in df.columns]
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
            for c in df.columns:
                if c != "Date":
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
            df.to_parquet(out_file, index=False, engine="pyarrow")
            print(f"Saved 09 Template Deliverable: {s} -> {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional Excel-to-Parquet cleaning. Not required for pipeline.py.")
    parser.add_argument("--raw-data", default=RAW_DATA_DIR, help="Folder containing the original .xlsx workbooks")
    parser.add_argument("--cases-out", default=CASES_OUTPUT_DIR)
    parser.add_argument("--raw-out", default=RAW_DATA_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    RAW_DATA_DIR = args.raw_data
    CASES_OUTPUT_DIR = args.cases_out
    RAW_DATA_OUTPUT_DIR = args.raw_out
    t_start = time.time()
    print("=== Processing 100 Case Workbooks ===")
    print("Raw data:", RAW_DATA_DIR)
    process_all_cases(overwrite=args.overwrite)
    print("\n=== Processing Other Raw Data Files ===")
    process_other_raw_data()
    t_end = time.time()
    print(f"\nAll tasks completed successfully in {t_end - t_start:.2f} seconds.")
