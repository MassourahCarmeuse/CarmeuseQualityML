import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from openpyxl import load_workbook


# =========================================================
# 1. FIND PROJECT FOLDER
# =========================================================

project_folder = Path(__file__).resolve().parent.parent

saved_model_folder = (
    project_folder / "Saved_Models"
)

data_folder = (
    project_folder / "Data"
)


# =========================================================
# 2. LOAD SAVED MODEL
# =========================================================

model = joblib.load(
    saved_model_folder / "K1_CO2_XGBoost_Deployment.pkl"
)


# =========================================================
# 3. LOAD FEATURE LIST
# =========================================================

features = joblib.load(
    saved_model_folder / "K1_CO2_Features.pkl"
)


# =========================================================
# 4. LOAD MODEL CONFIGURATION
# =========================================================

config = joblib.load(
    saved_model_folder / "K1_CO2_Config.pkl"
)


print("\n================================")
print("K1 CO2 PREDICTOR LOADED")
print("================================")

print(
    "Model type:",
    config["Model_Type"]
)

print(
    "Prediction delay:",
    config["Delay_Hours"],
    "hours"
)

print(
    "Number of required features:",
    len(features)
)

print(
    "Test MAE:",
    round(config["Final_Test_MAE"], 4)
)

print(
    "Test RMSE:",
    round(config["Final_Test_RMSE"], 4)
)

print(
    "Test R2:",
    round(config["Final_Test_R2"], 4)
)


# =========================================================
# 5. LOAD EXCEL PREDICTION CALCULATOR
# =========================================================

input_file = (
    data_folder
    / "Kiln_Live_Input.xlsm"
)

live_data = pd.read_excel(
    input_file,
    sheet_name="Prediction Calculator",
    engine="openpyxl"
)


print("\n================================")
print("EXCEL CALCULATOR LOADED")
print("================================")

print("\nColumns found:")
print(
    live_data.columns.tolist()
)

print("\nExcel input table:")
print(
    live_data.to_string(index=False)
)


# =========================================================
# 7. READ THE EXCEL INPUT TABLE
# =========================================================

# Based on your Excel layout:
#
# Column B = variable names
# Column C = entered values
#
# Pandas currently reads these as:
# "Input"      = variable names
# "Unnamed: 2" = values

label_column = "Input"
value_column = "Unnamed: 2"


raw_inputs = {}

for _, row in live_data.iterrows():

    label = row[label_column]
    value = row[value_column]

    # Ignore empty rows
    if pd.isna(label):
        continue

    label = str(label).strip()

    if label == "":
        continue

    raw_inputs[label] = value


# =========================================================
# 8. DISPLAY WHAT PYTHON READ
# =========================================================

print("\n================================")
print("INPUTS READ FROM EXCEL")
print("================================")

for name, value in raw_inputs.items():

    print(
        f"{name}: {value}"
    )


# =========================================================
# 9. REQUIRED KILN INPUTS
# =========================================================

required_inputs = [
    "Gas_SCFM",
    "Rate",
    "O2_pct",
    "Product_Temp_F",
    "Feed_End_Temp_F",
    "South_Flue_Temp_F",
    "North_Flue_Temp_F",
    "Primary_Air_Flow_SCFM",
    "Cooler_Air_SCFM",
    "Kiln_Amps",
    "ID_Fan_Amps"
]


# =========================================================
# 10. CHECK THE REQUIRED INPUTS
# =========================================================

missing_inputs = []

for variable in required_inputs:

    if variable not in raw_inputs:

        missing_inputs.append(
            variable
        )


if missing_inputs:

    print("\n================================")
    print("MISSING INPUTS")
    print("================================")

    for variable in missing_inputs:

        print(
            "-",
            variable
        )

else:

    print("\n================================")
    print("ALL INPUTS FOUND")
    print("================================")

    print(
        "All 11 required current "
        "kiln variables were found."
    )


# =========================================================
# 11. CONVERT INPUTS TO NUMBERS
# =========================================================

current_values = {}

for variable in required_inputs:

    if variable in raw_inputs:

        current_values[variable] = pd.to_numeric(
            raw_inputs[variable],
            errors="coerce"
        )


print("\n================================")
print("CURRENT KILN VALUES")
print("================================")

for name, value in current_values.items():

    print(
        f"{name}: {value}"
    )
# =========================================================
# 12. READ RECENT HISTORY SHEET
# =========================================================

history_data = pd.read_excel(
    input_file,
    sheet_name="Recent History",
    engine="openpyxl"
)

# Remove accidental spaces from column headers
history_data.columns = (
    history_data.columns
    .astype(str)
    .str.strip()
)

print("\n================================")
print("RECENT HISTORY LOADED")
print("================================")

print(
    "History rows found:",
    len(history_data)
)

print(
    "History columns:",
    history_data.columns.tolist()
)


# =========================================================
# 13. VERIFY RECENT HISTORY COLUMNS
# =========================================================

history_columns = [
    "Timestamp",
    "Gas_SCFM",
    "Rate",
    "O2_pct",
    "Product_Temp_F",
    "Feed_End_Temp_F",
    "South_Flue_Temp_F",
    "North_Flue_Temp_F",
    "Primary_Air_Flow_SCFM",
    "Cooler_Air_SCFM",
    "Kiln_Amps",
    "ID_Fan_Amps"
]

missing_history_columns = [
    column
    for column in history_columns
    if column not in history_data.columns
]

if missing_history_columns:

    raise ValueError(
        "Recent History is missing these columns: "
        + ", ".join(missing_history_columns)
    )


# =========================================================
# 14. CLEAN RECENT HISTORY
# =========================================================

history_data["Timestamp"] = pd.to_datetime(
    history_data["Timestamp"],
    errors="coerce"
)

for variable in required_inputs:

    history_data[variable] = pd.to_numeric(
        history_data[variable],
        errors="coerce"
    )

history_data = (
    history_data
    .dropna(subset=["Timestamp"])
    .sort_values("Timestamp")
    .reset_index(drop=True)
)

if history_data.empty:

    raise ValueError(
        "Recent History contains no valid timestamped rows."
    )

print("\nHistory date range:")

print(
    history_data["Timestamp"].min(),
    "to",
    history_data["Timestamp"].max()
)


# =========================================================
# 15. CREATE THE CURRENT PROCESS ROW
# =========================================================

current_timestamp = pd.to_datetime(
    raw_inputs["Timestamp"],
    errors="coerce"
)

if pd.isna(current_timestamp):

    raise ValueError(
        "The Prediction Calculator timestamp is invalid."
    )

current_row = {
    "Timestamp": current_timestamp
}

for variable in required_inputs:

    current_row[variable] = current_values[variable]


current_row = pd.DataFrame(
    [current_row]
)


# =========================================================
# 16. COMBINE HISTORY WITH CURRENT VALUES
# =========================================================

combined_data = pd.concat(
    [
        history_data[history_columns],
        current_row[history_columns]
    ],
    ignore_index=True
)

combined_data = (
    combined_data
    .sort_values("Timestamp")
    .drop_duplicates(
        subset=["Timestamp"],
        keep="last"
    )
    .reset_index(drop=True)
)

print("\n================================")
print("COMBINED PROCESS DATA")
print("================================")

print(
    combined_data.to_string(index=False)
)


# =========================================================
# 17. CHECK THAT HISTORY PRECEDES CURRENT TIME
# =========================================================

previous_rows = combined_data[
    combined_data["Timestamp"] < current_timestamp
]

if len(previous_rows) < 4:

    raise ValueError(
        "At least four valid historical readings before "
        "the current timestamp are required."
    )

history_duration = (
    current_timestamp
    - previous_rows["Timestamp"].min()
)

if history_duration < pd.Timedelta(hours=4):

    raise ValueError(
        "Recent History must cover at least four hours "
        "before the current timestamp."
    )


# =========================================================
# 18. CREATE 6-MINUTE PROCESS GRID
# =========================================================
#
# This matches the feature engineering used by kiln_model.py.
# The original live/history values are interpolated onto the
# same 6-minute grid used during model training.
# =========================================================

process_data = (
    combined_data[
        ["Timestamp"] + required_inputs
    ]
    .copy()
)

process_data["Timestamp"] = (
    process_data["Timestamp"].dt.round("h")
)

process_data = (
    process_data
    .drop_duplicates(
        subset=["Timestamp"],
        keep="last"
    )
    .set_index("Timestamp")
    .sort_index()
)

six_minute_index = pd.date_range(
    start=process_data.index.min(),
    end=current_timestamp.round("h"),
    freq="6min"
)

process_grid = process_data.reindex(
    six_minute_index
)

process_grid[required_inputs] = (
    process_grid[required_inputs]
    .interpolate(
        method="time",
        limit_area="inside"
    )
)

process_grid.index.name = "Feature_Timestamp"


# =========================================================
# 19. CALCULATE THE 59 FEATURES
# =========================================================

feature_grid = pd.DataFrame(
    index=process_grid.index
)

for variable in required_inputs:

    # Current value
    feature_grid[variable] = (
        process_grid[variable]
    )

    # One-hour average
    feature_grid[
        f"{variable}_avg_1hr"
    ] = (
        process_grid[variable]
        .rolling(
            "1h",
            min_periods=2
        )
        .mean()
    )

    # Two-hour average
    feature_grid[
        f"{variable}_avg_2hr"
    ] = (
        process_grid[variable]
        .rolling(
            "2h",
            min_periods=2
        )
        .mean()
    )

    # Four-hour average
    feature_grid[
        f"{variable}_avg_4hr"
    ] = (
        process_grid[variable]
        .rolling(
            "4h",
            min_periods=2
        )
        .mean()
    )

    # Change compared with one hour earlier
    # Ten 6-minute rows equal one hour
    feature_grid[
        f"{variable}_change_1hr"
    ] = (
        process_grid[variable]
        - process_grid[variable].shift(10)
    )


# =========================================================
# 20. CALCULATE ENGINEERED FEATURES
# =========================================================

feature_grid["Gas_per_Rate"] = (
    feature_grid["Gas_SCFM"]
    / feature_grid["Rate"].replace(
        0,
        np.nan
    )
)

feature_grid["Primary_Air_to_Gas"] = (
    feature_grid["Primary_Air_Flow_SCFM"]
    / feature_grid["Gas_SCFM"].replace(
        0,
        np.nan
    )
)

feature_grid["Average_Flue_Temp"] = (
    (
        feature_grid["North_Flue_Temp_F"]
        + feature_grid["South_Flue_Temp_F"]
    )
    / 2
)

feature_grid["Flue_Temp_Difference"] = (
    feature_grid["North_Flue_Temp_F"]
    - feature_grid["South_Flue_Temp_F"]
)


# =========================================================
# 21. SELECT THE NEWEST FEATURE ROW
# =========================================================

latest_feature_row = (
    feature_grid
    .iloc[[-1]]
    .copy()
)

missing_model_features = [
    feature
    for feature in features
    if feature not in latest_feature_row.columns
]

if missing_model_features:

    raise ValueError(
        "These saved model features were not created: "
        + ", ".join(missing_model_features)
    )


# Arrange columns in the exact order used during training
prediction_inputs = latest_feature_row[
    features
].copy()


# =========================================================
# 22. CHECK FOR INVALID FEATURE VALUES
# =========================================================

invalid_features = prediction_inputs.columns[
    prediction_inputs.isna().any()
].tolist()

if invalid_features:

    raise ValueError(
        "The following prediction features are blank or "
        "could not be calculated: "
        + ", ".join(invalid_features)
    )

if not np.isfinite(
    prediction_inputs.to_numpy(
        dtype=float
    )
).all():

    raise ValueError(
        "One or more prediction features contain an "
        "infinite or invalid value. Check for zero Rate "
        "or zero Gas_SCFM."
    )


print("\n================================")
print("MODEL FEATURES CREATED")
print("================================")

print(
    "Features created:",
    prediction_inputs.shape[1]
)

print(
    "Features expected:",
    len(features)
)


# =========================================================
# 23. PREDICT CO2
# =========================================================

predicted_co2 = model.predict(
    prediction_inputs
)[0]

# Prevent a physically impossible negative display
predicted_co2 = max(
    float(predicted_co2),
    0.0
)


# =========================================================
# 24. CALCULATE EXPECTED QUALITY TIME
# =========================================================

prediction_delay = float(
    config["Delay_Hours"]
)

expected_quality_time = (
    current_timestamp
    + pd.to_timedelta(
        prediction_delay,
        unit="h"
    )
)


# =========================================================
# 25. DISPLAY THE PREDICTION
# =========================================================

print("\n================================")
print("K1 CO2 PREDICTION")
print("================================")

print(
    "Current operating time:",
    current_timestamp
)

print(
    "Expected quality time:",
    expected_quality_time
)

print(
    "Approximate prediction horizon:",
    prediction_delay,
    "hours"
)

print(
    "Predicted CO2:",
    f"{predicted_co2:.3f}%"
)

print(
    "Development test MAE:",
    f"{config['Final_Test_MAE']:.3f} percentage points"
)

print("\nImportant:")

print(
    "This is a development prediction. "
    "The model still had significant errors during "
    "some high-CO2 events."
)
# =========================================================
# 26. WRITE PREDICTION BACK TO EXCEL
# =========================================================

workbook = load_workbook(
    input_file,
    keep_vba=True
)

sheet = workbook[
    "Prediction Calculator"
]

# Output labels
sheet["E2"] = "Prediction Output"
sheet["E3"] = "Predicted CO2"
sheet["E4"] = "Current Time"
sheet["E5"] = "Expected Quality Time"
sheet["E6"] = "Prediction Horizon"

# Output values
sheet["F3"] = predicted_co2/100
sheet["F4"] = current_timestamp
sheet["F5"] = expected_quality_time
sheet["F6"] = prediction_delay

# Formatting
sheet["F3"].number_format = "0.000%"
sheet["F4"].number_format = "m/d/yyyy h:mm"
sheet["F5"].number_format = "m/d/yyyy h:mm"
sheet["F6"].number_format = '0.0 "hours"'

workbook.save(
    input_file
)

print("\n================================")
print("EXCEL OUTPUT UPDATED")
print("================================")

print(
    "Prediction was written to:",
    input_file
)