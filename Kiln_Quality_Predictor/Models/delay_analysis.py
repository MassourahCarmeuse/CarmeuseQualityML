import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# 1. PROJECT PATHS
# =========================================================

project_folder = Path(__file__).resolve().parent.parent

data_folder = project_folder / "Data"

saved_model_folder = (
    project_folder / "Saved_Models"
)

saved_model_folder.mkdir(
    parents=True,
    exist_ok=True
)

input_file = (
    data_folder
    / "Kiln_Quality_Model_Data_Extract.xlsx"
)

output_file = (
    data_folder
    / "K1_CO2_CurrentOnly_Results.xlsx"
)


# =========================================================
# 2. MODEL SETTINGS
# =========================================================
#
# Because the original source data is hourly, this first
# comparison uses a four-row/four-hour future target.
#
# Do not claim this is an exact physical residence time.
# =========================================================

prediction_delay_hours = 4

training_fraction = 0.60
validation_fraction = 0.20


# =========================================================
# 3. PROVISIONAL K1 STATUS THRESHOLDS
# =========================================================
#
# IMPORTANT:
#
# These are provisional development thresholds based on the
# ranges observed during the earlier analysis.
#
# Confirm them with K1 operations before production use.
#
# Current provisional interpretation:
#
# Shutdown:
#     Rate approximately zero or Gas approximately zero
#
# Startup:
#     Very low feed rate with gas on
#
# Reduced Rate:
#     Feed rate below normal production
#
# Steady:
#     Normal-looking production rate
#
# Upset Recovery:
#     This cannot be identified reliably from current values
#     alone. It should eventually come from an operator input,
#     log, or confirmed process-state rule.
# =========================================================

SHUTDOWN_RATE_MAX = 0.5
SHUTDOWN_GAS_MAX = 100.0

STARTUP_RATE_MAX = 10.0
REDUCED_RATE_MAX = 20.0


# =========================================================
# 4. LOAD K1 DATA
# =========================================================

data = pd.read_excel(
    input_file,
    sheet_name="K1 Extract",
    engine="openpyxl"
)

print("\n========================================")
print("K1 DATA LOADED")
print("========================================")

print("Rows:", len(data))
print("Columns:", len(data.columns))


# =========================================================
# 5. CLEAN TIMESTAMPS
# =========================================================

data["Timestamp"] = pd.to_datetime(
    data["Timestamp"],
    errors="coerce"
)

data = (
    data
    .dropna(subset=["Timestamp"])
    .sort_values("Timestamp")
    .reset_index(drop=True)
)


# Round tiny millisecond differences such as:
# 10:59:59.999 -> 11:00:00

data["Timestamp"] = (
    data["Timestamp"].dt.round("h")
)


# Remove duplicate timestamps if any exist

data = (
    data
    .drop_duplicates(
        subset=["Timestamp"],
        keep="first"
    )
    .sort_values("Timestamp")
    .reset_index(drop=True)
)


# =========================================================
# 6. DEFINE RAW VARIABLES
# =========================================================

raw_process_variables = [
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

required_columns = (
    ["Timestamp", "CO2_pct"]
    + raw_process_variables
)

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:

    raise ValueError(
        "K1 Extract is missing these columns: "
        + ", ".join(missing_columns)
    )


# =========================================================
# 7. CONVERT VARIABLES TO NUMERIC
# =========================================================

numeric_columns = (
    raw_process_variables
    + ["CO2_pct"]
)

for column in numeric_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# =========================================================
# 8. CORRECT CONFIRMED DATA-ENTRY ERROR
# =========================================================

bad_timestamp = pd.Timestamp(
    "2026-07-23 09:00:00"
)

correction_mask = (
    data["Timestamp"] == bad_timestamp
)

data.loc[
    correction_mask,
    "CO2_pct"
] = 2.91


print("\n========================================")
print("CONFIRMED CO2 CORRECTION")
print("========================================")

print(
    data.loc[
        correction_mask,
        ["Timestamp", "CO2_pct"]
    ]
)


# =========================================================
# 9. CREATE OPERATING STATUS
# =========================================================

def classify_operating_status(row):

    gas = row["Gas_SCFM"]
    rate = row["Rate"]

    if pd.isna(gas) or pd.isna(rate):

        return "Unknown"

    if (
        rate <= SHUTDOWN_RATE_MAX
        or gas <= SHUTDOWN_GAS_MAX
    ):

        return "Shutdown"

    if rate <= STARTUP_RATE_MAX:

        return "Startup"

    if rate <= REDUCED_RATE_MAX:

        return "Reduced_Rate"

    return "Steady"


data["Operating_Status"] = data.apply(
    classify_operating_status,
    axis=1
)


print("\n========================================")
print("OPERATING STATUS COUNTS")
print("========================================")

print(
    data["Operating_Status"].value_counts(
        dropna=False
    )
)


# =========================================================
# 10. CREATE CURRENT-STATE ENGINEERED FEATURES
# =========================================================

data["Gas_per_Rate"] = (
    data["Gas_SCFM"]
    / data["Rate"].replace(
        0,
        np.nan
    )
)

data["Primary_Air_to_Gas"] = (
    data["Primary_Air_Flow_SCFM"]
    / data["Gas_SCFM"].replace(
        0,
        np.nan
    )
)

data["Average_Flue_Temp"] = (
    (
        data["North_Flue_Temp_F"]
        + data["South_Flue_Temp_F"]
    )
    / 2
)

data["Flue_Temp_Difference"] = (
    data["North_Flue_Temp_F"]
    - data["South_Flue_Temp_F"]
)


# =========================================================
# 11. ONE-HOT ENCODE OPERATING STATUS
# =========================================================

expected_statuses = [
    "Steady",
    "Reduced_Rate",
    "Startup",
    "Shutdown",
    "Upset_Recovery"
]

for status in expected_statuses:

    status_column = f"Status_{status}"

    data[status_column] = (
        data["Operating_Status"] == status
    ).astype(int)


# Upset_Recovery remains zero until a reliable historical
# upset/recovery label or rule is available.


# =========================================================
# 12. CREATE FUTURE CO2 TARGET
# =========================================================
#
# Example:
#
# Current values at 08:00
#             ↓
# CO2 measurement at approximately 12:00
# =========================================================

data["CO2_4hr_future"] = (
    data["CO2_pct"].shift(
        -prediction_delay_hours
    )
)


# Store the future quality timestamp for review

data["Quality_Timestamp"] = (
    data["Timestamp"]
    + pd.Timedelta(
        hours=prediction_delay_hours
    )
)


# =========================================================
# 13. DEFINE CURRENT-ONLY MODEL FEATURES
# =========================================================

engineered_features = [
    "Gas_per_Rate",
    "Primary_Air_to_Gas",
    "Average_Flue_Temp",
    "Flue_Temp_Difference"
]

status_features = [
    "Status_Steady",
    "Status_Reduced_Rate",
    "Status_Startup",
    "Status_Shutdown",
    "Status_Upset_Recovery"
]

features = (
    raw_process_variables
    + engineered_features
    + status_features
)


print("\n========================================")
print("CURRENT-ONLY FEATURES")
print("========================================")

print("Feature count:", len(features))

for feature in features:

    print("-", feature)


# =========================================================
# 14. CREATE MODEL DATASET
# =========================================================

model_data = data.dropna(
    subset=features + ["CO2_4hr_future"]
).copy()


# Replace any positive or negative infinity with NaN

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

model_data = model_data.dropna(
    subset=features + ["CO2_4hr_future"]
)

model_data = (
    model_data
    .sort_values("Timestamp")
    .reset_index(drop=True)
)


print("\n========================================")
print("MODEL DATASET")
print("========================================")

print(
    "Usable samples:",
    len(model_data)
)

print(
    "Start:",
    model_data["Timestamp"].min()
)

print(
    "End:",
    model_data["Timestamp"].max()
)

print("\nTarget distribution:")

print(
    model_data[
        "CO2_4hr_future"
    ].describe()
)


# =========================================================
# 15. CHRONOLOGICAL 60 / 20 / 20 SPLIT
# =========================================================

train_end = int(
    len(model_data) * training_fraction
)

validation_end = int(
    len(model_data)
    * (
        training_fraction
        + validation_fraction
    )
)


X = model_data[features]

y = model_data["CO2_4hr_future"]


# First 60%

X_train = X.iloc[
    :train_end
]

y_train = y.iloc[
    :train_end
]


# Next 20%

X_validation = X.iloc[
    train_end:validation_end
]

y_validation = y.iloc[
    train_end:validation_end
]


# Final 20%

X_test = X.iloc[
    validation_end:
]

y_test = y.iloc[
    validation_end:
]


print("\n========================================")
print("CHRONOLOGICAL SPLIT")
print("========================================")

print("Training samples:", len(X_train))
print("Validation samples:", len(X_validation))
print("Testing samples:", len(X_test))

print("\nTraining period:")

print(
    model_data["Timestamp"].iloc[0],
    "to",
    model_data["Timestamp"].iloc[
        train_end - 1
    ]
)

print("\nValidation period:")

print(
    model_data["Timestamp"].iloc[
        train_end
    ],
    "to",
    model_data["Timestamp"].iloc[
        validation_end - 1
    ]
)

print("\nTesting period:")

print(
    model_data["Timestamp"].iloc[
        validation_end
    ],
    "to",
    model_data["Timestamp"].iloc[-1]
)


# =========================================================
# 16. TRAIN DEVELOPMENT MODEL
# =========================================================

development_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

development_model.fit(
    X_train,
    y_train
)


# =========================================================
# 17. VALIDATION PREDICTIONS
# =========================================================

validation_predictions = (
    development_model.predict(
        X_validation
    )
)

validation_predictions = np.maximum(
    validation_predictions,
    0
)


validation_mae = mean_absolute_error(
    y_validation,
    validation_predictions
)

validation_rmse = (
    mean_squared_error(
        y_validation,
        validation_predictions
    ) ** 0.5
)

validation_r2 = r2_score(
    y_validation,
    validation_predictions
)


print("\n========================================")
print("VALIDATION RESULTS")
print("========================================")

print(
    "Validation MAE:",
    round(validation_mae, 4)
)

print(
    "Validation RMSE:",
    round(validation_rmse, 4)
)

print(
    "Validation R2:",
    round(validation_r2, 4)
)


# =========================================================
# 18. RETRAIN ON FIRST 80%
# =========================================================
#
# After checking validation, train a fresh model using:
#
# Training 60%
# +
# Validation 20%
#
# Then test against the untouched final 20%.
# =========================================================

X_train_final = X.iloc[
    :validation_end
]

y_train_final = y.iloc[
    :validation_end
]


final_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

final_model.fit(
    X_train_final,
    y_train_final
)


# =========================================================
# 19. FINAL UNTOUCHED TEST
# =========================================================

test_predictions = final_model.predict(
    X_test
)

test_predictions = np.maximum(
    test_predictions,
    0
)


test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = (
    mean_squared_error(
        y_test,
        test_predictions
    ) ** 0.5
)

test_r2 = r2_score(
    y_test,
    test_predictions
)


print("\n========================================")
print("FINAL UNTOUCHED TEST RESULTS")
print("========================================")

print(
    "Test MAE:",
    round(test_mae, 4)
)

print(
    "Test RMSE:",
    round(test_rmse, 4)
)

print(
    "Test R2:",
    round(test_r2, 4)
)


# =========================================================
# 20. MEAN BASELINE
# =========================================================

training_mean = y_train_final.mean()

baseline_predictions = np.full(
    shape=len(y_test),
    fill_value=training_mean
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = (
    mean_squared_error(
        y_test,
        baseline_predictions
    ) ** 0.5
)

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)


print("\n========================================")
print("MEAN BASELINE RESULTS")
print("========================================")

print(
    "Baseline MAE:",
    round(baseline_mae, 4)
)

print(
    "Baseline RMSE:",
    round(baseline_rmse, 4)
)

print(
    "Baseline R2:",
    round(baseline_r2, 4)
)


# =========================================================
# 21. COMPARE WITH VERSION 1
# =========================================================

v1_config_path = (
    saved_model_folder
    / "K1_CO2_Config.pkl"
)

if v1_config_path.exists():

    v1_config = joblib.load(
        v1_config_path
    )

    print("\n========================================")
    print("VERSION COMPARISON")
    print("========================================")

    print("\nVERSION 1: CURRENT + HISTORY")

    print(
        "Feature count:",
        len(v1_config.get("Features", []))
    )

    print(
        "Test MAE:",
        round(
            v1_config["Final_Test_MAE"],
            4
        )
    )

    print(
        "Test RMSE:",
        round(
            v1_config["Final_Test_RMSE"],
            4
        )
    )

    print(
        "Test R2:",
        round(
            v1_config["Final_Test_R2"],
            4
        )
    )

    print("\nVERSION 2: CURRENT ONLY + STATUS")

    print(
        "Feature count:",
        len(features)
    )

    print(
        "Test MAE:",
        round(test_mae, 4)
    )

    print(
        "Test RMSE:",
        round(test_rmse, 4)
    )

    print(
        "Test R2:",
        round(test_r2, 4)
    )

else:

    print(
        "\nVersion 1 configuration file "
        "was not found. Current-only results "
        "were calculated normally."
    )


# =========================================================
# 22. CREATE FINAL RESULTS TABLE
# =========================================================

test_results = pd.DataFrame(
    {
        "Prediction_Timestamp":
            model_data["Timestamp"]
            .iloc[validation_end:]
            .values,

        "Expected_Quality_Timestamp":
            model_data["Quality_Timestamp"]
            .iloc[validation_end:]
            .values,

        "Operating_Status":
            model_data["Operating_Status"]
            .iloc[validation_end:]
            .values,

        "Actual_CO2":
            y_test.values,

        "Predicted_CO2":
            test_predictions
    }
)

test_results["Absolute_Error"] = (
    test_results["Actual_CO2"]
    - test_results["Predicted_CO2"]
).abs()


print("\n========================================")
print("WORST 15 TEST PREDICTIONS")
print("========================================")

print(
    test_results
    .sort_values(
        "Absolute_Error",
        ascending=False
    )
    .head(15)
    .to_string(index=False)
)


# =========================================================
# 23. PERFORMANCE BY OPERATING STATUS
# =========================================================

status_performance = []

for status_name, group in test_results.groupby(
    "Operating_Status"
):

    if len(group) < 2:

        continue

    status_mae = mean_absolute_error(
        group["Actual_CO2"],
        group["Predicted_CO2"]
    )

    status_rmse = (
        mean_squared_error(
            group["Actual_CO2"],
            group["Predicted_CO2"]
        ) ** 0.5
    )

    if len(group) >= 3:

        status_r2 = r2_score(
            group["Actual_CO2"],
            group["Predicted_CO2"]
        )

    else:

        status_r2 = np.nan

    status_performance.append(
        {
            "Operating_Status":
                status_name,

            "Samples":
                len(group),

            "MAE":
                status_mae,

            "RMSE":
                status_rmse,

            "R2":
                status_r2
        }
    )


status_performance = pd.DataFrame(
    status_performance
)


print("\n========================================")
print("TEST PERFORMANCE BY OPERATING STATUS")
print("========================================")

if not status_performance.empty:

    print(
        status_performance.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

else:

    print(
        "Not enough samples were available "
        "to calculate status-level metrics."
    )


# =========================================================
# 24. FEATURE IMPORTANCE
# =========================================================

feature_importance = pd.DataFrame(
    {
        "Feature":
            features,

        "Importance":
            final_model.feature_importances_
    }
)

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\n========================================")
print("FEATURE IMPORTANCE")
print("========================================")

print(
    feature_importance.to_string(
        index=False
    )
)


# =========================================================
# 25. SAVE CURRENT-ONLY MODEL
# =========================================================

model_path = (
    saved_model_folder
    / "K1_CO2_CurrentOnly_XGBoost.pkl"
)

features_path = (
    saved_model_folder
    / "K1_CO2_CurrentOnly_Features.pkl"
)

config_path = (
    saved_model_folder
    / "K1_CO2_CurrentOnly_Config.pkl"
)


joblib.dump(
    final_model,
    model_path
)

joblib.dump(
    features,
    features_path
)


current_only_config = {
    "Kiln":
        "K1",

    "Model_Name":
        "K1 CO2 Current Only",

    "Model_Type":
        "XGBRegressor",

    "Target":
        "CO2_pct",

    "Prediction_Delay_Hours":
        prediction_delay_hours,

    "Uses_Recent_History":
        False,

    "Uses_Operating_Status":
        True,

    "Status_Thresholds_Provisional":
        True,

    "Shutdown_Rate_Max":
        SHUTDOWN_RATE_MAX,

    "Shutdown_Gas_Max":
        SHUTDOWN_GAS_MAX,

    "Startup_Rate_Max":
        STARTUP_RATE_MAX,

    "Reduced_Rate_Max":
        REDUCED_RATE_MAX,

    "Feature_Count":
        len(features),

    "Features":
        features,

    "Training_Samples":
        len(X_train_final),

    "Testing_Samples":
        len(X_test),

    "Validation_MAE":
        float(validation_mae),

    "Validation_RMSE":
        float(validation_rmse),

    "Validation_R2":
        float(validation_r2),

    "Final_Test_MAE":
        float(test_mae),

    "Final_Test_RMSE":
        float(test_rmse),

    "Final_Test_R2":
        float(test_r2),

    "Baseline_MAE":
        float(baseline_mae),

    "Baseline_RMSE":
        float(baseline_rmse),

    "Baseline_R2":
        float(baseline_r2)
}


joblib.dump(
    current_only_config,
    config_path
)


print("\n========================================")
print("CURRENT-ONLY MODEL SAVED")
print("========================================")

print("Model:")
print(model_path)

print("\nFeatures:")
print(features_path)

print("\nConfiguration:")
print(config_path)


# =========================================================
# 26. SAVE ANALYSIS TO EXCEL
# =========================================================

model_summary = pd.DataFrame(
    [
        {
            "Model":
                "Current Only + Status",

            "Features":
                len(features),

            "Prediction_Delay_Hours":
                prediction_delay_hours,

            "Validation_MAE":
                validation_mae,

            "Validation_RMSE":
                validation_rmse,

            "Validation_R2":
                validation_r2,

            "Test_MAE":
                test_mae,

            "Test_RMSE":
                test_rmse,

            "Test_R2":
                test_r2,

            "Baseline_MAE":
                baseline_mae,

            "Baseline_RMSE":
                baseline_rmse,

            "Baseline_R2":
                baseline_r2
        }
    ]
)


with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    model_summary.to_excel(
        writer,
        sheet_name="Model Summary",
        index=False
    )

    test_results.to_excel(
        writer,
        sheet_name="Test Results",
        index=False
    )

    status_performance.to_excel(
        writer,
        sheet_name="Status Performance",
        index=False
    )

    feature_importance.to_excel(
        writer,
        sheet_name="Feature Importance",
        index=False
    )


print("\n========================================")
print("RESULTS SAVED")
print("========================================")

print(output_file)


# =========================================================
# 27. GRAPH ACTUAL VS PREDICTED
# =========================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    test_results[
        "Expected_Quality_Timestamp"
    ],
    test_results["Actual_CO2"],
    label="Actual CO2",
    linewidth=1.5
)

plt.plot(
    test_results[
        "Expected_Quality_Timestamp"
    ],
    test_results["Predicted_CO2"],
    label="Current-Only Predicted CO2",
    linewidth=1.5
)

plt.xlabel(
    "Expected Quality Time"
)

plt.ylabel(
    "CO2 (%)"
)

plt.title(
    "K1 Current-Only Model: "
    "Actual vs Predicted CO2"
)

plt.legend()

plt.tight_layout()

plt.show()


# =========================================================
# 28. GRAPH FEATURE IMPORTANCE
# =========================================================

top_features = (
    feature_importance
    .head(20)
    .sort_values("Importance")
)

plt.figure(
    figsize=(10, 8)
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel(
    "XGBoost Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "K1 Current-Only Model: "
    "Feature Importance"
)

plt.tight_layout()

plt.show()


# =========================================================
# 29. FINISHED
# =========================================================

print("\n========================================")
print("CURRENT-ONLY MODEL COMPLETE")
print("========================================")

print(
    "Compare the Version 2 test metrics "
    "against Version 1 before changing "
    "the Excel prediction workflow."
)

print(
    "Do not treat the provisional status "
    "thresholds as final operating rules."
)