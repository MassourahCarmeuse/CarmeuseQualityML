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
# 1. LOAD DATA
# =========================================================

# Find the main Kiln_Quality_Predictor folder
project_folder = Path(__file__).resolve().parent.parent

# Find the Data folder
data_folder = project_folder / "Data"

# Path to the Excel file
file_path = (
    data_folder
    / "Kiln_Quality_Model_Data_Extract.xlsx"
)

data = pd.read_excel(
    file_path,
    sheet_name="K1 Extract",
    engine="openpyxl"
)


# =========================================================
# 2. CLEAN BASIC DATA
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


numeric_columns = [
    "Product_Temp_F",
    "Feed_End_Temp_F",
    "South_Flue_Temp_F",
    "North_Flue_Temp_F",
    "Precip_Inlet_Temp_F",
    "MMBTU_per_ton",
    "Rate",
    "Kiln_Rev_RPH",
    "Kiln_Amps",
    "Gas_SCFM",
    "Primary_Air_Flow_SCFM",
    "Hood_Exit_Temp_F",
    "Cooler_Air_SCFM",
    "Exit_Temp_F",
    "O2_pct",
    "CO_ppm",
    "ID_Fan_Amps",
    "CO2_pct"
]


for column in numeric_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# =========================================================
# 3. CORRECT CONFIRMED CO2 ENTRY ERROR
# =========================================================

bad_timestamp = pd.Timestamp(
    "2026-07-23 08:59:59.999"
)

data.loc[
    data["Timestamp"] == bad_timestamp,
    "CO2_pct"
] = 2.91


print("\n--- CONFIRMED CO2 CORRECTION ---")

print(
    data.loc[
        data["Timestamp"] == bad_timestamp,
        ["Timestamp", "CO2_pct"]
    ]
)


# =========================================================
# 4. MODEL VARIABLES
# =========================================================

process_variables = [
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
# 5. CREATE 6-MINUTE PROCESS GRID
# =========================================================
#
# 0.1 hour = 6 minutes.
#
# The original process data is hourly, so values between
# hours are linearly interpolated for this delay experiment.
# CO2 is NOT interpolated.
# =========================================================

process_data = (
    data[
        ["Timestamp"] + process_variables
    ]
    .copy()
)


# Round tiny millisecond timestamp differences
process_data["Timestamp"] = (
    process_data["Timestamp"].dt.round("h")
)


# Remove duplicate timestamps if any exist
process_data = (
    process_data
    .drop_duplicates(
        subset=["Timestamp"],
        keep="first"
    )
    .set_index("Timestamp")
    .sort_index()
)


# Create a complete 6-minute time grid
six_minute_index = pd.date_range(
    start=process_data.index.min(),
    end=process_data.index.max(),
    freq="6min"
)


process_grid = process_data.reindex(
    six_minute_index
)


# Interpolate process measurements only
process_grid[process_variables] = (
    process_grid[process_variables]
    .interpolate(
        method="time",
        limit_area="inside"
    )
)


process_grid.index.name = "Feature_Timestamp"


print("\n--- PROCESS GRID ---")
print("Original process rows:", len(process_data))
print("Six-minute process rows:", len(process_grid))


# =========================================================
# 6. CREATE PROCESS HISTORY FEATURES
# =========================================================
#
# At each 6-minute point, calculate:
#
# current value
# 1-hour average
# 2-hour average
# 4-hour average
# 1-hour change
# =========================================================

feature_grid = pd.DataFrame(
    index=process_grid.index
)


for variable in process_variables:

    # Current reading
    feature_grid[variable] = (
        process_grid[variable]
    )

    # Recent averages
    feature_grid[
        f"{variable}_avg_1hr"
    ] = (
        process_grid[variable]
        .rolling("1h", min_periods=2)
        .mean()
    )

    feature_grid[
        f"{variable}_avg_2hr"
    ] = (
        process_grid[variable]
        .rolling("2h", min_periods=2)
        .mean()
    )

    feature_grid[
        f"{variable}_avg_4hr"
    ] = (
        process_grid[variable]
        .rolling("4h", min_periods=2)
        .mean()
    )

    # 10 rows on a 6-minute grid equals 1 hour
    feature_grid[
        f"{variable}_change_1hr"
    ] = (
        process_grid[variable]
        - process_grid[variable].shift(10)
    )


# Engineered features
feature_grid["Gas_per_Rate"] = (
    feature_grid["Gas_SCFM"]
    / feature_grid["Rate"].replace(0, np.nan)
)

feature_grid["Primary_Air_to_Gas"] = (
    feature_grid["Primary_Air_Flow_SCFM"]
    / feature_grid["Gas_SCFM"].replace(0, np.nan)
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


features = feature_grid.columns.tolist()


# =========================================================
# 7. CREATE ACTUAL CO2 SAMPLE TABLE
# =========================================================
#
# Only actual CO2 samples are retained.
# CO2 values are not interpolated.
# =========================================================

co2_samples = (
    data.loc[
        data["CO2_pct"].notna(),
        ["Timestamp", "CO2_pct"]
    ]
    .copy()
    .rename(
        columns={
            "Timestamp": "Quality_Timestamp",
            "CO2_pct": "Actual_CO2"
        }
    )
    .sort_values("Quality_Timestamp")
    .reset_index(drop=True)
)


print("\nActual CO2 samples:", len(co2_samples))


# =========================================================
# 8. DELAYS TO TEST
# =========================================================

delays = np.round(
    np.arange(3.2, 4.51, 0.1),
    1
)

print("\nDelays being tested:")
print(delays)


# =========================================================
# 9. TEST DELAYS USING TRAINING + VALIDATION DATA
# =========================================================

validation_results = []

best_delay = None
best_validation_r2 = -np.inf
best_validation_mae = None
best_validation_rmse = None
best_delay_model_data = None


for delay_hours in delays:

    print(
        f"\nTesting delay: {delay_hours:.1f} hours"
    )

    delay_data = co2_samples.copy()

    # Calculate when the prediction would have been made
    delay_data["Requested_Feature_Timestamp"] = (
        delay_data["Quality_Timestamp"]
        - pd.to_timedelta(
            delay_hours,
            unit="h"
        )
    )

    # Round to the nearest 6-minute grid point
    delay_data["Feature_Timestamp"] = (
        delay_data[
            "Requested_Feature_Timestamp"
        ]
        .dt.round("6min")
    )

    # Join process features to the quality results
    delay_model_data = delay_data.merge(
        feature_grid.reset_index(),
        on="Feature_Timestamp",
        how="left"
    )

    # Remove rows missing process features or CO2
    delay_model_data = delay_model_data.dropna(
        subset=features + ["Actual_CO2"]
    )

    # Ensure chronological order
    delay_model_data = (
        delay_model_data
        .sort_values("Quality_Timestamp")
        .reset_index(drop=True)
    )

    if len(delay_model_data) < 100:

        print(
            "Skipped because fewer than 100 samples "
            "were available."
        )

        continue

    # -----------------------------------------------------
    # 60% TRAIN / 20% VALIDATION / 20% FINAL TEST
    # -----------------------------------------------------

    train_end = int(
        len(delay_model_data) * 0.60
    )

    validation_end = int(
        len(delay_model_data) * 0.80
    )

    X = delay_model_data[features]
    y = delay_model_data["Actual_CO2"]

    # First 60%
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    # Next 20%
    X_validation = X.iloc[
        train_end:validation_end
    ]

    y_validation = y.iloc[
        train_end:validation_end
    ]

    # The final 20% is deliberately not used here

    # -----------------------------------------------------
    # TRAIN VALIDATION MODEL
    # -----------------------------------------------------

    validation_model = XGBRegressor(
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

    validation_model.fit(
        X_train,
        y_train
    )

    validation_predictions = (
        validation_model.predict(
            X_validation
        )
    )

    # Prevent physically impossible negative predictions
    validation_predictions = np.maximum(
        validation_predictions,
        0
    )

    # -----------------------------------------------------
    # VALIDATION METRICS
    # -----------------------------------------------------

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

    validation_results.append({

        "Delay_Hours":
            delay_hours,

        "Total_Samples":
            len(delay_model_data),

        "Training_Samples":
            len(X_train),

        "Validation_Samples":
            len(X_validation),

        "Validation_MAE":
            validation_mae,

        "Validation_RMSE":
            validation_rmse,

        "Validation_R2":
            validation_r2

    })

    print(
        f"Validation MAE={validation_mae:.4f}, "
        f"RMSE={validation_rmse:.4f}, "
        f"R2={validation_r2:.4f}"
    )

    # Select best delay using validation R2 only
    if validation_r2 > best_validation_r2:

        best_delay = delay_hours

        best_validation_r2 = (
            validation_r2
        )

        best_validation_mae = (
            validation_mae
        )

        best_validation_rmse = (
            validation_rmse
        )

        best_delay_model_data = (
            delay_model_data.copy()
        )


# =========================================================
# 10. VALIDATION COMPARISON TABLE
# =========================================================

validation_comparison = pd.DataFrame(
    validation_results
)

validation_comparison = (
    validation_comparison
    .sort_values(
        by=[
            "Validation_R2",
            "Validation_MAE"
        ],
        ascending=[
            False,
            True
        ]
    )
    .reset_index(drop=True)
)


print("\n========================================")
print("VALIDATION DELAY COMPARISON")
print("========================================")

print(
    validation_comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


print("\n========================================")
print("BEST VALIDATION DELAY")
print("========================================")

print(
    f"Best delay: {best_delay:.1f} hours"
)

print(
    f"Validation MAE: "
    f"{best_validation_mae:.4f}"
)

print(
    f"Validation RMSE: "
    f"{best_validation_rmse:.4f}"
)

print(
    f"Validation R2: "
    f"{best_validation_r2:.4f}"
)


# =========================================================
# 11. FINAL TRAINING AND TESTING
# =========================================================
#
# After selecting the delay:
#
# First 80% = final model training
# Last 20%  = untouched final test
#
# The final test was not used to select the delay.
# =========================================================

final_data = best_delay_model_data.copy()

final_train_end = int(
    len(final_data) * 0.80
)

X_final = final_data[features]
y_final = final_data["Actual_CO2"]

# First 80%
X_train_final = X_final.iloc[
    :final_train_end
]

y_train_final = y_final.iloc[
    :final_train_end
]

# Final untouched 20%
X_test_final = X_final.iloc[
    final_train_end:
]

y_test_final = y_final.iloc[
    final_train_end:
]


print("\n========================================")
print("FINAL TRAIN / TEST SPLIT")
print("========================================")

print(
    "Total samples:",
    len(final_data)
)

print(
    "Final training samples:",
    len(X_train_final)
)

print(
    "Final testing samples:",
    len(X_test_final)
)

print("\nFinal training period:")

print(
    final_data["Quality_Timestamp"].iloc[0],
    "to",
    final_data["Quality_Timestamp"].iloc[
        final_train_end - 1
    ]
)

print("\nFinal testing period:")

print(
    final_data["Quality_Timestamp"].iloc[
        final_train_end
    ],
    "to",
    final_data["Quality_Timestamp"].iloc[-1]
)


# =========================================================
# 12. TRAIN FINAL XGBOOST MODEL
# =========================================================

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
# 13. FINAL UNTOUCHED TEST PREDICTIONS
# =========================================================

final_predictions = final_model.predict(
    X_test_final
)

final_predictions = np.maximum(
    final_predictions,
    0
)


# =========================================================
# 14. FINAL TEST METRICS
# =========================================================

final_mae = mean_absolute_error(
    y_test_final,
    final_predictions
)

final_rmse = (
    mean_squared_error(
        y_test_final,
        final_predictions
    ) ** 0.5
)

final_r2 = r2_score(
    y_test_final,
    final_predictions
)


print("\n========================================")
print("FINAL UNTOUCHED TEST RESULTS")
print("========================================")

print(
    f"Selected delay: "
    f"{best_delay:.1f} hours"
)

print(
    f"Final test MAE: "
    f"{final_mae:.4f}"
)

print(
    f"Final test RMSE: "
    f"{final_rmse:.4f}"
)

print(
    f"Final test R2: "
    f"{final_r2:.4f}"
)


# =========================================================
# 15. SIMPLE MEAN BASELINE
# =========================================================
#
# This confirms whether XGBoost beats simply predicting
# the training-period average CO2 for every test sample.
# =========================================================

training_mean = y_train_final.mean()

baseline_predictions = np.full(
    shape=len(y_test_final),
    fill_value=training_mean
)

baseline_mae = mean_absolute_error(
    y_test_final,
    baseline_predictions
)

baseline_rmse = (
    mean_squared_error(
        y_test_final,
        baseline_predictions
    ) ** 0.5
)

baseline_r2 = r2_score(
    y_test_final,
    baseline_predictions
)


print("\n========================================")
print("MEAN BASELINE RESULTS")
print("========================================")

print(
    f"Baseline MAE: "
    f"{baseline_mae:.4f}"
)

print(
    f"Baseline RMSE: "
    f"{baseline_rmse:.4f}"
)

print(
    f"Baseline R2: "
    f"{baseline_r2:.4f}"
)


# =========================================================
# 16. CREATE FINAL RESULTS TABLE
# =========================================================

final_results = pd.DataFrame({

    "Quality_Timestamp":
        final_data[
            "Quality_Timestamp"
        ]
        .iloc[final_train_end:]
        .values,

    "Feature_Timestamp":
        final_data[
            "Feature_Timestamp"
        ]
        .iloc[final_train_end:]
        .values,

    "Actual_CO2":
        y_test_final.values,

    "Predicted_CO2":
        final_predictions

})

final_results["Absolute_Error"] = (

    final_results["Actual_CO2"]
    - final_results["Predicted_CO2"]

).abs()


print("\n--- WORST 15 FINAL TEST PREDICTIONS ---")

print(
    final_results
    .sort_values(
        "Absolute_Error",
        ascending=False
    )
    .head(15)
    .to_string(index=False)
)


# =========================================================
# 17. SAVE MODEL FILES
# =========================================================

saved_model_folder = Path(
    "../Saved_Models"
)

saved_model_folder.mkdir(
    parents=True,
    exist_ok=True
)


model_path = (
    saved_model_folder
    / "K1_CO2_XGBoost.pkl"
)

features_path = (
    saved_model_folder
    / "K1_CO2_Features.pkl"
)

config_path = (
    saved_model_folder
    / "K1_CO2_Config.pkl"
)


joblib.dump(
    final_model,
    model_path
)

joblib.dump(
    features,
    features_path
)


model_config = {

    "Kiln":
        "K1",

    "Target":
        "CO2_pct",

    "Delay_Hours":
        float(best_delay),

    "Grid_Minutes":
        6,

    "Source_Data_Frequency":
        "Hourly",

    "Interpolation_Method":
        "Linear time interpolation",

    "Model_Type":
        "XGBRegressor",

    "Training_Samples":
        len(X_train_final),

    "Final_Test_Samples":
        len(X_test_final),

    "Final_Test_MAE":
        float(final_mae),

    "Final_Test_RMSE":
        float(final_rmse),

    "Final_Test_R2":
        float(final_r2),

    "Features":
        features

}


joblib.dump(
    model_config,
    config_path
)


print("\n========================================")
print("SAVED MODEL FILES")
print("========================================")

print("Model:")
print(model_path)

print("\nFeature list:")
print(features_path)

print("\nConfiguration:")
print(config_path)


# =========================================================
# 18. SAVE EXCEL RESULTS
# =========================================================

output_file = (
    "../Data/"
    "K1_CO2_Final_Model_Results.xlsx"
)

with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    validation_comparison.to_excel(
        writer,
        sheet_name="Delay Validation",
        index=False
    )

    final_results.to_excel(
        writer,
        sheet_name="Final Test Results",
        index=False
    )

    pd.DataFrame(
        [model_config]
    ).drop(
        columns=["Features"]
    ).to_excel(
        writer,
        sheet_name="Model Summary",
        index=False
    )


print("\nFinal results saved to:")
print(output_file)


# =========================================================
# 19. GRAPH VALIDATION R2 BY DELAY
# =========================================================

validation_by_delay = (
    validation_comparison
    .sort_values("Delay_Hours")
)


plt.figure(
    figsize=(12, 6)
)

plt.plot(
    validation_by_delay["Delay_Hours"],
    validation_by_delay["Validation_R2"],
    marker="o",
    label="Validation R2"
)

plt.axhline(
    y=0,
    color="black",
    linestyle="--",
    linewidth=1
)

plt.axvline(
    x=best_delay,
    color="red",
    linestyle="--",
    linewidth=1,
    label=(
        f"Best delay: "
        f"{best_delay:.1f} hours"
    )
)

plt.xlabel(
    "Prediction Delay (hours)"
)

plt.ylabel(
    "Validation R2"
)

plt.title(
    "K1 CO2 Validation Performance by Delay"
)

plt.xticks(
    validation_by_delay["Delay_Hours"]
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.show()


# =========================================================
# 20. GRAPH FINAL ACTUAL VS PREDICTED
# =========================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    final_results["Quality_Timestamp"],
    final_results["Actual_CO2"],
    label="Actual CO2",
    linewidth=1.5
)

plt.plot(
    final_results["Quality_Timestamp"],
    final_results["Predicted_CO2"],
    label=(
        f"Predicted CO2 "
        f"({best_delay:.1f}-hour delay)"
    ),
    linewidth=1.5
)

plt.xlabel(
    "Quality Sample Time"
)

plt.ylabel(
    "CO2 (%)"
)

plt.title(
    "K1 Final Untouched Test: "
    "Actual vs Predicted CO2"
)

plt.legend()

plt.tight_layout()

plt.show()


# =========================================================
# 21. FEATURE IMPORTANCE
# =========================================================

feature_importance = pd.DataFrame({

    "Feature":
        features,

    "Importance":
        final_model.feature_importances_

})

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\n--- TOP 20 MODEL FEATURES ---")

print(
    feature_importance
    .head(20)
    .to_string(index=False)
)


plt.figure(
    figsize=(10, 8)
)

top_features = (
    feature_importance
    .head(20)
    .sort_values("Importance")
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
    "K1 CO2 Model: Top 20 Features"
)

plt.tight_layout()

plt.show()
# =========================================================
# 22. TRAIN DEPLOYMENT MODEL
# =========================================================
#
# IMPORTANT:
# The evaluation model was tested using unseen data.
#
# Now that evaluation is finished, this second model
# trains on ALL approved historical model data so that
# future live predictions can use everything we know.
#
# Do NOT use this model's training performance as an
# accuracy estimate.
# =========================================================

deployment_model = XGBRegressor(
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


# =========================================================
# 23. TRAIN USING ALL APPROVED MODEL DATA
# =========================================================

deployment_model.fit(
    X_final,
    y_final
)


# =========================================================
# 24. SAVE DEPLOYMENT MODEL
# =========================================================

deployment_model_path = (
    saved_model_folder
    / "K1_CO2_XGBoost_Deployment.pkl"
)

joblib.dump(
    deployment_model,
    deployment_model_path
)


print("\n========================================")
print("DEPLOYMENT MODEL SAVED")
print("========================================")

print(
    "Training samples:",
    len(X_final)
)

print(
    "Saved to:",
    deployment_model_path
)
