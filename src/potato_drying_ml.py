import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

df = pd.read_csv("data/potato_drying_data.csv")

print(df.head())
print(df.shape)

print(df.columns)

df.info()

print(df.isnull().sum())

print(df.describe())
df_60 = df[df["Oven_T_C"] == 60]

print(df_60)

plt.scatter(df_60["Time_min"], df_60["Weight_g"])

plt.xlabel("Drying Time (min)")
plt.ylabel("Weight (g)")
plt.title("Weight vs Drying Time at 60°C")

plt.show()

for temp in df["Oven_T_C"].unique():

    df_temp = df[df["Oven_T_C"] == temp]

    plt.plot(
        df_temp["Time_min"],
        df_temp["Weight_g"],
        marker="o",
        label=f"{temp}°C"
    )

plt.xlabel("Drying Time (min)")
plt.ylabel("Weight (g)")
plt.title("Weight Loss at Different Oven Temperatures")
plt.legend()
plt.show()

grouped = (
    df.groupby(["Oven_T_C", "Time_min"])["Weight_g"]
    .agg(["mean", "std"])
    .reset_index()
)

print(grouped)



for temp in grouped["Oven_T_C"].unique():

    temp_data = grouped[grouped["Oven_T_C"] == temp]

    plt.errorbar(
        temp_data["Time_min"],
        temp_data["mean"],
        yerr=temp_data["std"],
        marker="o",
        capsize=4,
        label=f"{temp}°C"
    )

plt.xlabel("Drying Time (min)")
plt.ylabel("Mean Weight (g)")
plt.title("Weight Loss at Different Oven Temperatures")
plt.legend()

plt.show()

initial_weight = (
    df[df["Time_min"] == 0]
    .set_index("Sample")["Weight_g"]
)

print(initial_weight)
df["Initial_Weight_g"] = df["Sample"].map(initial_weight)
df["Normalized_Weight"] = (
    df["Weight_g"] / df["Initial_Weight_g"]
)
print(
    df[
        ["Sample", "Oven_T_C", "Time_min",
         "Weight_g", "Initial_Weight_g", "Normalized_Weight"]
    ]
)
norm_grouped = (
    df.groupby(["Oven_T_C", "Time_min"])["Normalized_Weight"]
    .agg(["mean", "std"])
    .reset_index()
)

print(norm_grouped)
for temp in norm_grouped["Oven_T_C"].unique():

    temp_data = norm_grouped[
        norm_grouped["Oven_T_C"] == temp
    ]

    plt.errorbar(
        temp_data["Time_min"],
        temp_data["mean"],
        yerr=temp_data["std"],
        marker="o",
        capsize=4,
        label=f"{temp}°C"
    )

plt.xlabel("Drying Time (min)")
plt.ylabel("Normalized Weight")
plt.title("Normalized Weight vs Drying Time")
plt.legend()
plt.show()

X = df[["Oven_T_C", "Time_min"]]

y = df["Normalized_Weight"]
print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

print("Temperature coefficient:", model.coef_[0])
print("Time coefficient:", model.coef_[1])

new_data = pd.DataFrame({
    "Oven_T_C": [80],
    "Time_min": [30]
})

prediction = model.predict(new_data)

print(prediction)

# Prediction on unseen test data
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print(results)

results["Residual"] = results["Actual"] - results["Predicted"]

print(results)

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Normalized Weight")
plt.ylabel("Predicted Normalized Weight")
plt.title("Actual vs Predicted")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "--"
)

plt.show()

plt.scatter(y_pred, results["Residual"])

plt.axhline(0, linestyle="--")

plt.xlabel("Predicted Normalized Weight")
plt.ylabel("Residual")
plt.title("Residual Plot")

plt.show()

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# RMSE
train_rmse = np.sqrt(
    mean_squared_error(y_train, y_train_pred)
)

test_rmse = np.sqrt(
    mean_squared_error(y_test, y_test_pred)
)

# R²
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

print("Train R²:", train_r2)
print("Test R²:", test_r2)

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

rf_train_rmse = np.sqrt(
    mean_squared_error(y_train, rf_train_pred)
)

rf_test_rmse = np.sqrt(
    mean_squared_error(y_test, rf_test_pred)
)

rf_train_r2 = r2_score(y_train, rf_train_pred)
rf_test_r2 = r2_score(y_test, rf_test_pred)

print("RF Train RMSE:", rf_train_rmse)
print("RF Test RMSE:", rf_test_rmse)

print("RF Train R²:", rf_train_r2)
print("RF Test R²:", rf_test_r2)


lr_cv = cross_val_score(
    LinearRegression(),
    X,
    y,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

print(lr_cv)

lr_rmse = -lr_cv

print("LR RMSE for each fold:", lr_rmse)
print("LR Mean RMSE:", lr_rmse.mean())
print("LR SD:", lr_rmse.std())

rf_cv = cross_val_score(
    RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    X,
    y,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

rf_rmse = -rf_cv

print("RF RMSE for each fold:", rf_rmse)
print("RF Mean RMSE:", rf_rmse.mean())
print("RF SD:", rf_rmse.std())

from sklearn.model_selection import GroupKFold

groups = df["Sample"]

group_cv = GroupKFold(n_splits=4)

lr_group_scores = cross_val_score(
    LinearRegression(),
    X,
    y,
    cv=group_cv,
    groups=groups,
    scoring="neg_root_mean_squared_error"
)

lr_group_rmse = -lr_group_scores

print("LR Group RMSE:", lr_group_rmse)
print("LR Group Mean RMSE:", lr_group_rmse.mean())
print("LR Group SD:", lr_group_rmse.std())

rf_group_scores = cross_val_score(
    RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    X,
    y,
    cv=group_cv,
    groups=groups,
    scoring="neg_root_mean_squared_error"
)

rf_group_rmse = -rf_group_scores

print("RF Group RMSE:", rf_group_rmse)
print("RF Group Mean RMSE:", rf_group_rmse.mean())
print("RF Group SD:", rf_group_rmse.std())

from sklearn.model_selection import LeaveOneGroupOut

temp_groups = df["Oven_T_C"]

logo = LeaveOneGroupOut()

lr_temp_scores = cross_val_score(
    LinearRegression(),
    X,
    y,
    cv=logo,
    groups=temp_groups,
    scoring="neg_root_mean_squared_error"
)

lr_temp_rmse = -lr_temp_scores

print("LR Temperature RMSE:", lr_temp_rmse)
print("LR Mean:", lr_temp_rmse.mean())
print("LR SD:", lr_temp_rmse.std())

rf_temp_scores = cross_val_score(
    RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    X,
    y,
    cv=logo,
    groups=temp_groups,
    scoring="neg_root_mean_squared_error"
)

rf_temp_rmse = -rf_temp_scores

print("RF Temperature RMSE:", rf_temp_rmse)
print("RF Mean:", rf_temp_rmse.mean())
print("RF SD:", rf_temp_rmse.std())

for train_index, test_index in logo.split(X, y, groups=temp_groups):

    test_temperature = df.iloc[test_index]["Oven_T_C"].unique()

    print("Test Temperature:", test_temperature)



from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 5, None]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

grid_search.fit(X, y)

from sklearn.model_selection import GridSearchCV, GroupKFold

group_cv = GroupKFold(n_splits=4)

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=group_cv,
    scoring="neg_root_mean_squared_error"
)

grid_search.fit(
    X,
    y,
    groups=df["Sample"]
)

print("Best parameters:")
print(grid_search.best_params_)

print("Best CV RMSE:")
print(-grid_search.best_score_)

best_rf = grid_search.best_estimator_

for feature, importance in zip(
    X.columns,
    best_rf.feature_importances_
):
    print(feature, ":", importance)