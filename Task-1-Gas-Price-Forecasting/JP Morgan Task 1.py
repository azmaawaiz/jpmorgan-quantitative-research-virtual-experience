import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

#--------------------------
# FILE DIALOG
#--------------------------


def choose_file_dialog(initial_dir):
    Tk().withdraw()
    return askopenfilename(
        initialdir=initial_dir,
        title="Select the CSV file",
        filetypes=[("CSV files", "*.csv")]
    ) 

#IMPORTANT NOTE: The code will show an error/exit the program if the user cancels the file selection in the file dialouge box.
#The code uses the file pattern of "Nat_Gas.csv"
# Please ensure to select the correct CSV file to avoid any issues.

#The CSV file names of the comlumns was changed from "Dates" and "Prices" to "Date" and "Price" respectively. The code has been updated accordingly.
#The CSV file's column "Price" was also changed to include the £ symbol. 

df = pd.read_csv(
    choose_file_dialog("."),
    encoding="cp1252"
)
df = df[["Date", "Price"]]

# Convert dates
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%m/%d/%y"
)

# Remove £ if necessary
df["Price"] = (
    df["Price"]
    .astype(str)
    .str.replace("£", "", regex=False)
    .astype(float)
)

df = df.sort_values("Date").reset_index(drop=True)

# Extract month
df["Month"] = df["Date"].dt.month

print("=" * 70)
print("HISTORICAL DATA")
print("=" * 70)
print(df.head())

# ------------------------------
# CALCULATE PERCENTAGE CHANGES
#-------------------------------

df["Pct_Change"] = df["Price"].pct_change()

print("\nAverage Monthly Percentage Change")
print("----------------------------------")
print(f"{df['Pct_Change'].mean()*100:.2f}%")

# ------------------------------------------------
# SEASONAL GROUPS
# ------------------------------------------------

decreasing_months = [3, 4, 5, 6, 7, 8, 9]
increasing_months = [9, 10, 11, 12, 1, 2, 3]

decreasing_data = df[df["Month"].isin(decreasing_months)]
increasing_data = df[df["Month"].isin(increasing_months)]

decreasing_pct = (
    decreasing_data["Pct_Change"]
    .dropna()
    .mean()
)

increasing_pct = (
    increasing_data["Pct_Change"]
    .dropna()
    .mean()
)

overall_pct = (
    df["Pct_Change"]
    .dropna()
    .mean()
)

print("\n")
print("=" * 70)
print("SEASONAL ANALYSIS")
print("=" * 70)

print(f"Increasing Group Average : {increasing_pct*100:.2f}%")
print(f"Decreasing Group Average : {decreasing_pct*100:.2f}%")
print(f"Overall Average          : {overall_pct*100:.2f}%")

# -----------------------------
# FORECAST SETTINGS
# -----------------------------

forecast_start = pd.Timestamp("2024-10-31")
forecast_end = pd.Timestamp("2026-09-30")

forecast_dates = pd.date_range(
    forecast_start,
    forecast_end,
    freq="ME"
)

last_price = df.iloc[-1]["Price"]

forecast_prices = []

print("\n")
print("=" * 70)
print("FORECASTING")
print("=" * 70)

# ------------------
# FORECAST LOOP
# ------------------

for forecast_date in forecast_dates:

    month = forecast_date.month

    if month in [10, 11, 12, 1, 2]:

        expected_change = (
            increasing_pct +
            overall_pct
        ) / 2

        logic = "Increasing Group"

    elif month in [4, 5, 6, 7, 8]:

        expected_change = (
            decreasing_pct +
            overall_pct
        ) / 2

        logic = "Decreasing Group"

    else:

        expected_change = (
            increasing_pct +
            decreasing_pct +
            overall_pct
        ) / 3

        logic = "Overlap"

    predicted_price = last_price * (1 + expected_change)

    forecast_prices.append({
        "Date": forecast_date,
        "Price": predicted_price,
        "Logic": logic,
        "Expected Change": expected_change * 100
    })

    print(f"{forecast_date.strftime('%B %Y')}")
    print(f"Logic Used        : {logic}")
    print(f"Previous Price    : £{last_price:.2f}")
    print(f"Expected Change   : {expected_change*100:.2f}%")
    print(f"Forecast Price    : £{predicted_price:.2f}")
    print("-" * 55)

    # Update for next month's prediction
    last_price = predicted_price
    
# ------------------------------
# CREATE FORECAST DATAFRAME
# ------------------------------

forecast_df = pd.DataFrame(forecast_prices)

forecast_df["Price"] = forecast_df["Price"].round(2)
forecast_df["Expected Change"] = (
    forecast_df["Expected Change"]
    .round(2)
)

print("\n")
print("=" * 70)
print("FORECAST RESULTS")
print("=" * 70)

print(forecast_df)

# -----------------------------------
# COMBINE HISTORICAL + FORECAST
# -----------------------------------

historical = df[["Date", "Price"]].copy()
historical["Type"] = "Historical"

forecast = forecast_df[["Date", "Price"]].copy()
forecast["Type"] = "Forecast"

combined = pd.concat(
    [historical, forecast],
    ignore_index=True
)

combined = combined.sort_values("Date")
combined.reset_index(drop=True, inplace=True)

# -----------------------------------
# SUMMARY STATISTICS
# -----------------------------------

print("\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Historical observations : {len(historical)}")
print(f"Forecast observations   : {len(forecast)}")
print(f"Combined observations   : {len(combined)}")

print(f"\nLast Historical Price : £{historical.iloc[-1]['Price']:.2f}")
print(f"Last Forecast Price   : £{forecast.iloc[-1]['Price']:.2f}")

# -----------------------------------
# VISUALISATION
# -----------------------------------

plt.figure(figsize=(14,6))

plt.plot(
    historical["Date"],
    historical["Price"],
    linewidth=2,
    label="Historical Prices"
)

plt.plot(
    forecast["Date"],
    forecast["Price"],
    linewidth=2,
    linestyle="--",
    label="Forecast"
)

plt.axvline(
    historical.iloc[-1]["Date"],
    linestyle=":",
    linewidth=2,
    label="Forecast Start"
)

plt.title("Natural Gas Prices\nHistorical and Forecast")
plt.xlabel("Date")
plt.ylabel("Price (£)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------------
# DISPLAY LAST FEW OBSERVATIONS
# -----------------------------------

print("\n")
print("=" * 70)
print("LAST 10 HISTORICAL OBSERVATIONS")
print("=" * 70)

print(historical.tail(10))

print("\n")
print("=" * 70)
print("12-MONTH FORECAST")
print("=" * 70)

print(forecast)

# -----------------------------------
# PREPARE DATA FOR PRICE ESTIMATION
# -----------------------------------

price_curve = combined.copy()

price_curve = (
    price_curve
    .drop_duplicates("Date")
    .sort_values("Date")
    .reset_index(drop=True)
)

price_curve["Timestamp"] = (
    price_curve["Date"]
    .astype("int64")
)

print("\nPrice curve successfully created.")
print("Ready to estimate prices for any date.")

# -----------------------------------
# PRICE ESTIMATION FUNCTION
# -----------------------------------

def estimate_price(input_date):

    input_date = pd.to_datetime(input_date)

    # Check date range
    if input_date < price_curve["Date"].min():
        raise ValueError(
            f"Date must not be earlier than "
            f"{price_curve['Date'].min().date()}"
        )

    if input_date > price_curve["Date"].max():
        raise ValueError(
            f"Date must not be later than "
            f"{price_curve['Date'].max().date()}"
        )

    # Exact monthly observation
    exact = price_curve.loc[
        price_curve["Date"] == input_date
    ]

    if not exact.empty:
        return float(exact.iloc[0]["Price"])

    # Previous and next month-end observations
    previous = price_curve.loc[
        price_curve["Date"] < input_date
    ].iloc[-1]

    following = price_curve.loc[
        price_curve["Date"] > input_date
    ].iloc[0]

    total_days = (
        following["Date"] -
        previous["Date"]
    ).days

    elapsed_days = (
        input_date -
        previous["Date"]
    ).days

    weight = elapsed_days / total_days

    estimated_price = (
        previous["Price"] +
        weight *
        (
            following["Price"] -
            previous["Price"]
        )
    )

    return estimated_price


# -----------------------------------
# USER INPUT
# -----------------------------------

print("\n")
print("=" * 70)
print("PRICE ESTIMATOR")
print("=" * 70)

print(f"Available dates:")
print(f"{price_curve['Date'].min().date()}  ->  {price_curve['Date'].max().date()}")

while True:

    user_date = input(
        "\nEnter a date (YYYY-MM-DD) or 'q' to quit: "
    )

    if user_date.lower() == "q":
        break

    try:

        estimate = estimate_price(user_date)

        print(
            f"\nEstimated Natural Gas Price on "
            f"{pd.to_datetime(user_date).date()} "
            f"= £{estimate:.2f}"
        )

    except Exception as e:

        print(f"\nError: {e}")

# -----------------------------------
# EXPORT COMPLETE PRICE CURVE
# -----------------------------------

price_curve.to_csv(
    "natural_gas_price_curve.csv",
    index=False
)

forecast_df.to_csv(
    "natural_gas_forecast.csv",
    index=False
)

print("\n")
print("=" * 70)
print("FILES EXPORTED")
print("=" * 70)

print("natural_gas_price_curve.csv")
print("natural_gas_forecast.csv")

print("\nScript completed successfully.")