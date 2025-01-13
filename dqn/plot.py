import pandas as pd

render_logs_dir = "render_stable"
name = "SOLUSDT_2025-01-13_21-42-58.pkl"
df = pd.read_pickle(f"{render_logs_dir}/{name}")


import matplotlib.pyplot as plt


fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot 'feature_close' on the left y-axis
ax1.plot(df.index, df["data_close"], label="Price", color="b")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price", color="b")
ax1.tick_params(axis="y", labelcolor="b")

# Create a second y-axis on the right
ax2 = ax1.twinx()
ax2.plot(
    df.index,
    df["portfolio_valuation"],
    label="Portfolio valuation",
    color="g",
)
ax2.set_ylabel("Portfolio valuation", color="g")
ax2.tick_params(axis="y", labelcolor="g")

# Title of the chart
plt.title("Feature Close and Portfolio Distribution Fiat over Time")

# Show the plot
fig.tight_layout()  # Adjusts the layout to ensure everything fits
plt.xticks(rotation=45)  # Rotates the x-axis labels for better readability
plt.show()
