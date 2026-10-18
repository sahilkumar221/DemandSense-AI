import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("data/hour.csv")

# Convert date column to datetime
df["dteday"] = pd.to_datetime(df["dteday"])

sns.set_theme(style="whitegrid")

# 1. Bike rentals by hour
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="hr", y="cnt", errorbar=None)
plt.title("Average Bike Rental Demand by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Rentals")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("demand_by_hour.png")
plt.show()

# 2. Bike rentals by season
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="season", y="cnt", errorbar=None)
plt.title("Average Bike Rental Demand by Season")
plt.xlabel("Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)")
plt.ylabel("Average Rentals")
plt.tight_layout()
plt.savefig("demand_by_season.png")
plt.show()

# 3. Weather and demand
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="weathersit", y="cnt", errorbar=None)
plt.title("Bike Rental Demand by Weather Condition")
plt.xlabel("Weather Situation")
plt.ylabel("Average Rentals")
plt.tight_layout()
plt.savefig("demand_by_weather.png")
plt.show()

print("Visualizations created successfully!")
print("Saved: demand_by_hour.png, demand_by_season.png, demand_by_weather.png")