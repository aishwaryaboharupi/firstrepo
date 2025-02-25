import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("hypersonic_flight_synthetic.csv")

# Display first few rows
print(df.head())

# Get basic info
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Assumptions for calculations
vehicle_mass = 5000  # kg (assumed mass of the hypersonic vehicle)
air_density = 1.225  # kg/m³ (sea level, but varies with altitude)

# Add Lift-to-Drag Ratio
df["L/D Ratio"] = 1 / df["Drag Coefficient"]

# Add Kinetic Energy (J)
df["Kinetic Energy (J)"] = 0.5 * vehicle_mass * (df["Velocity (m/s)"] ** 2)

# Add Dynamic Pressure (q)
df["Dynamic Pressure (Pa)"] = 0.5 * air_density * (df["Velocity (m/s)"] ** 2)

# Save the updated dataset
df.to_csv("hypersonic_flight_features.csv", index=False)

# Print summary
print(df.head())

# Plot correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")  # Save before show
plt.show()

# Plot Altitude vs Velocity
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Altitude (km)"], y=df["Velocity (m/s)"], color='blue', edgecolor='black')
plt.xlabel("Altitude (km)")
plt.ylabel("Velocity (m/s)")
plt.title("Altitude vs Velocity")
plt.grid(True)
plt.savefig("altitude_vs_velocity.png")  # Save before show
plt.show()

# Additional Visualization: Velocity vs. Mach Number
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Velocity (m/s)"], y=df["Mach Number"], color='red', edgecolor='black')
plt.xlabel("Velocity (m/s)")
plt.ylabel("Mach Number")
plt.title("Velocity vs Mach Number")
plt.grid(True)
plt.savefig("velocity_vs_mach.png")  # Save before show
plt.show()

# Additional Visualization: Trajectory Angle vs. Altitude
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Altitude (km)"], y=df["Trajectory Angle (°)"], color='green', edgecolor='black')
plt.xlabel("Altitude (km)")
plt.ylabel("Trajectory Angle (°)")
plt.title("Trajectory Angle vs Altitude")
plt.grid(True)
plt.savefig("trajectory_angle_vs_altitude.png")  # Save before show
plt.show()

print("Feature engineering and visualization completed successfully!")
