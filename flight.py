import pandas as pd  

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