import pandas as pd

# Load dataset
file_path = "D:/ml_1/housing.csv"  # Change the filename if needed
df = pd.read_csv(file_path)

# Display first 5 rows
print(df.head())

print(df.isnull().sum())

print(df.info())

for col in df.columns:
    print(f"{col}: {df[col].unique()[:5]}")  # Show first 5 unique values

cat_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus"]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder

for col in cat_cols:
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])

print(df.head())



