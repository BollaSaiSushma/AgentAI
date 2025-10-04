import pandas as pd
import re

# Load dataset
df = pd.read_csv(r"C:\Users\saisu\OneDrive\Desktop\AgentAI\AgentAI\Day-1\DirtyDatasetforDataCleaning.csv")

# Fix column names (remove spaces, non-breaking spaces, special chars)
df.columns = df.columns.str.replace("\xa0", "_", regex=True)
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(r"[^\w\s]", "", regex=True)

# Function to clean money/gross values
def clean_money(value):
    if pd.isna(value):
        return None
    return float(re.sub(r"[^\d.]", "", str(value)))

# Apply cleaning to numeric/money columns
for col in ["Actual_gross", "Adjusted_gross_in_2022_dollars", "Average_gross"]:
    df[col] = df[col].apply(clean_money)

# Convert other numeric-like columns
df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
df["All_Time_Peak"] = df["All_Time_Peak"].astype(str).str.extract(r"(\d+)").astype(float)
df["Shows"] = pd.to_numeric(df["Shows"], errors="coerce")

# Drop unnecessary columns
if "Ref" in df.columns:
    df = df.drop(columns=["Ref"])

# Save cleaned dataset
df.to_csv("CleanedDataset.csv", index=False)

print("Data cleaned and saved as CleanedDataset.csv")
print(df.head())
