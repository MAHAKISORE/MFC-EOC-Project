import sqlite3
import pandas as pd
import os

# Create the "database" folder if it doesn't exist
os.makedirs("database", exist_ok=True)

# Sample dataset
data = {
    'feature1': [10, 20, 30, 40, 50],
    'feature2': [1.2, 3.4, 2.1, 0.9, 5.5],
    'feature3': [100, 200, 150, 300, 250],
    'target': [0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Create and save the database
conn = sqlite3.connect("database/data.db")  # Save it inside the database/ folder
df.to_sql("sample_data", conn, if_exists="replace", index=False)
conn.close()

print("✅ SQLite3 database created successfully!")
