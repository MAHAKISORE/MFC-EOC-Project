import sqlite3
import pandas as pd
import os

# Ensures the database folder exists
os.makedirs("database", exist_ok=True)


df = pd.read_csv(r"C:\Users\MAHAKISORE\Desktop\ACADEMIC\SEM 2\MFC-2\MATLAB CODE\CICFlowMeter_Training_Balanced_added.csv")


conn = sqlite3.connect("database/data.db")

# Saves DataFrame to SQLite (replace if table exists)
df.to_sql("cic_flow_data", conn, if_exists="replace", index=False)

# Close connection
conn.close()

print("Your CICFlowMeter dataset has been stored successfully in the SQLite database!")
