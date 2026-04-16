import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv('data/german_credit.csv')
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

engine = create_engine('postgresql+psycopg2://localhost/creditsense')

df.to_sql('credit_data', engine, if_exists='append', index=False)

print("\n✅ Data loaded into PostgreSQL successfully!")
print(f"   {len(df)} rows inserted into credit_data table.")