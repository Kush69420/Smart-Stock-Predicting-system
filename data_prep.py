import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_sales_data():
    conn = sqlite3.connect('inventory.db')
    
    query = '''
        SELECT 
            s.SaleID,
            s.ProductID,
            s.SaleDate,
            s.QuantitySold,
            s.TotalAmount,
            p.ProductName,
            p.Category,
            p.UnitPrice
        FROM Sales s
        JOIN Products p ON s.ProductID = p.ProductID
        ORDER BY s.SaleDate
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['SaleDate'] = pd.to_datetime(df['SaleDate'])
    
    return df

def create_features(df):
    df = df.copy()
    
    df['DayOfWeek'] = df['SaleDate'].dt.dayofweek
    df['Month'] = df['SaleDate'].dt.month
    df['WeekOfYear'] = df['SaleDate'].dt.isocalendar().week
    df['DayOfMonth'] = df['SaleDate'].dt.day
    df['Quarter'] = df['SaleDate'].dt.quarter
    
    product_dfs = []
    
    for product_id in df['ProductID'].unique():
        product_df = df[df['ProductID'] == product_id].copy()
        product_df = product_df.sort_values(by='SaleDate')
        
        product_df['Sales_Lag_7'] = product_df['QuantitySold'].shift(7)
        product_df['Sales_Lag_14'] = product_df['QuantitySold'].shift(14)
        product_df['Sales_Lag_30'] = product_df['QuantitySold'].shift(30)
        
        product_df['Sales_Rolling_7'] = product_df['QuantitySold'].rolling(window=7, min_periods=1).mean()
        product_df['Sales_Rolling_30'] = product_df['QuantitySold'].rolling(window=30, min_periods=1).mean()
        
        product_dfs.append(product_df)
    
    df_with_features = pd.concat(product_dfs, ignore_index=True)
    
    df_with_features = df_with_features.bfill().fillna(0)
    
    return df_with_features

def prepare_train_test_split(df, test_size=0.25):
    df = df.sort_values(by='SaleDate')
    
    split_index = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    feature_columns = [
        'ProductID', 'DayOfWeek', 'Month', 'WeekOfYear', 'DayOfMonth', 'Quarter',
        'Sales_Lag_7', 'Sales_Lag_14', 'Sales_Lag_30',
        'Sales_Rolling_7', 'Sales_Rolling_30'
    ]
    
    X_train = train_df[feature_columns]
    y_train = train_df['QuantitySold']
    
    X_test = test_df[feature_columns]
    y_test = test_df['QuantitySold']
    
    return X_train, X_test, y_train, y_test, train_df, test_df

if __name__ == '__main__':
    print("Loading sales data from database...")
    df = load_sales_data()
    print(f"Loaded {len(df)} sales records")
    
    print("\nCreating features (time-based and lag features)...")
    df_with_features = create_features(df)
    print(f"Created features: {df_with_features.columns.tolist()}")
    
    print("\nSplitting data into train/test (75/25)...")
    X_train, X_test, y_train, y_test, train_df, test_df = prepare_train_test_split(df_with_features)
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"\nFeature columns: {X_train.columns.tolist()}")
    print(f"\nTarget variable stats:")
    print(f"  Mean: {y_train.mean():.2f}")
    print(f"  Std: {y_train.std():.2f}")
    print(f"  Min: {y_train.min()}")
    print(f"  Max: {y_train.max()}")
    
    print("\nData preprocessing complete!")
