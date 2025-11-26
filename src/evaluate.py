import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import pickle
import sqlite3
from datetime import datetime, timedelta

def load_model(filename='inventory_model.pkl'):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def load_database_data():
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

def evaluate_model_performance():
    print("="*60)
    print("INVENTORY MANAGEMENT SYSTEM - MODEL EVALUATION")
    print("="*60)
    
    print("\n1. Loading trained model...")
    try:
        model = load_model('inventory_model.pkl')
        print("   ✓ Model loaded successfully")
    except FileNotFoundError:
        print("   ✗ Model file not found. Please run train_model_kaggle.py first.")
        return
    
    print("\n2. Loading and preprocessing data...")
    df = load_database_data()
    df_with_features = create_features(df)
    X_train, X_test, y_train, y_test, train_df, test_df = prepare_train_test_split(df_with_features)
    print(f"   ✓ Data loaded: {len(X_train)} training, {len(X_test)} test samples")
    
    print("\n3. Making predictions...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    print(f"   ✓ Predictions completed in {prediction_time:.4f} seconds")
    
    print("\n4. Calculating evaluation metrics...")
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f"\nMean Absolute Error (MAE):        {mae:.4f} units")
    print(f"Root Mean Squared Error (RMSE):   {rmse:.4f} units")
    print(f"R² Score:                         {r2:.4f}")
    
    if r2 > 0.75:
        print(f"\n✓ EXCELLENT: Model achieves R² > 0.75 (target met!)")
    elif r2 > 0.60:
        print(f"\n✓ GOOD: Model achieves R² > 0.60")
    else:
        print(f"\n⚠ WARNING: Model R² is below 0.60, consider retraining")
    
    print("\n" + "="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        'Actual': y_test.values[:20],
        'Predicted': np.round(y_pred[:20]).astype(int),
        'Error': np.abs(y_test.values[:20] - y_pred[:20])
    })
    
    print("\nSample Predictions (First 20):")
    print(comparison_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("PERFORMANCE STATISTICS")
    print("="*60)
    
    errors = np.abs(y_test - y_pred)
    print(f"\nError Distribution:")
    print(f"  Mean Error:        {errors.mean():.4f} units")
    print(f"  Median Error:      {np.median(errors):.4f} units")
    print(f"  Max Error:         {errors.max():.4f} units")
    print(f"  Min Error:         {errors.min():.4f} units")
    
    within_1_unit = (errors <= 1).sum() / len(errors) * 100
    within_3_units = (errors <= 3).sum() / len(errors) * 100
    within_5_units = (errors <= 5).sum() / len(errors) * 100
    
    print(f"\nPrediction Accuracy:")
    print(f"  Within ±1 unit:    {within_1_unit:.1f}%")
    print(f"  Within ±3 units:   {within_3_units:.1f}%")
    print(f"  Within ±5 units:   {within_5_units:.1f}%")
    
    print("\n" + "="*60)
    print("QUERY PERFORMANCE TEST")
    print("="*60)
    
    test_queries = 100
    start_time = time.time()
    for _ in range(test_queries):
        _ = model.predict(X_test[:10])
    total_time = time.time() - start_time
    avg_time = (total_time / test_queries) * 1000
    
    print(f"\nQuery Performance ({test_queries} test queries):")
    print(f"  Total Time:        {total_time:.4f} seconds")
    print(f"  Average Time:      {avg_time:.4f} ms per query")
    print(f"  Queries per second: {test_queries / total_time:.2f}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'prediction_time': prediction_time,
        'avg_query_time': avg_time
    }

if __name__ == '__main__':
    evaluate_model_performance()
