import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from datetime import datetime, timedelta

def train_model_on_kaggle_data(csv_path):
    """
    Train ML model on Kaggle dataset without modifying the database.
    This allows using real data for model training while keeping sample data in the database.
    """
    
    print("="*60)
    print("TRAINING MODEL ON KAGGLE DATA")
    print("="*60)
    
    print(f"\nLoading Kaggle dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Map CSV columns to our expected format
    df_renamed = df.copy()
    
    # Standardize column names
    column_mapping = {
        'product_name': 'ProductName',
        'Product_name': 'ProductName',
        'product': 'ProductName',
        'Product': 'ProductName',
        'product_id': 'ProductID',
        'Product_ID': 'ProductID',
        'product_category': 'Category',
        'Product_category': 'Category',
        'category': 'Category',
        'Category': 'Category',
        'price': 'Price',
        'Price': 'Price',
        'unit_price': 'Price',
        'quantity': 'Quantity',
        'quantity_sold': 'Quantity',
        'Quantity': 'Quantity',
        'order_date': 'SaleDate',
        'Order_date': 'SaleDate',
        'sale_date': 'SaleDate',
        'SaleDate': 'SaleDate',
        'date': 'SaleDate',
        'Date': 'SaleDate',
    }
    
    df_renamed = df_renamed.rename(columns=column_mapping)
    
    required_cols = ['ProductID', 'SaleDate', 'Quantity']
    for col in required_cols:
        if col not in df_renamed.columns:
            print(f"Error: Required column '{col}' not found in dataset")
            print(f"Available columns: {df_renamed.columns.tolist()}")
            return False
    
    # Convert date column
    df_renamed['SaleDate'] = pd.to_datetime(df_renamed['SaleDate'])
    
    # Create features for ML model
    print("\nCreating features...")
    df_with_features = df_renamed.copy()
    
    df_with_features['DayOfWeek'] = df_with_features['SaleDate'].dt.dayofweek
    df_with_features['Month'] = df_with_features['SaleDate'].dt.month
    df_with_features['WeekOfYear'] = df_with_features['SaleDate'].dt.isocalendar().week
    df_with_features['DayOfMonth'] = df_with_features['SaleDate'].dt.day
    df_with_features['Quarter'] = df_with_features['SaleDate'].dt.quarter
    
    # Create lag and rolling features per product
    product_dfs = []
    for product_id in df_with_features['ProductID'].unique():
        product_df = df_with_features[df_with_features['ProductID'] == product_id].copy()
        product_df = product_df.sort_values(by='SaleDate')
        
        product_df['Sales_Lag_7'] = product_df['Quantity'].shift(7)
        product_df['Sales_Lag_14'] = product_df['Quantity'].shift(14)
        product_df['Sales_Lag_30'] = product_df['Quantity'].shift(30)
        
        product_df['Sales_Rolling_7'] = product_df['Quantity'].rolling(window=7, min_periods=1).mean()
        product_df['Sales_Rolling_30'] = product_df['Quantity'].rolling(window=30, min_periods=1).mean()
        
        product_dfs.append(product_df)
    
    df_with_features = pd.concat(product_dfs, ignore_index=True)
    df_with_features = df_with_features.bfill().fillna(0)
    
    # Train/test split
    df_with_features = df_with_features.sort_values(by='SaleDate')
    split_index = int(len(df_with_features) * 0.75)
    
    train_df = df_with_features.iloc[:split_index]
    test_df = df_with_features.iloc[split_index:]
    
    feature_columns = [
        'ProductID', 'DayOfWeek', 'Month', 'WeekOfYear', 'DayOfMonth', 'Quarter',
        'Sales_Lag_7', 'Sales_Lag_14', 'Sales_Lag_30',
        'Sales_Rolling_7', 'Sales_Rolling_30'
    ]
    
    X_train = train_df[feature_columns]
    y_train = train_df['Quantity']
    X_test = test_df[feature_columns]
    y_test = test_df['Quantity']
    
    print(f"\nDataset Info:")
    print(f"  Total samples: {len(df_with_features)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Unique products: {df_with_features['ProductID'].nunique()}")
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING GRADIENT BOOSTING MODEL")
    print("="*60)
    print("Parameters: n_estimators=100, learning_rate=0.1, max_depth=5")
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("Model training complete!")
    
    # Evaluate
    print("\n" + "="*60)
    print("MODEL EVALUATION METRICS")
    print("="*60)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\nTraining Set Performance:")
    print(f"  MAE (Mean Absolute Error): {train_mae:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {train_rmse:.4f}")
    print(f"  R² Score: {train_r2:.4f}")
    
    print("\nTest Set Performance:")
    print(f"  MAE (Mean Absolute Error): {test_mae:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {test_rmse:.4f}")
    print(f"  R² Score: {test_r2:.4f}")
    
    if test_r2 > 0.75:
        print(f"\n✓ EXCELLENT: Model achieves R² > 0.75 (target met!)")
    elif test_r2 > 0.60:
        print(f"\n✓ GOOD: Model achieves R² > 0.60")
    else:
        print(f"\n⚠ WARNING: Model R² is below 0.60")
    
    # Save model
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    with open('inventory_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to inventory_model.pkl")
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE!")
    print("="*60)
    
    return True

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage: python train_model_kaggle.py <path_to_kaggle_csv>")
        print("\nExample: python train_model_kaggle.py sales_inventory.csv")
        print("\nThis script trains the ML model on Kaggle data without modifying the database.")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    success = train_model_on_kaggle_data(csv_path)
    
    if success:
        print("\nNext steps:")
        print("1. The model has been trained on Kaggle data")
        print("2. Your database still contains the sample data")
        print("3. Run: python app.py to start the dashboard")
        print("4. The model predictions will use Kaggle-trained patterns with sample data")
