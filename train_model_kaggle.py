import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

def download_kaggle_dataset():
    """Download dataset from Kaggle using API."""
    try:
        import kaggle
    except ImportError:
        print("Error: kaggle package not installed. Install: pip install kaggle")
        return None
    
    try:
        kaggle.api.dataset_download_files(
            'yukisim/sales-and-inventory-dataset',
            path='.',
            unzip=True
        )
        
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if not csv_files:
            print("Error: No CSV file found after download")
            return None
        
        return csv_files[0]
        
    except Exception as e:
        print(f"Error: {e}. Ensure Kaggle API token at ~/.kaggle/kaggle.json")
        return None

def train_model_on_kaggle_data(csv_path=None):
    """Train ML model on Kaggle dataset."""
    
    if csv_path is None:
        csv_path = download_kaggle_dataset()
        if csv_path is None:
            return False
    
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
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
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    print(f"Training on {len(X_train)} samples...")
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"Test Performance: R²={test_r2:.4f}, MAE={test_mae:.2f} units")
    
    with open('inventory_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    if test_r2 > 0.75:
        print(f"✓ Model ready (R² = {test_r2:.4f})")
    
    return True

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Use provided CSV path
        csv_path = sys.argv[1]
        if not os.path.exists(csv_path):
            print(f"Error: File not found: {csv_path}")
            sys.exit(1)
        success = train_model_on_kaggle_data(csv_path)
    else:
        # Auto-download from Kaggle
        success = train_model_on_kaggle_data()
    
    if not success:
        print("Model training failed.")
        sys.exit(1)
