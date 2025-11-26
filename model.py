import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_prep import load_sales_data, create_features, prepare_train_test_split

def train_model(X_train, y_train):
    print("Training Gradient Boosting Regressor...")
    print("Parameters: n_estimators=100, learning_rate=0.1, max_depth=5")
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("Model training complete!")
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    
    print("\nTraining Set Performance:")
    print(f"  MAE (Mean Absolute Error): {train_mae:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {train_rmse:.4f}")
    print(f"  R² Score: {train_r2:.4f}")
    
    print("\nTest Set Performance:")
    print(f"  MAE (Mean Absolute Error): {test_mae:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {test_rmse:.4f}")
    print(f"  R² Score: {test_r2:.4f}")
    
    print("\n" + "="*50)
    
    return {
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    }

def save_model(model, filename='inventory_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {filename}")

def load_model(filename='inventory_model.pkl'):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_future_demand(model, product_id, days_ahead=7, df_with_features=None):
    if df_with_features is None:
        df = load_sales_data()
        df_with_features = create_features(df)
    
    product_data = df_with_features[df_with_features['ProductID'] == product_id].copy()
    product_data = product_data.sort_values(by='SaleDate')
    
    if len(product_data) == 0:
        return []
    
    last_row = product_data.iloc[-1]
    predictions = []
    
    feature_columns = [
        'ProductID', 'DayOfWeek', 'Month', 'WeekOfYear', 'DayOfMonth', 'Quarter',
        'Sales_Lag_7', 'Sales_Lag_14', 'Sales_Lag_30',
        'Sales_Rolling_7', 'Sales_Rolling_30'
    ]
    
    recent_sales = product_data['QuantitySold'].tail(30).tolist()
    
    from datetime import datetime, timedelta
    last_date = last_row['SaleDate']
    
    for day in range(1, days_ahead + 1):
        future_date = last_date + timedelta(days=day)
        
        features = {
            'ProductID': product_id,
            'DayOfWeek': future_date.dayofweek,
            'Month': future_date.month,
            'WeekOfYear': future_date.isocalendar()[1],
            'DayOfMonth': future_date.day,
            'Quarter': (future_date.month - 1) // 3 + 1,
            'Sales_Lag_7': recent_sales[-7] if len(recent_sales) >= 7 else recent_sales[-1],
            'Sales_Lag_14': recent_sales[-14] if len(recent_sales) >= 14 else recent_sales[-1],
            'Sales_Lag_30': recent_sales[-30] if len(recent_sales) >= 30 else recent_sales[-1],
            'Sales_Rolling_7': np.mean(recent_sales[-7:]) if len(recent_sales) >= 7 else np.mean(recent_sales),
            'Sales_Rolling_30': np.mean(recent_sales[-30:]) if len(recent_sales) >= 30 else np.mean(recent_sales)
        }
        
        X_future = np.array([[features[col] for col in feature_columns]])
        prediction = model.predict(X_future)[0]
        prediction = max(0, int(round(prediction)))
        
        predictions.append(prediction)
        recent_sales.append(prediction)
    
    return predictions

if __name__ == '__main__':
    print("Starting ML model training pipeline...")
    print("\n" + "="*50)
    
    print("Step 1: Loading and preprocessing data...")
    df = load_sales_data()
    df_with_features = create_features(df)
    X_train, X_test, y_train, y_test, train_df, test_df = prepare_train_test_split(df_with_features)
    
    print(f"\nDataset info:")
    print(f"  Total samples: {len(df_with_features)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    print("\n" + "="*50)
    print("Step 2: Training model...")
    model = train_model(X_train, y_train)
    
    print("\n" + "="*50)
    print("Step 3: Evaluating model...")
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    print("\n" + "="*50)
    print("Step 4: Saving model...")
    save_model(model)
    
    print("\n" + "="*50)
    print("Step 5: Testing prediction function...")
    print("\nPredicting demand for Product 1 (next 7 days):")
    predictions = predict_future_demand(model, product_id=1, days_ahead=7, df_with_features=df_with_features)
    for i, pred in enumerate(predictions, 1):
        print(f"  Day {i}: {pred} units")
    
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETE!")
    print("="*50)
