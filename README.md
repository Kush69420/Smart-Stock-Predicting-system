# Smart Inventory Management System with ML-Based Predictive Restocking

A complete database-driven inventory management system that uses Gradient Boosting Machine Learning to predict product demand and automate restocking alerts.

## Features

- **ML-Powered Demand Prediction**: Uses GradientBoostingRegressor to forecast future product demand
- **Automated Restock Alerts**: Triggers alerts when inventory falls below reorder points
- **Real-time Dashboard**: Responsive web interface with Bootstrap and Chart.js visualizations
- **REST API**: Complete API for inventory management operations
- **Historical Analysis**: 180 days of sales data with seasonal patterns and trends

## Technical Stack

- **Backend**: Python Flask
- **Database**: SQLite
- **ML Library**: scikit-learn (GradientBoostingRegressor)
- **Data Processing**: pandas, numpy
- **Frontend**: HTML, CSS, Bootstrap 5, Chart.js

## Database Schema

### Tables
1. **Products**: ProductID, ProductName, Category, UnitPrice, SupplierID
2. **Inventory**: InventoryID, ProductID, QuantityAvailable, MinimumStockLevel, ReorderPoint, LastUpdated
3. **Sales**: SaleID, ProductID, SaleDate, QuantitySold, TotalAmount
4. **Suppliers**: SupplierID, SupplierName, ContactInfo, Email

## Installation & Setup

### 1. Install Dependencies
All dependencies are already installed in this Replit environment.

### 2. Initialize Database
```bash
python db_setup.py
```
This creates the SQLite database, tables, and generates 180 days of realistic sales data for 10 products.

### 3. Preprocess Data
```bash
python data_prep.py
```
Extracts features including time-based features and sales lag features.

### 4. Train ML Model
```bash
python model.py
```
Trains the GradientBoostingRegressor model and saves it as `inventory_model.pkl`.

### 5. Evaluate Model
```bash
python evaluate.py
```
Displays model performance metrics (MAE, RMSE, R² Score).

### 6. Run Flask Application
```bash
python app.py
```
Access the dashboard at http://localhost:5000

## API Endpoints

### GET /api/products
Returns all products with current inventory status.

### GET /api/restock-alerts
Returns products below reorder point requiring restocking.

### POST /api/predict-demand
Predicts demand for a specific product.

**Request Body:**
```json
{
  "product_id": 1,
  "days_ahead": 7
}
```

**Response:**
```json
{
  "success": true,
  "product_id": 1,
  "product_name": "Wireless Mouse",
  "predictions": [23, 25, 22, 24, 26, 23, 21]
}
```

### GET /api/sales-history/<product_id>
Returns sales history for a specific product.

### POST /api/add-sale
Adds a new sale transaction and updates inventory.

**Request Body:**
```json
{
  "product_id": 1,
  "quantity_sold": 5,
  "sale_date": "2025-11-26"
}
```

### GET /api/dashboard-stats
Returns summary statistics for the dashboard.

## Model Performance

The Gradient Boosting model is trained with the following parameters:
- **n_estimators**: 100
- **learning_rate**: 0.1
- **max_depth**: 5

### Expected Metrics
- **R² Score**: > 0.75 (target achieved)
- **MAE**: Low mean absolute error for accurate predictions
- **RMSE**: Low root mean squared error

## Dashboard Features

1. **Summary Cards**: Total products, low stock alerts, weekly sales
2. **Restock Alerts Table**: Products requiring immediate restocking
3. **Demand Prediction Tool**: Select product and predict future demand
4. **Visualization Chart**: Line chart comparing predicted vs actual demand
5. **Top Products**: Best-selling products this week
6. **Full Inventory Table**: Complete product and stock status overview

## File Structure

```
project/
├── app.py                  # Flask application
├── db_setup.py            # Database initialization
├── data_prep.py           # Data preprocessing
├── model.py               # ML model training
├── evaluate.py            # Model evaluation
├── inventory_model.pkl    # Trained model (generated)
├── inventory.db           # SQLite database (generated)
├── requirements.txt       # Dependencies
├── README.md             # Documentation
├── templates/
│   └── index.html        # Dashboard UI
└── static/
    └── style.css         # Custom styles
```

## Model Features

The ML model uses the following features for prediction:
- **ProductID**: Unique product identifier
- **DayOfWeek**: Day of the week (0-6)
- **Month**: Month of the year (1-12)
- **WeekOfYear**: Week number in the year
- **DayOfMonth**: Day of the month
- **Quarter**: Quarter of the year (1-4)
- **Sales_Lag_7**: Sales from 7 days ago
- **Sales_Lag_14**: Sales from 14 days ago
- **Sales_Lag_30**: Sales from 30 days ago
- **Sales_Rolling_7**: 7-day rolling average
- **Sales_Rolling_30**: 30-day rolling average

## Sample Data

The system includes sample data for:
- **10 Products**: Electronics and Office Supplies
- **5 Suppliers**: Various suppliers with contact information
- **180 Days**: Realistic sales history with seasonal patterns

## Usage Example

1. Access the dashboard at http://localhost:5000
2. View summary statistics and low stock alerts
3. Select a product from the dropdown
4. Click "Predict Demand" to see 7-day forecast
5. View the prediction chart comparing forecasts with historical data
6. Monitor restock alerts and contact suppliers as needed

## Testing

Run the evaluation script to verify model performance:
```bash
python evaluate.py
```

This will display:
- Model accuracy metrics
- Prediction samples
- Query performance statistics
- Error distribution analysis

## Future Enhancements

- Email notifications to suppliers for restock alerts
- User authentication and role-based access control
- Advanced filtering and search capabilities
- Data export functionality (CSV/Excel)
- Multi-product batch predictions
- Integration with supplier APIs for automated ordering

## License

This project is for educational and demonstration purposes.
