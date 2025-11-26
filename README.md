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

## Installation & Setup (Local Machine)

### Prerequisites

Before you begin, ensure you have the following installed on your device:
- **Python 3.8 or higher** - Download from https://www.python.org/downloads/
- **pip** (Python package manager - usually comes with Python)
- **Git** (optional, for cloning the repository)

### Step 1: Get the Project Files

**Option A: Clone via Git**
```bash
git clone https://github.com/Kush69420/Smart-Stock-Predicting-system
cd smart-inventory-system
```

**Option B: Download as ZIP**
- Download the project files and extract them to a folder
- Open terminal/command prompt and navigate to the project folder

### Step 2: Create a Virtual Environment

It's recommended to use a virtual environment to avoid conflicts with other Python projects.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt after activation.

### Step 3: Install Dependencies

With the virtual environment activated, install all required packages:
```bash
pip install -r requirements.txt
```

This installs:
- Flask 3.0.0
- scikit-learn 1.3.2
- pandas 2.1.3
- numpy 1.26.2

### Step 4: Initialize the Database

Create the SQLite database with sample data:
```bash
python db_setup.py
```

**Expected output:**
```
Database created successfully!
Inserted 10 products
Inserted 5 suppliers
Generated 180 days of sales data
```

This creates a file called `inventory.db` in your project folder with sample inventory data for the web interface.

### Step 5: Train the ML Model

Choose your training data source:

#### Quick Option: Train on Sample Database Data
```bash
python train_model_kaggle.py
```

#### Production Option: Train on Real Kaggle Data (Recommended)

1. Download from Kaggle:
   - Go to: https://www.kaggle.com/datasets/yukisim/sales-and-inventory-dataset
   - Click "Download" (requires free Kaggle account)
   - Extract the CSV file

2. Train on the Kaggle data:
   ```bash
   python train_model_kaggle.py sales_inventory.csv
   ```

**Expected output:**
```
TRAINING MODEL ON KAGGLE DATA
Loading Kaggle dataset from: sales_inventory.csv
Dataset shape: [rows, columns]
...
TRAINING GRADIENT BOOSTING MODEL
Model training complete!

TEST SET PERFORMANCE:
  MAE: [value]
  RMSE: [value]
  R² Score: [value]

✓ EXCELLENT: Model achieves R² > 0.75 (target met!)
Model saved to inventory_model.pkl
```

This creates `inventory_model.pkl` with your trained model.

### Step 6: (Optional) Evaluate Model Performance

View detailed model accuracy metrics:
```bash
python evaluate.py
```

**This shows:**
- R² Score (target: > 0.75)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Prediction samples
- Error distribution analysis

### Step 7: Run the Flask Application

Start the web server:
```bash
python app.py
```

**Expected output:**
```
* Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

### Step 8: Access the Dashboard

Open your web browser and navigate to:
```
http://localhost:5000
```

You should see the Smart Inventory Dashboard with:
- Summary cards (Total Products, Low Stock Alerts, Weekly Sales)
- Restock alerts table
- Demand prediction tool
- Top products chart
- Full inventory table

### Quick Start (Local Machine)

For quick reference, here's the complete sequence of commands:

```bash
# 1. Navigate to project folder
cd path/to/smart-inventory-system

# 2. Create and activate virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup database
python db_setup.py

# 5. Train model (choose one):
# Option A: Sample data
python train_model_kaggle.py

# Option B: Real Kaggle data
python train_model_kaggle.py sales_inventory.csv

# 6. Run application
python app.py

# 7. Open http://localhost:5000 in your browser
```

## Troubleshooting

### Issue: "Python command not found"
**Solution**: Python may not be in your system PATH. Try:
- Windows: Use `python` or `py`
- macOS/Linux: Use `python3`

### Issue: "No module named 'flask'" after pip install
**Solution**: Make sure your virtual environment is activated (look for `(venv)` in terminal)

### Issue: "Port 5000 already in use"
**Solution**: Flask is already running or another app uses port 5000
- Stop other applications using port 5000
- Or modify `app.py` to use a different port (change `port=5000` to `port=5001`)

### Issue: "inventory.db not found" error
**Solution**: Run `python db_setup.py` first to create the database

### Issue: "inventory_model.pkl not found" error
**Solution**: Run `python model.py` to train and save the model

### Issue: "ModuleNotFoundError" when running scripts
**Solution**: Ensure requirements are installed: `pip install -r requirements.txt`

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
├── db_setup.py            # Database initialization & sample data
├── train_model_kaggle.py   # ML model training (supports CSV or database)
├── import_kaggle_dataset.py# (Optional) Import CSV data into database
├── evaluate.py            # Model evaluation & metrics
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
