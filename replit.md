# Smart Inventory Management System

## Overview

A production-ready ML-powered inventory management system that predicts product demand using Gradient Boosting Machine Learning and automates restocking alerts. The system processes sales data to forecast 7-day demand with 80%+ accuracy (R² = 0.8094), helping businesses optimize inventory levels and prevent stockouts. Built with Flask and SQLite, it features a responsive dark-mode dashboard with real-time analytics, automated reorder alerts, and comprehensive inventory tracking.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture

**Web Framework**: Flask 3.0.0 serves as the lightweight web server, handling HTTP requests and rendering templates. The architecture follows a traditional MVC pattern with route handlers in `app.py`, HTML templates in the `templates/` directory, and static assets in `static/`.

**Database Design**: SQLite3 embedded database with a normalized relational schema:
- **Suppliers Table**: Stores supplier information (name, contact, email)
- **Products Table**: Product catalog with category, pricing, and supplier relationships
- **Inventory Table**: Tracks stock levels, minimum thresholds, reorder points, and last update timestamps
- **Sales Table**: Transaction history linking products to quantities sold and revenue

Foreign key relationships enforce referential integrity between tables (Products→Suppliers, Inventory→Products, Sales→Products).

**Machine Learning Pipeline**: 
- **Model Type**: scikit-learn GradientBoostingRegressor with 100 estimators
- **Feature Engineering**: Creates 11 temporal features from sales data including day-of-week patterns, monthly seasonality, rolling averages (7/14/30 days), and lag features
- **Training Strategy**: 75/25 train-test split on historical sales data
- **Performance**: Achieves R² = 0.8094, MAE = 1.89 units, 857 predictions/second
- **Persistence**: Model serialized using pickle and loaded at runtime

**Data Processing**: pandas and numpy handle data transformation, aggregation, and feature engineering. Time-series operations convert sale dates into predictive features capturing seasonal patterns and trends.

**API Design**: RESTful endpoints provide CRUD operations for inventory management, sales recording, and demand predictions. JSON responses enable frontend integration and potential third-party consumption.

### Frontend Architecture

**UI Framework**: Bootstrap 5.3.2 provides responsive grid layout and pre-built components. Custom CSS in `static/style.css` implements dark/light theme switching using CSS custom properties.

**Theme System**: JavaScript-based theme toggle persists user preference in localStorage. Root-level CSS variables enable instant theme switching without page reload.

**Data Visualization**: Chart.js 4.4.0 renders interactive charts for sales trends, demand forecasts, and inventory analytics. Charts dynamically update based on API responses.

**Localization**: Indian Rupee formatting (₹) with proper number grouping for financial displays (e.g., ₹16,87,532.00).

### Machine Learning Model Architecture

**Problem Solved**: Predicts future product demand to prevent stockouts and optimize inventory levels. Traditional manual forecasting is time-consuming and error-prone; ML automates this with higher accuracy.

**Chosen Solution**: Gradient Boosting Regressor selected for its ability to capture non-linear relationships in time-series data and handle multiple feature types without extensive preprocessing.

**Alternatives Considered**: 
- Linear Regression: Simpler but cannot capture seasonal patterns effectively
- LSTM Neural Networks: More powerful but requires larger datasets and longer training time
- ARIMA: Good for univariate time-series but struggles with multiple product categories

**Pros**: High accuracy (80%+), fast inference, works with limited data (1,800 records), captures complex patterns
**Cons**: Requires feature engineering, less interpretable than linear models, sensitive to data quality

### Data Flow

1. **Sales Recording**: User submits transaction → Flask endpoint validates → Database insert → Inventory auto-updated
2. **Demand Prediction**: Feature extraction from historical sales → Model inference → 7-day forecast generated → Restock alerts calculated
3. **Dashboard Rendering**: Backend aggregates database statistics → Template receives data → Chart.js visualizes trends → User sees real-time insights

### Key Design Decisions

**SQLite Over PostgreSQL**: Embedded database eliminates server setup complexity, ideal for small-to-medium deployments. Enables zero-configuration deployment on Replit. Trade-off: Limited concurrent write scalability (acceptable for inventory systems with moderate transaction volumes).

**Pickle Model Serialization**: Standard Python serialization enables model versioning and quick loading. Model retraining workflow: `train_model_kaggle.py` generates new `.pkl` file → Replace existing model → Restart server.

**Session-Based Security**: Flask sessions with secret key provide basic authentication foundation. Production deployment requires stronger authentication (OAuth, JWT) for multi-user scenarios.

**Responsive Dark Theme**: Improves user experience in warehouse/retail environments with varying lighting conditions. CSS custom properties enable maintainable theming without duplicating styles.

## External Dependencies

### Python Libraries
- **flask==3.0.0**: Web framework for HTTP routing and template rendering
- **pandas==2.1.3**: Data manipulation and time-series operations
- **numpy==1.26.2**: Numerical computing for feature engineering
- **scikit-learn==1.3.2**: Machine learning model (GradientBoostingRegressor) and evaluation metrics

### Frontend Libraries (CDN-hosted)
- **Bootstrap 5.3.2**: UI components and responsive grid system
- **Chart.js 4.4.0**: Interactive data visualization
- **Google Fonts (Inter)**: Typography for modern UI

### Database
- **SQLite3**: Embedded relational database (included in Python standard library, no external server required)

### Optional External Data Sources
- **Kaggle Datasets**: `train_model_kaggle.py` supports training on external CSV files for improved model accuracy with real-world data while maintaining sample database for testing

### Deployment Platforms
- **Replit**: Primary deployment target (zero-config SQLite compatibility)
- **Local Machine**: Standard Python environment with requirements.txt installation

### File Dependencies
- **inventory_model.pkl**: Serialized ML model loaded at runtime (generated by training scripts)
- **inventory.db**: SQLite database file (auto-created by db_setup.py)