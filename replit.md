# Smart Inventory Management System

## Overview

This is a machine learning-powered inventory management system that predicts product demand and automates restocking alerts. The application uses historical sales data (180 days) to train a Gradient Boosting model that forecasts future demand patterns, helping businesses maintain optimal stock levels and prevent stockouts.

The system provides a web-based dashboard for monitoring inventory levels, viewing predictions, and managing products in real-time.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Application Architecture

**Web Framework**: Flask-based monolithic application with a traditional MVC pattern
- Flask handles routing, templating, and API endpoints
- Single-file application structure (`app.py`) for the web server
- Standalone Python scripts for ML pipeline (`model.py`, `data_prep.py`, `db_setup.py`, `evaluate.py`)

**Rationale**: Flask provides a lightweight, simple framework suitable for ML-integrated applications. The separation of ML pipeline scripts from the web application allows for independent model training and evaluation without affecting the running web service.

### Frontend Architecture

**Technology Stack**: Server-side rendered HTML with client-side JavaScript
- Bootstrap 5 for responsive UI components
- Chart.js for data visualization
- Vanilla JavaScript for API consumption and DOM manipulation

**Rationale**: Server-side rendering keeps the frontend simple while Bootstrap and Chart.js provide professional-looking interfaces without complex build processes. This approach is ideal for data-heavy dashboards where the backend does the computational work.

### Data Storage

**Database**: SQLite relational database
- Schema includes 4 tables: Products, Inventory, Sales, Suppliers
- Foreign key relationships enforce data integrity
- Row factory enabled for dictionary-like access

**Rationale**: SQLite is embedded, requires no separate server, and is perfect for single-instance applications. The relational structure naturally models the business domain (products, suppliers, inventory, sales) with proper normalization.

### Machine Learning Pipeline

**Model**: Gradient Boosting Regressor (scikit-learn)
- Parameters: 100 estimators, 0.1 learning rate, max depth 5
- Trained on time-series features extracted from sales history
- Model persistence via pickle serialization

**Feature Engineering**:
- Time-based features: day of week, month, week of year, day of month, quarter
- Lag features: 7-day, 14-day, 30-day sales history
- Rolling averages: 7-day, 14-day, 30-day windows

**Rationale**: Gradient Boosting excels at capturing non-linear patterns in time-series data. The feature engineering approach combines temporal patterns (seasonality) with historical trends (lag features) to improve prediction accuracy. The target R² score of >0.75 indicates the model should explain at least 75% of demand variance.

### API Design

**REST API Endpoints**:
- `GET /api/products` - Retrieves product inventory data with JOIN queries
- Additional endpoints likely exist for predictions and restock alerts (referenced but not fully visible in provided code)

**Response Format**: JSON with structured product information including inventory levels, supplier details, and pricing

**Rationale**: RESTful API design separates data access from presentation, allowing future mobile apps or third-party integrations to consume the same backend services.

### Authentication & Security

**Session Management**: Flask sessions with configurable secret key
- Environment variable support: `SESSION_SECRET`
- Fallback development key for local testing

**Rationale**: Environment-based configuration allows different secrets for development and production without code changes.

## External Dependencies

### Core Libraries
- **Flask 3.0.0**: Web framework for HTTP server and routing
- **scikit-learn 1.3.2**: Machine learning library providing GradientBoostingRegressor
- **pandas 2.1.3**: Data manipulation and analysis for feature engineering
- **numpy 1.26.2**: Numerical computing for array operations and calculations

### Frontend Libraries (CDN)
- **Bootstrap 5.3.2**: CSS framework for responsive UI components
- **Chart.js 4.4.0**: JavaScript charting library for data visualizations

### Database
- **SQLite3**: Embedded relational database (Python standard library, no external dependency)

### Model Persistence
- **pickle**: Python serialization for saving/loading trained models (standard library)

### No External APIs or Cloud Services
The application operates entirely as a self-contained system with no external API calls, cloud services, or third-party integrations beyond the libraries listed above.