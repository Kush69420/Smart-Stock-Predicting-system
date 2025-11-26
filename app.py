from flask import Flask, render_template, jsonify, request
import sqlite3
import pickle
from datetime import datetime, timedelta
import os
from model import predict_future_demand, load_model
from data_prep import load_sales_data, create_features

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'dev-secret-key-change-in-production')

def get_db_connection():
    conn = sqlite3.connect('inventory.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/products', methods=['GET'])
def get_products():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                p.ProductID,
                p.ProductName,
                p.Category,
                p.UnitPrice,
                i.QuantityAvailable,
                i.MinimumStockLevel,
                i.ReorderPoint,
                i.LastUpdated,
                s.SupplierName
            FROM Products p
            LEFT JOIN Inventory i ON p.ProductID = i.ProductID
            LEFT JOIN Suppliers s ON p.SupplierID = s.SupplierID
            ORDER BY p.ProductID
        ''')
        
        products = []
        for row in cursor.fetchall():
            products.append({
                'product_id': row['ProductID'],
                'product_name': row['ProductName'],
                'category': row['Category'],
                'unit_price': row['UnitPrice'],
                'quantity_available': row['QuantityAvailable'],
                'minimum_stock_level': row['MinimumStockLevel'],
                'reorder_point': row['ReorderPoint'],
                'last_updated': row['LastUpdated'],
                'supplier_name': row['SupplierName']
            })
        
        conn.close()
        return jsonify({'success': True, 'products': products})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/restock-alerts', methods=['GET'])
def get_restock_alerts():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                p.ProductID,
                p.ProductName,
                p.Category,
                i.QuantityAvailable,
                i.ReorderPoint,
                i.MinimumStockLevel,
                s.SupplierName,
                s.Email as SupplierEmail
            FROM Products p
            JOIN Inventory i ON p.ProductID = i.ProductID
            JOIN Suppliers s ON p.SupplierID = s.SupplierID
            WHERE i.QuantityAvailable <= i.ReorderPoint
            ORDER BY (i.QuantityAvailable - i.ReorderPoint) ASC
        ''')
        
        alerts = []
        for row in cursor.fetchall():
            alerts.append({
                'product_id': row['ProductID'],
                'product_name': row['ProductName'],
                'category': row['Category'],
                'quantity_available': row['QuantityAvailable'],
                'reorder_point': row['ReorderPoint'],
                'minimum_stock_level': row['MinimumStockLevel'],
                'supplier_name': row['SupplierName'],
                'supplier_email': row['SupplierEmail'],
                'deficit': row['ReorderPoint'] - row['QuantityAvailable']
            })
        
        conn.close()
        return jsonify({'success': True, 'alerts': alerts})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict-demand', methods=['POST'])
def predict_demand():
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        days_ahead = data.get('days_ahead', 7)
        
        if not product_id:
            return jsonify({'success': False, 'error': 'product_id is required'}), 400
        
        if not os.path.exists('inventory_model.pkl'):
            return jsonify({'success': False, 'error': 'Model not found. Please train the model first.'}), 404
        
        model = load_model('inventory_model.pkl')
        
        df = load_sales_data()
        df_with_features = create_features(df)
        
        predictions = predict_future_demand(model, product_id, days_ahead, df_with_features)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT ProductName FROM Products WHERE ProductID = ?', (product_id,))
        result = cursor.fetchone()
        conn.close()
        
        product_name = result['ProductName'] if result else 'Unknown'
        
        return jsonify({
            'success': True,
            'product_id': product_id,
            'product_name': product_name,
            'days_ahead': days_ahead,
            'predictions': predictions
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/sales-history/<int:product_id>', methods=['GET'])
def get_sales_history(product_id):
    try:
        days = request.args.get('days', 30, type=int)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                s.SaleDate,
                s.QuantitySold,
                s.TotalAmount,
                p.ProductName
            FROM Sales s
            JOIN Products p ON s.ProductID = p.ProductID
            WHERE s.ProductID = ?
            ORDER BY s.SaleDate DESC
            LIMIT ?
        ''', (product_id, days))
        
        sales = []
        for row in cursor.fetchall():
            sales.append({
                'sale_date': row['SaleDate'],
                'quantity_sold': row['QuantitySold'],
                'total_amount': row['TotalAmount'],
                'product_name': row['ProductName']
            })
        
        conn.close()
        return jsonify({'success': True, 'sales': sales})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/recent-sales', methods=['GET'])
def get_recent_sales():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                s.SaleID,
                s.SaleDate,
                s.QuantitySold,
                s.TotalAmount,
                p.ProductName
            FROM Sales s
            JOIN Products p ON s.ProductID = p.ProductID
            ORDER BY s.SaleDate DESC
            LIMIT 50
        ''')
        
        sales = []
        for row in cursor.fetchall():
            sales.append({
                'sale_id': row['SaleID'],
                'sale_date': row['SaleDate'],
                'quantity_sold': row['QuantitySold'],
                'total_amount': row['TotalAmount'],
                'product_name': row['ProductName']
            })
        
        conn.close()
        return jsonify({'success': True, 'sales': sales})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/delete-sale', methods=['POST'])
def delete_sale():
    try:
        data = request.get_json()
        sale_id = data.get('sale_id')
        
        if not sale_id:
            return jsonify({'success': False, 'error': 'sale_id is required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT ProductID, QuantitySold FROM Sales WHERE SaleID = ?', (sale_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return jsonify({'success': False, 'error': 'Sale not found'}), 404
        
        product_id = result['ProductID']
        quantity_sold = result['QuantitySold']
        
        cursor.execute('DELETE FROM Sales WHERE SaleID = ?', (sale_id,))
        
        cursor.execute(
            'UPDATE Inventory SET QuantityAvailable = QuantityAvailable + ?, LastUpdated = ? WHERE ProductID = ?',
            (quantity_sold, datetime.now(), product_id)
        )
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Sale deleted and inventory restored'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/add-sale', methods=['POST'])
def add_sale():
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        quantity_sold = data.get('quantity_sold')
        sale_date = data.get('sale_date', datetime.now().date().isoformat())
        
        if not product_id or not quantity_sold:
            return jsonify({'success': False, 'error': 'product_id and quantity_sold are required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT UnitPrice FROM Products WHERE ProductID = ?', (product_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return jsonify({'success': False, 'error': 'Product not found'}), 404
        
        unit_price = result['UnitPrice']
        total_amount = quantity_sold * unit_price
        
        cursor.execute(
            'INSERT INTO Sales (ProductID, SaleDate, QuantitySold, TotalAmount) VALUES (?, ?, ?, ?)',
            (product_id, sale_date, quantity_sold, total_amount)
        )
        
        cursor.execute(
            'UPDATE Inventory SET QuantityAvailable = QuantityAvailable - ?, LastUpdated = ? WHERE ProductID = ?',
            (quantity_sold, datetime.now(), product_id)
        )
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Sale added successfully',
            'total_amount': total_amount
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/add-purchase', methods=['POST'])
def add_purchase():
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        quantity_purchased = data.get('quantity_purchased')
        
        if not product_id or not quantity_purchased:
            return jsonify({'success': False, 'error': 'product_id and quantity_purchased are required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT ProductID FROM Products WHERE ProductID = ?', (product_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'error': 'Product not found'}), 404
        
        cursor.execute(
            'UPDATE Inventory SET QuantityAvailable = QuantityAvailable + ?, LastUpdated = ? WHERE ProductID = ?',
            (quantity_purchased, datetime.now(), product_id)
        )
        
        cursor.execute('SELECT QuantityAvailable FROM Inventory WHERE ProductID = ?', (product_id,))
        new_quantity = cursor.fetchone()['QuantityAvailable']
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Stock updated successfully',
            'product_id': product_id,
            'quantity_added': quantity_purchased,
            'new_quantity': new_quantity
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/suppliers', methods=['GET'])
def get_suppliers():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT SupplierID, SupplierName FROM Suppliers ORDER BY SupplierName')
        suppliers = []
        for row in cursor.fetchall():
            suppliers.append({
                'supplier_id': row['SupplierID'],
                'supplier_name': row['SupplierName']
            })
        
        conn.close()
        return jsonify({'success': True, 'suppliers': suppliers})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/add-product', methods=['POST'])
def add_product():
    try:
        data = request.get_json()
        product_name = data.get('product_name')
        category = data.get('category')
        unit_price = data.get('unit_price')
        supplier_id = data.get('supplier_id')
        initial_quantity = data.get('initial_quantity', 100)
        min_stock_level = data.get('min_stock_level', 50)
        reorder_point = data.get('reorder_point', 75)
        
        if not all([product_name, category, unit_price, supplier_id]):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            unit_price = float(unit_price)
            initial_quantity = int(initial_quantity)
            min_stock_level = int(min_stock_level)
            reorder_point = int(reorder_point)
        except ValueError:
            conn.close()
            return jsonify({'success': False, 'error': 'Invalid data types'}), 400
        
        cursor.execute(
            'INSERT INTO Products (ProductName, Category, UnitPrice, SupplierID) VALUES (?, ?, ?, ?)',
            (product_name, category, unit_price, supplier_id)
        )
        
        product_id = cursor.lastrowid
        
        cursor.execute(
            'INSERT INTO Inventory (ProductID, QuantityAvailable, MinimumStockLevel, ReorderPoint, LastUpdated) VALUES (?, ?, ?, ?, ?)',
            (product_id, initial_quantity, min_stock_level, reorder_point, datetime.now())
        )
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Product added successfully',
            'product_id': product_id,
            'product_name': product_name
        })
    
    except Exception as e:
        conn.rollback()
        conn.close()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/dashboard-stats', methods=['GET'])
def get_dashboard_stats():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) as count FROM Products')
        total_products = cursor.fetchone()['count']
        
        cursor.execute('''
            SELECT COUNT(*) as count 
            FROM Inventory 
            WHERE QuantityAvailable <= ReorderPoint
        ''')
        low_stock_count = cursor.fetchone()['count']
        
        week_ago = (datetime.now() - timedelta(days=7)).date()
        cursor.execute('''
            SELECT COALESCE(SUM(TotalAmount), 0) as total
            FROM Sales
            WHERE SaleDate >= ?
        ''', (week_ago,))
        weekly_sales = cursor.fetchone()['total']
        
        cursor.execute('''
            SELECT COALESCE(SUM(QuantitySold), 0) as total
            FROM Sales
            WHERE SaleDate >= ?
        ''', (week_ago,))
        weekly_units = cursor.fetchone()['total']
        
        cursor.execute('''
            SELECT 
                p.ProductName,
                SUM(s.QuantitySold) as TotalSold
            FROM Sales s
            JOIN Products p ON s.ProductID = p.ProductID
            WHERE s.SaleDate >= ?
            GROUP BY p.ProductID, p.ProductName
            ORDER BY TotalSold DESC
            LIMIT 5
        ''', (week_ago,))
        
        top_products = []
        for row in cursor.fetchall():
            top_products.append({
                'product_name': row['ProductName'],
                'total_sold': row['TotalSold']
            })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'total_products': total_products,
            'low_stock_alerts': low_stock_count,
            'weekly_sales': round(weekly_sales, 2),
            'weekly_units': weekly_units,
            'top_products': top_products
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
