import sqlite3
import random
from datetime import datetime, timedelta
import numpy as np

def create_database():
    conn = sqlite3.connect('inventory.db')
    cursor = conn.cursor()
    
    cursor.execute('DROP TABLE IF EXISTS Sales')
    cursor.execute('DROP TABLE IF EXISTS Inventory')
    cursor.execute('DROP TABLE IF EXISTS Products')
    cursor.execute('DROP TABLE IF EXISTS Suppliers')
    
    cursor.execute('''
        CREATE TABLE Suppliers (
            SupplierID INTEGER PRIMARY KEY AUTOINCREMENT,
            SupplierName TEXT NOT NULL,
            ContactInfo TEXT,
            Email TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE Products (
            ProductID INTEGER PRIMARY KEY AUTOINCREMENT,
            ProductName TEXT NOT NULL,
            Category TEXT,
            UnitPrice REAL,
            SupplierID INTEGER,
            FOREIGN KEY (SupplierID) REFERENCES Suppliers(SupplierID)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE Inventory (
            InventoryID INTEGER PRIMARY KEY AUTOINCREMENT,
            ProductID INTEGER,
            QuantityAvailable INTEGER,
            MinimumStockLevel INTEGER,
            ReorderPoint INTEGER,
            LastUpdated DATETIME,
            FOREIGN KEY (ProductID) REFERENCES Products(ProductID)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE Sales (
            SaleID INTEGER PRIMARY KEY AUTOINCREMENT,
            ProductID INTEGER,
            SaleDate DATE,
            QuantitySold INTEGER,
            TotalAmount REAL,
            FOREIGN KEY (ProductID) REFERENCES Products(ProductID)
        )
    ''')
    
    conn.commit()
    return conn, cursor

def insert_sample_data(conn, cursor):
    suppliers = [
        ('TechSupply Inc', '555-0101', 'contact@techsupply.com'),
        ('Global Electronics', '555-0102', 'sales@globalelec.com'),
        ('Office Mart', '555-0103', 'info@officemart.com'),
        ('HomeGoods Ltd', '555-0104', 'support@homegoods.com'),
        ('Digital Wholesale', '555-0105', 'orders@digitalwholesale.com')
    ]
    
    cursor.executemany(
        'INSERT INTO Suppliers (SupplierName, ContactInfo, Email) VALUES (?, ?, ?)',
        suppliers
    )
    
    products = [
        ('Wireless Mouse', 'Electronics', 599, 1),
        ('USB Keyboard', 'Electronics', 1299, 1),
        ('Monitor Stand', 'Office Supplies', 899, 3),
        ('Desk Lamp', 'Office Supplies', 599, 4),
        ('External Hard Drive', 'Electronics', 4999, 2),
        ('Webcam HD', 'Electronics', 2499, 1),
        ('Office Chair Mat', 'Office Supplies', 1299, 3),
        ('Cable Organizer', 'Office Supplies', 299, 3),
        ('Bluetooth Speaker', 'Electronics', 1999, 5),
        ('Smart Watch', 'Electronics', 9999, 5)
    ]
    
    cursor.executemany(
        'INSERT INTO Products (ProductName, Category, UnitPrice, SupplierID) VALUES (?, ?, ?, ?)',
        products
    )
    
    inventory_data = [
        (1, 150, 50, 75),
        (2, 120, 40, 60),
        (3, 80, 30, 45),
        (4, 100, 35, 50),
        (5, 60, 20, 30),
        (6, 90, 30, 45),
        (7, 70, 25, 40),
        (8, 200, 60, 80),
        (9, 110, 35, 50),
        (10, 50, 15, 25)
    ]
    
    now = datetime.now()
    for product_id, qty, min_stock, reorder_point in inventory_data:
        cursor.execute(
            'INSERT INTO Inventory (ProductID, QuantityAvailable, MinimumStockLevel, ReorderPoint, LastUpdated) VALUES (?, ?, ?, ?, ?)',
            (product_id, qty, min_stock, reorder_point, now)
        )
    
    conn.commit()

def generate_sales_data(conn, cursor, days=180):
    start_date = datetime.now() - timedelta(days=days)
    
    product_patterns = {
        1: {'base': 25, 'trend': 0.05, 'seasonality': 5, 'noise': 3},
        2: {'base': 20, 'trend': 0.03, 'seasonality': 4, 'noise': 3},
        3: {'base': 15, 'trend': 0.02, 'seasonality': 3, 'noise': 2},
        4: {'base': 18, 'trend': 0.04, 'seasonality': 4, 'noise': 2},
        5: {'base': 12, 'trend': 0.01, 'seasonality': 2, 'noise': 2},
        6: {'base': 16, 'trend': 0.06, 'seasonality': 3, 'noise': 2},
        7: {'base': 14, 'trend': 0.02, 'seasonality': 2, 'noise': 2},
        8: {'base': 30, 'trend': 0.03, 'seasonality': 6, 'noise': 4},
        9: {'base': 19, 'trend': 0.05, 'seasonality': 4, 'noise': 3},
        10: {'base': 10, 'trend': 0.07, 'seasonality': 2, 'noise': 2}
    }
    
    cursor.execute('SELECT ProductID, UnitPrice FROM Products')
    products = cursor.fetchall()
    
    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        day_of_week = current_date.weekday()
        week_of_year = current_date.isocalendar()[1]
        
        for product_id, unit_price in products:
            pattern = product_patterns[product_id]
            
            trend_component = pattern['base'] + (day_offset * pattern['trend'] / 30)
            
            seasonality_component = pattern['seasonality'] * np.sin(2 * np.pi * week_of_year / 52)
            
            weekend_factor = 0.7 if day_of_week >= 5 else 1.0
            
            noise = random.gauss(0, pattern['noise'])
            
            quantity = max(0, int(trend_component + seasonality_component + noise))
            quantity = int(quantity * weekend_factor)
            
            if quantity > 0:
                total_amount = quantity * unit_price
                cursor.execute(
                    'INSERT INTO Sales (ProductID, SaleDate, QuantitySold, TotalAmount) VALUES (?, ?, ?, ?)',
                    (product_id, current_date.date(), quantity, total_amount)
                )
    
    conn.commit()
    print(f"Generated {days} days of sales data for {len(products)} products")

if __name__ == '__main__':
    print("Creating database and tables...")
    conn, cursor = create_database()
    
    print("Inserting sample data (suppliers, products, inventory)...")
    insert_sample_data(conn, cursor)
    
    print("Generating 180 days of sales history with realistic patterns...")
    generate_sales_data(conn, cursor, days=180)
    
    cursor.execute('SELECT COUNT(*) FROM Sales')
    sales_count = cursor.fetchone()[0]
    print(f"\nDatabase setup complete!")
    print(f"Total sales records created: {sales_count}")
    
    conn.close()
