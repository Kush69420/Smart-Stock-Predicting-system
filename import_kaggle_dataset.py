import sqlite3
import pandas as pd
from datetime import datetime
import os

def import_kaggle_sales_inventory_dataset(csv_path):
    """
    Import Kaggle Sales and Inventory Dataset into our inventory system.
    Download from: https://www.kaggle.com/datasets/yukisim/sales-and-inventory-dataset
    """
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return False
    
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Connect to database
    conn = sqlite3.connect('inventory.db')
    cursor = conn.cursor()
    
    # Clear existing data
    print("\nClearing existing data...")
    cursor.execute('DELETE FROM Sales')
    cursor.execute('DELETE FROM Inventory')
    cursor.execute('DELETE FROM Products')
    cursor.execute('DELETE FROM Suppliers')
    
    # Map CSV columns to our schema (adjust based on your CSV structure)
    try:
        # Extract unique products
        products_data = df[['Product_name', 'Product_category', 'Price']].drop_duplicates()
        
        # Create a default supplier
        suppliers = [('Kaggle Dataset', '000-0000', 'dataset@kaggle.com')]
        cursor.executemany(
            'INSERT INTO Suppliers (SupplierName, ContactInfo, Email) VALUES (?, ?, ?)',
            suppliers
        )
        supplier_id = 1
        
        product_mapping = {}
        for idx, row in products_data.iterrows():
            product_name = str(row['Product_name']).strip()
            category = str(row['Product_category']).strip()
            price = float(row['Price'])
            
            cursor.execute(
                'INSERT INTO Products (ProductName, Category, UnitPrice, SupplierID) VALUES (?, ?, ?, ?)',
                (product_name, category, price, supplier_id)
            )
            product_id = cursor.lastrowid
            product_mapping[product_name] = (product_id, price)
            
            # Add inventory for each product
            cursor.execute(
                'INSERT INTO Inventory (ProductID, QuantityAvailable, MinimumStockLevel, ReorderPoint, LastUpdated) VALUES (?, ?, ?, ?, ?)',
                (product_id, 100, 30, 50, datetime.now())
            )
        
        conn.commit()
        print(f"Imported {len(product_mapping)} unique products")
        
        # Import sales data
        sales_count = 0
        for idx, row in df.iterrows():
            product_name = str(row['Product_name']).strip()
            
            if product_name not in product_mapping:
                continue
            
            product_id, price_inr = product_mapping[product_name]
            
            # Map columns - adjust these based on your CSV
            quantity = int(row.get('Quantity', row.get('quantity_sold', 1)))
            
            # Try to find date column
            date_col = None
            for col in ['Order_date', 'Date', 'SaleDate', 'date']:
                if col in row and pd.notna(row[col]):
                    date_col = col
                    break
            
            if date_col is None:
                continue
            
            try:
                sale_date = pd.to_datetime(row[date_col]).date()
            except:
                continue
            
            total_amount = quantity * price_inr
            
            cursor.execute(
                'INSERT INTO Sales (ProductID, SaleDate, QuantitySold, TotalAmount) VALUES (?, ?, ?, ?)',
                (product_id, sale_date, quantity, total_amount)
            )
            sales_count += 1
        
        conn.commit()
        print(f"Imported {sales_count} sales transactions")
        
        cursor.execute('SELECT COUNT(*) FROM Products')
        product_count = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM Sales')
        total_sales = cursor.fetchone()[0]
        
        print(f"\n✓ Import successful!")
        print(f"  Total products: {product_count}")
        print(f"  Total sales: {total_sales}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error during import: {e}")
        conn.close()
        return False

if __name__ == '__main__':
    print("="*60)
    print("KAGGLE DATASET IMPORT TOOL")
    print("="*60)
    print("\nInstructions:")
    print("1. Download dataset from Kaggle:")
    print("   https://www.kaggle.com/datasets/yukisim/sales-and-inventory-dataset")
    print("2. Extract the CSV file")
    print("3. Run: python import_kaggle_dataset.py <path_to_csv>")
    print("\nExample:")
    print("   python import_kaggle_dataset.py sales_inventory.csv")
    print("="*60)
    
    import sys
    if len(sys.argv) < 2:
        csv_path = 'sales_inventory.csv'
        print(f"\nNo CSV path provided. Looking for: {csv_path}")
    else:
        csv_path = sys.argv[1]
    
    success = import_kaggle_sales_inventory_dataset(csv_path)
    
    if success:
        print("\nNext steps:")
        print("1. Run: python data_prep.py")
        print("2. Run: python model.py")
        print("3. Run: python app.py")
        print("4. Open dashboard at http://localhost:5000")
    else:
        print("\nImport failed. Please check the file path and CSV structure.")
