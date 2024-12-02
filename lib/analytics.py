import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Descriptive Analytics
def perform_descriptive_analysis(conn):
    """
    Performs descriptive analytics on the data.
    Args:
        conn: MySQL connection object.
    """
    # Fetch top customers by credit limit
    query_customers = "SELECT * FROM customers;"
    customers = pd.read_sql(query_customers, conn)
    print("Top 5 Customers by Credit Limit:")
    print(customers[['customerNumber', 'customerName', 'creditLimit']].head())

    # Fetch monthly sales trends
    query_sales_trends = """
    SELECT 
        DATE_FORMAT(o.orderDate, '%Y-%m') AS month,
        SUM(od.quantityOrdered * od.priceEach) AS revenue
    FROM orders o
    JOIN orderdetails od ON o.orderNumber = od.orderNumber
    GROUP BY month
    ORDER BY month;
    """
    sales_data = pd.read_sql(query_sales_trends, conn)
    print("\nMonthly Sales Trends:")
    print(sales_data.head())

    # Visualize sales trends
    plt.figure(figsize=(8, 6))
    plt.plot(sales_data['month'], sales_data['revenue'], marker='o')
    plt.title('Monthly Sales Trends')
    plt.xlabel('Month')
    plt.ylabel('Revenue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Customer Segmentation
def customer_segmentation(conn):
    """
    Performs RFM analysis and clusters customers using K-Means.
    Args:
        conn: MySQL database connection object.
    """
    query = """
    SELECT 
        c.customerNumber,
        MAX(o.orderDate) AS last_order_date,
        COUNT(o.orderNumber) AS frequency,
        SUM(od.quantityOrdered * od.priceEach) AS monetary
    FROM customers c
    JOIN orders o ON c.customerNumber = o.customerNumber
    JOIN orderdetails od ON o.orderNumber = od.orderNumber
    GROUP BY c.customerNumber;
    """
    rfm_data = pd.read_sql(query, conn)

    # Calculate Recency
    rfm_data['recency'] = (pd.Timestamp.now() - pd.to_datetime(rfm_data['last_order_date'])).dt.days
    rfm_data = rfm_data[['customerNumber', 'recency', 'frequency', 'monetary']]

    # Scale RFM data
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data[['recency', 'frequency', 'monetary']])

    # Apply K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm_data['cluster'] = kmeans.fit_predict(rfm_scaled)

    print("\nCustomer Segmentation:")
    print(rfm_data)

    # Visualize Clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(rfm_data['recency'], rfm_data['monetary'], c=rfm_data['cluster'], cmap='viridis', alpha=0.6)
    plt.title('Customer Segmentation (RFM Clusters)')
    plt.xlabel('Recency')
    plt.ylabel('Monetary')
    plt.colorbar(label='Cluster')
    plt.tight_layout()
    plt.show()

    # Save RFM clusters to database
    save_clusters_to_db(conn, rfm_data)


def save_clusters_to_db(conn, rfm_data):
    """
    Saves RFM clusters to the database.
    Args:
        conn: MySQL database connection object.
        rfm_data: DataFrame with RFM and cluster data.
    """
    table_name = "rfm_clusters"
    print(f"Saving RFM clusters to table: {table_name}")
    rfm_data.to_sql(table_name, conn, if_exists='replace', index=False)
    print("RFM clusters saved successfully!")

def top_selling_products(conn):
    query = """
    SELECT 
        p.productName,
        SUM(od.quantityOrdered) AS revenue
    FROM orderdetails od
    JOIN products p ON od.productCode = p.productCode
    GROUP BY p.productName
    ORDER BY revenue DESC
    LIMIT 5;
    """
    product_data = pd.read_sql(query, conn)

    print("\nTop 5 Selling Products:")
    print(product_data)

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.bar(product_data['productName'], product_data['revenue'], color='green')
    plt.title("Top 5 Selling Products")
    plt.xlabel("Product Name")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Product Demand Prediction
def demand_prediction(conn):
    """
    Predicts product demand using Linear Regression.
    Args:
        conn: MySQL database connection object.
    """
    query = """
    SELECT 
        p.productLine,
        SUM(od.quantityOrdered) AS total_quantity,
        AVG(od.priceEach) AS avg_price
    FROM products p
    JOIN orderdetails od ON p.productCode = od.productCode
    GROUP BY p.productLine;
    """
    demand_data = pd.read_sql(query, conn)

    # Prepare data
    X = demand_data[['avg_price']]
    y = demand_data['total_quantity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    print("\nProduct Demand Prediction:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R^2 Score: {r2_score(y_test, y_pred):.2f}")

    # Visualize actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.title('Actual vs Predicted Product Demand')
    plt.xlabel('Actual Demand')
    plt.ylabel('Predicted Demand')
    plt.tight_layout()
    plt.show()
