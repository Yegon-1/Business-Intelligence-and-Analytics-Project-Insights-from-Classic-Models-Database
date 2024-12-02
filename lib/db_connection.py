import mysql.connector
from sqlalchemy import create_engine

def connect_to_db():
    """
    Establishes a connection to the MySQL database.
    Returns a SQLAlchemy engine for pandas compatibility.
    """
    try:
        # MySQL connection parameters
        host = "localhost"
        user = "root"  # Replace with your MySQL username
        password = "hiSc70R78Vgjnj0VT"  # Replace with your MySQL password
        database = "classicmodels"  # Replace with your database name

        # Connect using mysql.connector
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        print("Connected to the database successfully!")

        # Create SQLAlchemy engine from the mysql.connector connection
        # The 'mysql+mysqlconnector' part tells SQLAlchemy to use mysql.connector as the DBAPI
        engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{database}')
        
        return engine  # Returning engine for pandas compatibility
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None
