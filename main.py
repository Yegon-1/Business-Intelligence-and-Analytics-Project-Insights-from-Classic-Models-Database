from lib.db_connection import connect_to_db
from lib.analytics import perform_descriptive_analysis, customer_segmentation, demand_prediction, top_selling_products
from lib.topic_modelling import perform_topic_modeling
import pandas as pd

def main():
    """
    Main entry point for the script.
    Connects to the database, performs descriptive analysis,
    customer segmentation, demand prediction, and topic modeling.
    """
    try:
        # Establish a connection to the database
        print("Connecting to the database...")
        db_connection = connect_to_db()
        
        if db_connection is None:
            print("Failed to connect to the database. Exiting...")
            return
        
        # Perform descriptive analysis
        print("\nPerforming descriptive analysis...")
        perform_descriptive_analysis(db_connection)
        
        # Perform customer segmentation
        print("\nPerforming customer segmentation...")
        customer_segmentation(db_connection)
        
        # Predict product demand
        print("\nPredicting product demand...")
        demand_prediction(db_connection)

        # Analyze top-selling products
        print("\nPerforming top-selling products analysis...")
        top_selling_products(db_connection)

        # Perform topic modeling
        print("\nPerforming topic modeling...")
        perform_topic_modeling(db_connection, n_topics=4)

    except Exception as e:
        # Handle any unexpected errors
        print(f"An error occurred: {e}")
    
    finally:
        # No need to explicitly close the connection for SQLAlchemy engine
        print("Execution complete.")

if __name__ == "__main__":
    main()
