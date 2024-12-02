from lib.db_connection import connect_to_db
from lib.analytics import perform_descriptive_analysis

if __name__ == "__main__":
    conn = connect_to_db()
    if conn:
        perform_descriptive_analysis(conn)
        conn.close()
