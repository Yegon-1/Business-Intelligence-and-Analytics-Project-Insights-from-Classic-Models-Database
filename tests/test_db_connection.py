from lib.db_connection import connect_to_db

def test_connect_to_db():
    conn = connect_to_db()
    assert conn is not None, "Database connection failed!"
