from lib.topic_modeling import perform_topic_modeling

if __name__ == "__main__":
    comments = ["Great service", "Late delivery", "Excellent quality", "Poor packaging"]  # Example data
    perform_topic_modeling(comments)
