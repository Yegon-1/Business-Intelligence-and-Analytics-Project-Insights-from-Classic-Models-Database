import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re




nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_comments(comments):
    """
    Preprocesses customer comments by cleaning, lemmatizing, and removing stopwords.
    Args:
        comments (pd.Series): Series of text comments.
    Returns:
        List of processed comments.
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    processed_comments = []
    for comment in comments:
        # Remove special characters and numbers
        comment = re.sub(r'[^a-zA-Z\s]', '', str(comment))
        # Convert to lowercase
        comment = comment.lower()
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in comment.split() if word not in stop_words]
        processed_comments.append(' '.join(tokens))
    
    return processed_comments

def perform_topic_modeling(conn, n_topics=3):
    """
    Performs topic modeling on customer comments from the orders table.
    Args:
        conn: MySQL database connection object.
        n_topics (int): Number of topics for LDA.
    """
    try:
        # Query customer comments from the database
        query = "SELECT comments FROM orders WHERE comments IS NOT NULL;"
        comments = pd.read_sql(query, conn)
        
        if comments.empty:
            print("No comments found in the database.")
            return
        
        # Preprocess comments
        print("Preprocessing comments...")
        comments['processed'] = preprocess_comments(comments['comments'])
        
        # Vectorize text
        print("Vectorizing text data...")
        vectorizer = CountVectorizer(stop_words='english', max_features=1000)  # Limit features for efficiency
        data_vectorized = vectorizer.fit_transform(comments['processed'])
        
        # Apply LDA
        print("Applying Latent Dirichlet Allocation (LDA)...")
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(data_vectorized)
        
        # Display topics
        print("\nIdentified Topics:")
        topic_labels = []
        for idx, topic in enumerate(lda.components_):
            top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
            topic_labels.append(f"Topic {idx + 1}: {' '.join(top_words[:3])}")
            print(f"{topic_labels[-1]}:")
            print(top_words)
        
        # Create word clouds for each topic
        print("\nGenerating word clouds for each topic...")
        for idx, topic in enumerate(lda.components_):
            word_freq = {vectorizer.get_feature_names_out()[i]: topic[i] for i in topic.argsort()}
            wordcloud = WordCloud(background_color='white').generate_from_frequencies(word_freq)
            plt.figure(figsize=(8, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(topic_labels[idx])
            plt.axis("off")
            plt.show()

        # Save topic results for further analysis
        print("\nSaving topics to database...")
        topic_df = pd.DataFrame({
            "Topic": [f"Topic {i+1}" for i in range(n_topics)],
            "Top Words": [' '.join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]) for topic in lda.components_]
        })
        topic_df.to_sql("topic_modeling_results", conn, if_exists="replace", index=False)
        print("Topics saved successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")
