"""Quick debug of fulltext sentiment data."""
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/news.db')

# Check fulltext sentiment scores
df = pd.read_sql_query('''
    SELECT id, ticker, published_date, sentiment_score, fulltext_sentiment_score
    FROM articles 
    WHERE fulltext_sentiment_score IS NOT NULL
    LIMIT 10
''', conn)
print("Sample articles with fulltext sentiment:")
print(df)

# Count
count = pd.read_sql_query("SELECT COUNT(*) as cnt FROM articles WHERE fulltext_sentiment_score IS NOT NULL", conn)
print(f"\nTotal with fulltext sentiment: {count.iloc[0,0]:,}")

# Check date range  
dates = pd.read_sql_query('''
    SELECT MIN(published_date) as min_date, MAX(published_date) as max_date
    FROM articles 
    WHERE fulltext_sentiment_score IS NOT NULL
''', conn)
print(f"Date range: {dates.iloc[0,0]} to {dates.iloc[0,1]}")

conn.close()
