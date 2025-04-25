from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_models(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Description'])

    kmeans = KMeans(n_clusters=10, random_state=42)
    df['Cluster'] = kmeans.fit_predict(tfidf_matrix)

    sim_matrix = cosine_similarity(tfidf_matrix)
    return df, sim_matrix

def recommend(df, sim_matrix, book_title, top_n=5):
    if book_title not in df['Book Name'].values:
        return pd.DataFrame()

    idx = df[df['Book Name'] == book_title].index[0]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    indices = [i[0] for i in sim_scores]

    # Dynamically select only available columns
    available_cols = [col for col in ['Book Name', 'Author', 'Rating', 'Genre'] if col in df.columns]
    return df.iloc[indices][available_cols]
