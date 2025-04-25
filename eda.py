import matplotlib.pyplot as plt
import seaborn as sns

def plot_genre_distribution(df):
    genre_counts = df['Genre'].value_counts().head(10)
    sns.barplot(x=genre_counts.values, y=genre_counts.index)
    plt.title("Top 10 Genres")
    plt.xlabel("Count")
    return plt
