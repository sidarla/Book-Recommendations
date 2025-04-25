import streamlit as st
from data_preparation import load_and_clean_data
from recommendation_engine import build_models, recommend

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Audible Insights | Personalized Book Recommendations",
    page_icon="📚",
    layout="wide",
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        body {
            background-color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-title {
            font-size: 3.5rem;
            font-weight: 700;
            color: #0A66C2;
            text-align: center;
            margin-top: 10px;
        }
        .sub-title {
            font-size: 1.4rem;
            font-weight: 400;
            color: #495057;
            text-align: center;
            margin-bottom: 20px;
        }
        .info-box {
            background-color: #f1f3f5;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            font-size: 1rem;
            color: #212529;
        }
        .footer {
            text-align: center;
            font-size: 0.9rem;
            color: #adb5bd;
            margin-top: 40px;
            padding-bottom: 10px;
        }
        .stButton>button {
            background-color: #0A66C2;
            color: white;
            border: None;
            border-radius: 8px;
            padding: 0.5rem 1.2rem;
            font-size: 1rem;
            font-weight: 600;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #004080;
        }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135693.png", width=100)  # optional nice user icon
    st.title("🔖 Welcome to Book Recommender System")
    
    st.markdown("---")
    st.header("📚 Problem Statement")
    st.markdown("""
    Design a book recommendation system that retrieves book details from given datasets, processes and cleans the data before applying NLP techniques and clustering methods and builds multiple recommendation models. The final application will allow users to search for book recommendations using a user-friendly interface deployed with Streamlit and hosted on AWS. 
    """)
    
    st.markdown("---")
    st.subheader("📢 Action")
    st.markdown("Select a book you like from the main panel and press **Get Recommendations** to discover new favorites!")

# --- HEADER ---
st.markdown('<div class="main-title">Intelligent Book Recommendations</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Smarter Book Recommendations, Tailored for You</div>', unsafe_allow_html=True)

# --- INFORMATION SECTION ---
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown("""
**🔎 About Audible Insights:**  
Your personal AI librarian recommending books based on your reading taste using Natural Language Processing (NLP) and Machine Learning (ML).

**🚀 How It Works:**  
- **TF-IDF** transforms book descriptions into numbers.
- **KMeans** groups similar books together.
- **Cosine Similarity** ranks recommendations based on deep meaning, not just genre.

**💡 Why Audible Insights?**  
- Deep content understanding  
- Personalized for every user  
- Fresh discovery beyond bestsellers
""")
st.markdown('</div>', unsafe_allow_html=True)

# --- LOAD DATA ---
with st.spinner("Loading book database and initializing recommendation engine..."):
    df = load_and_clean_data()
    df, sim_matrix = build_models(df)

st.success("✅ System Ready! Find your next great read.")

# --- MAIN FUNCTIONALITY ---
st.subheader("🎯 Get Personalized Recommendations")
book_list = df['Book Name'].dropna().sort_values().unique()
selected_book = st.selectbox("Pick a book you enjoyed recently:", book_list)

if st.button("🚀 Get Recommendations"):
    recs = recommend(df, sim_matrix, selected_book)
    if recs.empty:
        st.error("❌ No recommendations found. Try another book.")
    else:
        st.success("📚 Here’s what you might love next:")
        st.table(recs.reset_index(drop=True))

# --- FOOTER ---
st.markdown('<div class="footer">© 2025 Audible Insights | Crafted with 💙 for curious minds worldwide.</div>', unsafe_allow_html=True)
