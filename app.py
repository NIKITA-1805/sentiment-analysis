import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Config ---
# For big datasets, you may set nrows to a smaller value for development/testing
USE_SAMPLE = st.sidebar.checkbox("Use a data sample (for fast loading)?", value=True)
NROWS_SAMPLE = st.sidebar.number_input("Number of rows to load:", min_value=1000, max_value=1000000, value=50000, step=1000)

# --- Helper: preprocess review text ---
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        return text.translate(str.maketrans('', '', string.punctuation))
    return ""

# --- Data loading ---
st.title("📊 Sentiment Analysis on Amazon Reviews (Logistic Regression)")

@st.cache_data
def load_data(sample=USE_SAMPLE, nrows=NROWS_SAMPLE):
    # Adjust the number of rows loaded based on sidebar widget
    if sample:
        df = pd.read_csv('reviews.csv', nrows=nrows)
    else:
        df = pd.read_csv('reviews.csv')
    # Clean, filter, and standardize
    df.dropna(subset=['review', 'sentiment'], inplace=True)
    df['clean_review'] = df['review'].apply(preprocess_text)
    df['sentiment'] = df['sentiment'].str.lower().str.strip()
    df = df[df['sentiment'].isin(['positive', 'negative'])]
    return df

df = load_data()

if len(df) == 0:
    st.error("📛 Your dataset is empty or incorrectly formatted! Please check reviews.csv.")
    st.stop()

# --- Display Sample Data ---
st.subheader("Sample Data")
num_samples = min(10, len(df))
if num_samples < 10:
    st.info(f"Only {num_samples} rows available in dataset.")
st.dataframe(df[['review', 'sentiment']].sample(num_samples, random_state=42))

# --- Sentiment Distribution ---
st.subheader("Sentiment Distribution")
sentiment_counts = df['sentiment'].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette="viridis")
ax.set_ylabel('Count')
ax.set_xlabel('Sentiment')
ax.set_title('Number of Reviews per Sentiment')
st.pyplot(fig)

# --- Prepare Features for Model ---
X = df['clean_review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})

if len(df['sentiment'].unique()) < 2 or len(df) < 4:
    st.warning("Not enough data for modeling. Please use more data!")
    st.stop()

# --- Train/test split (stratified) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- TF-IDF Vectorizer (memory efficient) ---
vectorizer = TfidfVectorizer(
    max_features=5000,            # Change this number as needed for RAM/speed
    ngram_range=(1,2),
    stop_words='english'
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# --- Train Logistic Regression Model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# --- Evaluate Model ---
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
st.subheader("✅ Model Performance")
st.write(f"Accuracy: **{accuracy * 100:.2f}%**")
st.text(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative','Positive'],
            yticklabels=['Negative','Positive'],
            ax=ax_cm)
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('Actual')
ax_cm.set_title('Confusion Matrix')
st.pyplot(fig_cm)

# --- Predict Sentiment for User Input ---
st.subheader("💬 Predict Sentiment of Your Review")
user_input = st.text_area("Enter a product review to classify its sentiment:")

if user_input:
    cleaned_input = preprocess_text(user_input)
    input_vec = vectorizer.transform([cleaned_input])
    pred = model.predict(input_vec)[0]
    pred_proba = model.predict_proba(input_vec)[0]
    sentiment_label = 'Positive' if pred == 1 else 'Negative'
    st.markdown(f"### Predicted Sentiment: **{sentiment_label}**")
    st.write(f"Confidence Scores - Positive: {pred_proba[1]:.2f}, Negative: {pred_proba[0]:.2f}")
