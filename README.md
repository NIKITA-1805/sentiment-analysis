# Sentiment Analysis

This project is a sentiment analysis web application built with Streamlit that classifies Amazon product reviews as positive or negative using logistic regression and TF-IDF vectorization. It includes tools to preprocess raw text data, visualize sentiment distribution, and make real-time predictions from user input. The app supports large datasets with sampling options for efficient development and testing.

The raw dataset is sourced from Kaggle and converted using a custom script for streamlined model training and deployment.

---

## 📦 Dataset

The original dataset is **not included** in this repository due to its size.

You can download it from Kaggle:  
🔗 [Amazon Reviews for Sentiment Analysis](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)

After downloading, place the following files in your project directory:

- `train.txt`
- `test.txt`

Then run:

```bash
python dataTransform.py
```

This script will merge and convert the data into a CSV file named `reviews.csv`.

---

## 🚀 How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/NIKITA-1805/sentiment-analysis.git
cd sentiment-analysis
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn
```

3. **Start the app:**

```bash
streamlit run app.py
```

---

## 🧠 Features

- Clean and preprocess Amazon reviews
- TF-IDF vectorization
- Logistic Regression for binary sentiment classification
- Evaluation metrics: Accuracy, Classification Report, Confusion Matrix
- Live sentiment prediction for user-provided text
- Sidebar options for sampling large datasets

---

## 🗂️ Project Structure

| File              | Description                                              |
|------------------|----------------------------------------------------------|
| `app.py`          | Main Streamlit app for sentiment analysis               |
| `dataTransform.py`| Script to generate `reviews.csv` from raw `.txt` files  |
| `reviews.csv`     | Cleaned dataset used for training and testing (auto-generated) |

---

## ⚠️ Notes

- Large data files like `train.txt`, `test.txt`, and `reviews.csv` have been deleted from the repo.
- To recreate them, use the Kaggle link and `dataTransform.py`.
- Temporary files (e.g., `.pkl`) are not included in version control.

---

## 📚 Credits

- Dataset from [Amazon Reviews for Sentiment Analysis](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
- Built using Python, Streamlit, Scikit-learn, Pandas, Matplotlib, and Seaborn
