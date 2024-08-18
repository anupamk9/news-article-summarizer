# ⭐ Advanced Data Analytics Project ⭐

## Overview

Welcome to the Advanced Data Analytics Project  🚀 This project focuses on evaluating machine learning models for news article classification. The goal is to preprocess data, implement various algorithms, analyze results using key metrics, and document the findings.

**Evaluating Machine Learning for News Article Classification: Metrics, Insights, and Best Practices**

## Project Components

1. **Domain & Problem Statement** 🗂️
   - **Domain:** News Article Classification
   - **Problem Statement:** Classify news articles into predefined categories (technology, politics, sports, economy) using various machine learning and deep learning models.

2. **Dataset Collection** 📊
   - **Source:** NewsAPI
   - **Dataset:** News articles collected from various categories.
   - **File:** `news_articles.csv`

3. **Data Preprocessing** 🧹
   - **Steps:** 
     - Combine text features (title, description, content)
     - Handle missing values and outliers
     - Convert text data into TF-IDF features
     - Split data into training and testing sets

4. **Model Training & Evaluation** 🧠
   - **Algorithms Implemented:**
     - Naive Bayes
     - Random Forest
     - SVM
     - BERT
   - **Metrics Used:**
     - Accuracy
     - Precision
     - Recall
     - F1 Score
     - ROC-AUC

5. **Results & Insights** 🔍
   - Performance comparison of the models
   - Insights into strengths and weaknesses of each model

## Tools & Technology Stack 🛠️

- **Programming Language:** Python 🐍

- **Libraries:** 
  - `sklearn` ⭐: For traditional ML models (Naive Bayes, Random Forest, SVM)
  - `transformers` 🌟: For BERT model
  - `newsapi-python` 📰: For data scraping
  - `pandas` 🧑‍💻, `numpy` 🔢: For data manipulation
  - `matplotlib` 📈, `seaborn` 🌈: For visualization

- **Environment:** Jupyter Notebook 💻

![Company Logo](link-to-your-company-logo.png)

## Code Examples

### Data Collection

```python
from newsapi import NewsApiClient
import pandas as pd

newsapi = NewsApiClient(api_key='YOUR_API_KEY')

categories = ['technology', 'politics', 'sports', 'economy']
articles_list = []

for category in categories:
    for page in range(1, 2):
        articles = newsapi.get_everything(q=category, language='en', page=page)
        for article in articles['articles']:
            articles_list.append({
                'title': article['title'],
                'description': article['description'],
                'content': article['content'],
                'publishedAt': article['publishedAt'],
                'source': article['source']['name'],
                'url': article['url'],
                'category': category
            })

df = pd.DataFrame(articles_list)
df.to_csv('news_articles.csv', index=False)
```

### Data Preprocessing

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df['text'] = df['title'] + " " + df['description'] + " " + df['content']
df = df.dropna().reset_index(drop=True)
X = df['text']
y = df['category']

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
```

## Model Performance 📈

- **Naive Bayes:**
  - Accuracy: 21%
  - Precision: 20%
  - Recall: 20%
  - F1-Score: 20%

- **Random Forest:**
  - Accuracy: 28%
  - Precision: 25%
  - Recall: 26%
  - F1-Score: 23%

- **SVM:**
  - Accuracy: 24%
  - Precision: 21%
  - Recall: 22%
  - F1-Score: 21%

- **BERT:**
  - Accuracy: 23%
  - Precision: 17%
  - Recall: 22%
  - F1-Score: 19%

## Insights & Conclusion 🔍

- **Naive Bayes:** Consistent but limited in handling complex tasks.
- **Random Forest:** Better accuracy, struggles with data imbalance.
- **SVM:** Slightly better in precision and recall but similar to Naive Bayes.
- **BERT:** Requires fine-tuning and more data for optimal performance.

### Best Practices

- Ensure high-quality, well-labeled data.
- Choose models based on task complexity.
- Use multiple metrics for a comprehensive view.
- Continuously fine-tune models and preprocess data.

## Documentation 📚

For a detailed discussion on the implementation and evaluation of these models, please refer to the [Medium blog post](#).

## Contribution & Contact 📧

For any questions or contributions, please contact [Your Email/Contact Info].

