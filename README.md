🎬 Movie Review Sentiment Analysis (IMDB Dataset)
This Python project performs sentiment analysis on movie reviews using Natural Language Processing (NLP) and the VADER sentiment analyzer. It allows users to input custom reviews or analyze a subset of the IMDB dataset, classifying each review as positive, negative, or neutral.

🔧 Features
📝 Custom review input: Type your own reviews and get instant sentiment prediction.

📊 Bulk review analysis: Analyze n reviews from the IMDB dataset.

🧹 Text preprocessing: Lowercasing, stopword removal, punctuation cleaning, stemming.

💬 Sentiment scoring: Using VADER's compound polarity.

✅ Performance tracking: Compares predicted vs actual sentiment and reports accuracy.

📄 Output formatting: Full reviews, predictions, and correctness are displayed cleanly.

📁 Dataset
Uses the IMDB Movie Reviews Dataset (50,000 reviews).

You can download it from official sources and place it in your desired directory (e.g., Downloads/project internship/IMDB Dataset.csv).

🚀 How It Works
User choice: Enter your own reviews or load from dataset.

Preprocessing: Cleans each review using NLTK.

Sentiment scoring: Uses SentimentIntensityAnalyzer from VADER to get polarity scores.

Classification: Converts polarity into sentiment categories.

Evaluation: Compares predictions with actual labels and displays performance.

🛠️ Libraries Used
pandas, numpy, matplotlib, seaborn

nltk, textblob, wordcloud

sklearn: for metrics like confusion matrix

vaderSentiment (through NLTK): for sentiment analysis

📦 How to Run
Make sure to install the necessary packages:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn nltk textblob wordcloud scikit-learn
Then run the Python script:

bash
Copy
Edit
python sentiment_analysis.py
Follow the prompts to analyze reviews.

📸 Example Output
yaml
Copy
Edit
Do you want to write your own review? (y/n): n
Enter the number of reviews to be accessed: 5

Before cleaning:
  review
0 One of the other reviewers has mentioned that ...
...

After cleaning:
  review
0 one reviewers mentioned watching 1 oz episode ...
...

				Results:
  predicted_sentiment  actual_sentiment  is_correct
0          positive          positive         True
...
Correct predictions: 4
Incorrect predictions: 1
📌 Notes
You can adjust pd.set_option('display.max_colwidth', 150) to control how much of the review text is shown in output.

The data_processing() function is customizable for further cleaning (e.g., lemmatization or emoji removal).

