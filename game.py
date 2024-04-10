import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load data
data = pd.read_csv('essays_dataset.csv')

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)

data['cleaned_text'] = data['essay'].apply(preprocess_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['cleaned_text'])
y = data[['content_score', 'grammar_score', 'other_score']]

# Model building
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

content_model = LinearRegression()
grammar_model = LinearRegression()
other_model = LinearRegression()

content_model.fit(X_train, y_train['content_score'])
grammar_model.fit(X_train, y_train['grammar_score'])
other_model.fit(X_train, y_train['other_score'])

# Model evaluation
content_pred = content_model.predict(X_test)
grammar_pred = grammar_model.predict(X_test)
other_pred = other_model.predict(X_test)

content_mse = mean_squared_error(y_test['content_score'], content_pred)
grammar_mse = mean_squared_error(y_test['grammar_score'], grammar_pred)
other_mse = mean_squared_error(y_test['other_score'], other_pred)

print("Content MSE:", content_mse)
print("Grammar MSE:", grammar_mse)
print("Other MSE:", other_mse)
