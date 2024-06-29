# for data analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# for models
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from wordcloud import WordCloud


class AppReviewModel:
    def prepare_data(self, train_data: pd.DataFrame):
        # Балансировка классов
        positive = train_data[train_data.Sentiment == 'Positive']
        negative = train_data[train_data.Sentiment == 'Negative']
        neutral = train_data[train_data.Sentiment == 'Neutral']

        # Апсемплинг классов
        negative_upsampled = resample(negative, replace=True, n_samples=len(positive), random_state=42)
        neutral_upsampled = resample(neutral, replace=True, n_samples=len(positive), random_state=42)

        # Объединение сбалансированных данных
        balanced_upsampled = pd.concat([positive, negative_upsampled, neutral_upsampled])

        return balanced_upsampled

    def __init__(self, path_to_data_for_model: str):
        self.df = pd.read_csv(path_to_data_for_model)

        X_train, X_test, y_train, y_test = train_test_split(self.df['processed_review'], self.df['Sentiment'], test_size=0.2, random_state=42)

        train_data = pd.concat([X_train, y_train], axis=1)

        # Балансировка данных
        self.balanced_upsampled = self.prepare_data(train_data)

    def turn_sentiment_prediction(self):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('scaler', StandardScaler(with_mean=False)),
            ('rf', RandomForestClassifier())
        ])

        # Определение гиперпараметров для оптимизации
        param_grid = {
            'rf__n_estimators': [100, 200, 300],
            'rf__max_depth': [10, 20, 30],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4]
        }

        # Настройка GridSearchCV
        self.model = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)

        # Обучение модели на сбалансированных данных
        self.model.fit(self.balanced_upsampled['processed_review'], self.balanced_upsampled['Sentiment'])

    def predict_sentiment(self, X_data: pd.Series, true_y_data: pd.Series, show_report: bool):
        predictions = self.model.predict(X_data)

        if show_report:
            classification_report_result = classification_report(true_y_data, predictions, zero_division=0)

            print(classification_report_result)
            print("Best parameters found: ", self.model.best_params_)
        
        return predictions
    
    def create_wordcloud(self, isPositive: bool):
        if isPositive:
            # Wordcloud для положительных отзывов
            positive_reviews = ' '.join(self.df[self.df['Sentiment'] == 'Positive']['processed_review'])
            positive_wordcloud = WordCloud(background_color='white', colormap='Greens').generate(positive_reviews)

            plt.figure(figsize=(10, 6))
            plt.imshow(positive_wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Wordcloud for Positive Reviews')
            plt.show()
        else:
            # Wordcloud для негативных отзывов
            negative_reviews = ' '.join(df[df['Sentiment'] == 'Negative']['processed_review'])
            negative_wordcloud = WordCloud(background_color='white', colormap='Reds').generate(positive_reviews)

            plt.figure(figsize=(10, 6))
            plt.imshow(negative_wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Wordcloud for Negative Reviews')
            plt.show()
