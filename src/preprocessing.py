import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))


def preprocessing(df):
    df = df.copy()
    df.dropna(axis=0, how='all', inplace=True)
    df['source_domain'] = df['source_domain'].str.replace(r'^www\.', '', regex=True)
    df['news_url'] = df['news_url'].str.replace(r'^https?://[^/]+/', '', regex=True)
    df['news_url'] = df['news_url'].str.replace(r'/', ' ')

    return df

# Формирование текстовых фич в одну
def build_text(df):
    df = df.copy()

    df['text'] = (
            df['title'].fillna('') + ' ' +
            df['news_url'].fillna('') + ' ' +
            df['source_domain'].fillna('')
    ).str.strip()
    df = df.drop(columns=['title', 'news_url', 'source_domain', 'tweet_num'])
    return df

# Обработка текста
def preprocess_text(text):
    # нижний регистр
    text = text.lower()
    # Удаление спецсимволов(без заглавных т.к. уже в нижнем регистре)
    text = re.sub(r'[-_/]', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Токенизация
    tokens = text.split()
    # Удаление стоп-слов
    stop_words = STOP_WORDS
    tokens = [token for token in tokens if token not in stop_words]
    return tokens