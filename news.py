import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# 뉴스 기사(본문일수록 정확도 높음)
news_article = "‘쏘니’ 드디어 돌아오나? AV전 출전 가능성↑ 포스테코글루 감독 “훈련 소화했어, 마지막 훈련만 통과하면 괜찮을 것”"

# 키워드 로드
with open('news.json', 'r', encoding='utf-8') as f:
    keywords = json.load(f)


category_keywords = {cat: ' '.join(words) for cat, words in keywords.items()}
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(category_keywords.values())


# 유사도 확인 작업
news_vector = vectorizer.transform([news_article])
cosine_similarities = cosine_similarity(news_vector, X)


# 코사인 유사도
most_similar_category_index = cosine_similarities.argmax()
most_similar_category = list(category_keywords.keys())[most_similar_category_index]


print(f"뉴스 기사는 '{most_similar_category}' 카테고리에 속합니다.")