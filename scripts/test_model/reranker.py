import requests

payload = {
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "query": "Как увеличить производительность Python?",
    "documents": [
        "Python можно ускорить с помощью PyPy и оптимизации алгоритмов.",
        "Для увеличения производительности Python часто используют "
        "multiprocessing и Cython.",
        "Docker позволяет запускать приложения в изолированных контейнерах.",
        "Для работы с базами данных в Python можно использовать SQLAlchemy.",
    ],
    "top_n": 4,
}

resp = requests.post(
    "http://localhost:8003/v1/rerank",
    json=payload,
    timeout=60,
)
print(resp.status_code, resp.text)
