import requests

url = "http://localhost:8002/v1/embeddings"
payload = {
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": "Тест на русском",
}

resp = requests.post(url, json=payload, timeout=60)
resp.raise_for_status()
data = resp.json()

embedding = data["data"][0]["embedding"]
print(f"Длина вектора: {len(embedding)}")
print(embedding[:10], "...")
