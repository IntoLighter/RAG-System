import requests

url = "http://localhost:8001/v1/chat/completions"
payload = {
    "model": "Qwen/Qwen3-4B-AWQ",
    "messages": [
        {"role": "user", "content": "Say hi in one word"},
    ],
}

resp = requests.post(url, json=payload, timeout=60)
resp.raise_for_status()
data = resp.json()

print(data["choices"][0]["message"]["content"])
