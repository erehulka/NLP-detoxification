import requests


def callLlamaApi(prompt: str) -> str:
  url = "http://localhost:11434/api/generate"
  data = {
    "model": "llama3",
    "prompt": prompt,
    "stream": False
  }

  try:
    response = requests.post(url, json=data)
    if response.status_code == 200:
      return response.json()['response']
    else:
      print("Error: Unexpected status code:", response.status_code)
  except requests.exceptions.RequestException as e:
    print("Error: Request failed:", e)