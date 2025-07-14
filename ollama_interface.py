import requests

def ask_ollama(prompt, model='mistral'):
    try:
        test_response = requests.get("http://localhost:11434", timeout=5)
        if test_response.status_code != 200:
            return "Ollama is not ready. Please wait a moment and try again."
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=180
        )
        response.raise_for_status()
        result = response.json()
        return result['response'].strip()
    except requests.exceptions.ConnectionError:
        return "Ollama server is not running. Please start Ollama to enable AI analysis."
    except requests.exceptions.ReadTimeout:
        return "Ollama timed out. It might be busy loading a model or just slow. Try again in a few seconds."
    except Exception as e:
        return f"Ollama Error: {str(e)}"