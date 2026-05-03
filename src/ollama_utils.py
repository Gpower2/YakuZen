import requests


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"


def call_ollama_model(model_name, prompt, system_prompt="", options=None, timeout=600, ollama_url=DEFAULT_OLLAMA_URL):
    payload_options = {
        "temperature": 0.0,
        "num_ctx": 4096,
        "seed": 0,
    }
    if options:
        payload_options.update(options)

    payload = {
        "model": model_name,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "options": payload_options,
    }

    response = requests.post(ollama_url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json().get("response", "")
