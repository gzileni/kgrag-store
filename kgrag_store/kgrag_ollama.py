import requests
import json
import os
from log import get_logger

logger = get_logger(name="Ollama", loki_url=os.getenv("LOKI_URL"))


def ollama_pull(ollama_url: str, model_name: str) -> tuple[bool, str]:
    """
    Pulls a model from the Ollama server.

    Args:
        ollama_url (str): The base URL of the Ollama server.
        model_name (str): The name of the model to pull.

    Returns:
        dict: The response from the Ollama server.
    """
    payload = {"name": model_name}
    ollama_api = f"{ollama_url}/api/pull"
    error: bool = False
    response: str = ""

    with requests.post(ollama_api, json=payload, stream=True) as r:
        for line in r.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))

                if data is None:
                    response = (
                        f"Model {model_name} not found on Ollama server."
                    )
                    error = True
                    break

                if "error" in data:
                    response = (
                        f"Error pulling model {model_name}: {data['error']}"
                    )
                    error = True
                    break

                if "status" not in data:
                    response = (
                        f"Unexpected response format for model {model_name}: "
                        f"{data}"
                    )
                    error = True
                    break

                if data.get("status") == "success":
                    response = "Modello scaricato con successo!"
                    error = False
                    break

                if data.get("status") == "error":
                    response = f"Errore durante il download: {data}"
                    error = True
                    break

                if data.get("status") == "stream":
                    logger.debug("Streaming output:", data)

    return error, response
