import os
import sys

import httpx
from dotenv import load_dotenv

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "nvidia/nemotron-3-super-120b-a12b:free"


def main() -> None:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY is not set. Add it to .env in the project root.")

    request_body = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user", 
                "content": "Respond with exactly: connection successful"
            }
        ],
    }

    # httpx's default timeout is 5 seconds; free models are often slower than that.
    response = httpx.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=request_body,
        timeout=60,
    )

    print(f"HTTP status: {response.status_code}")

    if response.status_code != 200:
        sys.exit(f"Request failed: {response.text}")

    response_data = response.json()
    print(f"Model: {response_data['model']}")
    print(f"Response: {response_data['choices'][0]['message']['content']}")


if __name__ == "__main__":
    main()
