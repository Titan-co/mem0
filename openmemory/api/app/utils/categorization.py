import json
import logging
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT
import os

load_dotenv()

# Set SSL certificate path for Docker environment
os.environ.setdefault('SSL_CERT_FILE', '/etc/ssl/certs/ca-certificates.crt')

# Use a function to get OpenAI client to delay initialization
def get_openai_client():
    from openai import OpenAI
    return OpenAI()

class MemoryCategories(BaseModel):
    categories: List[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    """Get categories for a memory."""
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT},
                {"role": "user", "content": memory}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        response_content = response.choices[0].message.content
        if not response_content:
            raise ValueError("Empty response from OpenAI API")
            
        response_json = json.loads(response_content)
        categories = response_json['categories']
        return [cat.strip().lower() for cat in categories]
    except Exception as e:
        logging.error(f"Error categorizing memory: {e}")
        raise
