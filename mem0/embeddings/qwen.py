import os
from typing import Literal, Optional

from openai import OpenAI

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase


class QwenEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)
        
        # Set default model for Qwen embeddings
        self.model = self.config.model or "text-embedding-v4"
        
        # Get API key from config or environment variable
        api_key = self.config.api_key or os.getenv("QWEN_API_KEY")
        if not api_key:
            raise ValueError("QWEN_API_KEY not set")
        
        # Get base URL from config or use default DashScope URL
        base_url = self.config.openai_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # Initialize OpenAI client with Qwen-specific settings
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, text: str, memory_action: Optional[Literal["add", "search", "update"]] = None) -> list:
        """
        Get the embedding for the given text using Qwen's DashScope API.
        
        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use.
                Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        if not text:
            return []
            
        # Clean text by removing newlines
        text = text.replace("\n", " ")
        
        # Create embedding (Qwen API doesn't support dimension parameter)
        response = self.client.embeddings.create(
            input=[text],
            model=self.model
        )
        
        return response.data[0].embedding