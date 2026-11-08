"""Settings, loaded from environment variables or a `.env` file.

Any OpenAI-compatible endpoint works: LM Studio, Ollama, vLLM, OpenAI, DeepSeek,
OpenRouter... Switching between local and cloud is a config change only.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOCASSIST_", env_file=".env", extra="ignore")

    # Chat model (answers, extraction, query routing)
    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "qwen3.5-9b-mtp"
    # Sent as `reasoning_effort` on every chat call. "none" turns off the thinking
    # phase of hybrid reasoning models (Qwen 3.x) - much faster and fine for RAG.
    # Leave empty for models/providers that reject the parameter.
    llm_reasoning_effort: str = "none"
    llm_timeout: float = 180.0
    # Model requests in flight at once (uploads process several documents and the parts of a
    # long document side by side). 0 = automatic: recommended for this machine's hardware
    # when the chat model is local, 4 for a cloud API. At most 10.
    concurrency: int = 0

    # Embedding model - may live on a different endpoint than the chat model.
    embed_base_url: str | None = None  # defaults to llm_base_url
    embed_api_key: str | None = None
    embed_model: str = "text-embedding-nomic-embed-text-v1.5"
    # nomic-embed needs task prefixes; set both to "" for models that don't.
    embed_doc_prefix: str = "search_document: "
    embed_query_prefix: str = "search_query: "

    data_dir: Path = Path("data/store")
    # Model presets the UI can switch between (see profiles.example.yaml); None = one
    # profile from the settings above, stored directly in data_dir.
    profiles_path: Path | None = None
    # Extraction schema (one YAML per document type); None = schemas/contract.yaml.
    schema_path: Path | None = None
    chunk_tokens: int = 220
    chunk_overlap: int = 40
    top_k: int = 6
    # Pipeline switches - the eval report compares "naive" vs "full".
    hybrid: bool = True  # BM25 + vectors (False = vectors only)
    context_headers: bool = True  # prepend document identity to each chunk
    doc_scoping: bool = True  # question names a document (ID, counterparty) -> search only that document
    structured: bool = True  # route aggregate questions to the extracted table
    # A question that names one document is answered from the whole document (plus its
    # extracted record) when it is at most this many characters; longer ones use search.
    # 0 = always search.
    full_doc_chars: int = 60_000
    # A "yes" for a clause (boolean field) counts only if the model's quote for it is
    # found in the document; otherwise the value is stored as unknown.
    require_clause_quote: bool = True

    @property
    def db_path(self) -> Path:
        return self.data_dir / "docassist.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
