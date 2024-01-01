
from chromadb.config import Settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        anonymized_telemetry=False,
        persist_directory="db"
)


