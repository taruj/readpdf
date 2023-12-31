
from chromadb.config import Settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        anonymized_telemetry=False,
        persist_directory="db"
)
# import os
# from pydantic_settings import BaseSettings

# # If Settings is a custom class defined in your project that inherits from BaseSettings, 
# # ensure it's now inheriting from BaseSettings from pydantic_settings.
# # If it's a predefined class from the chromadb package that internally uses BaseSettings, 
# # the chromadb package itself may need to be updated to be compatible with pydantic-settings.
# from chromadb.config import Settings

# class MyChromaSettings(Settings):
#     # Inherits from the original Settings class.
#     # Add any additional settings or overrides here if needed.

#     CHROMA_SETTINGS = MyChromaSettings(
#         chroma_db_impl = 'duckdb+parquet',
#         persist_db = "db",
#         anonymized_telemetry = False
#     )



# import os
# from chromadb.config import Settings
# # from pydantic_settings import BaseSettings


# CHROMA_SETTINGS = Settings(
#     chroma_db_impl = 'duckdb+parquet',
#     persist_db = "db",
#     anonymized_telemetry = False
# )
    
# )