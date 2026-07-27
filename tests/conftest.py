"""Test configuration — ensures pymilvus can import cleanly."""

import os

# pymilvus ORM module reads MILVUS_URI at import time and only accepts
# http:// or https:// URIs. Our .env sets "milvus.db" for app use,
# so we must set a valid dummy URI before any pymilvus import.
os.environ.setdefault("MILVUS_URI", "http://localhost:19530")
