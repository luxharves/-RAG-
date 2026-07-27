"""Workaround for ragas 0.4.3 import error on langchain_community.chat_models.vertexai.

Must be imported BEFORE any ragas import.
"""

import sys
import types as _types

_stub = _types.ModuleType("langchain_community.chat_models.vertexai")

class _FakeChatVertexAI:
    pass

_stub.ChatVertexAI = _FakeChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _stub
