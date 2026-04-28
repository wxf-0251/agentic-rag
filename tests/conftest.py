import os

# Set test environment variables before any imports
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
