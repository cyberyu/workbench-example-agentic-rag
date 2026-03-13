#!/bin/bash
# Startup script for Agentic RAG application

# Set API keys
export NVIDIA_API_KEY='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'
export TAVILY_API_KEY='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'

# Set recursion limit (default is 10, increase for complex queries)
export RECURSION_LIMIT=20

# Navigate to code directory and start the application
cd "$(dirname "$0")/code"
python -m chatui --host 0.0.0.0 --port 8080
