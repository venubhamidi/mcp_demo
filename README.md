# Enhanced MCP Customer Support Agent

This project provides a multi-tool customer support agent that can use different LLM backends. You can run the agent with OpenAI (mcp_openai.py) or with IBM Granite via Replicate (mcp_granite.py).

## Features
- Proactive customer support agent that takes real action (refunds, shipping, emails, etc.)
- Multi-step tool use and reasoning
- Mock Shopify, Stripe, and Email API layers
- Interactive CLI chat interface

## Quick Start

### 1. Clone the repository and install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up your environment variables
Create a `.env` file in the project root with the following for Replicate/Granite:
```
REPLICATE_API_TOKEN=your_replicate_token_here
```

For OpenAI (if you want to use mcp_openai.py):
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the agent

#### To use IBM Granite via Replicate:
```bash
python3 mcp_granite.py
```

#### To use OpenAI (GPT-4o):
```bash
python3 mcp_openai.py
```

## About mcp_granite.py
- Uses the [Replicate](https://replicate.com/) API to access IBM Granite LLMs.
- No client instantiation needed; just set the `REPLICATE_API_TOKEN` in your `.env`.
- The logic and interface are identical to mcp_openai.py, but the LLM backend is IBM Granite.

## .env Example
```
REPLICATE_API_TOKEN=your_replicate_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

## .gitignore
A sample .gitignore is provided to avoid committing secrets and virtual environments.

## License
MIT
