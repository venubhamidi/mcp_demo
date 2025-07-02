# Enhanced MCP Customer Support Agent

## Overview

This project is a simulated, proactive customer support agent that can leverage either IBM Watsonx or OpenAI's GPT models (choose your backend) and a mock Multi-Channel Platform (MCP) API layer. It is designed to handle customer support scenarios for e-commerce businesses, including order status, payment issues, shipping delays, lost packages, refunds, and more. The agent takes immediate action to resolve customer issues and always sends a follow-up email notification.

**Key Features:**
- Proactive, action-oriented customer support agent
- Simulated integration with Shopify (orders), Stripe (payments), and email (notifications)
- Mock data for realistic customer support scenarios
- Automatic planning and execution of multi-step support workflows
- Always includes at least one action (refund, credit, shipping upgrade, etc.) and an email notification
- Interactive CLI chat interface for demo and testing
- **Supports both IBM Watsonx and OpenAI backends**

---

## Running with IBM Watsonx (`mcp_watsonx.py`)

### Requirements
- Python 3.8+
- IBM Watsonx credentials (API key, project ID, URL)

### Python Dependencies
Listed in `requirements.txt`:
```
openai==1.12.0 
httpx==0.24.1
python-dotenv
ibm-watson-machine-learning>=1.0.335
```
Install with:
```bash
pip install -r requirements.txt
```

### Setup
1. **Clone the repository** and navigate to the project directory.
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set your Watsonx environment variables:**
   Create a `.env` file in the project root with the following keys:
   ```env
   WATSONX_APIKEY=your-watsonx-api-key
   WATSONX_PROJECT_ID=your-watsonx-project-id
   WATSONX_URL=https://your-region.ml.cloud.ibm.com
   # Optional: override the default model
   WATSONX_MODEL_ID=ibm/granite-13b-chat-v2
   ```
   Or export them in your shell:
   ```bash
   export WATSONX_APIKEY=your-watsonx-api-key
   export WATSONX_PROJECT_ID=your-watsonx-project-id
   export WATSONX_URL=https://your-region.ml.cloud.ibm.com
   export WATSONX_MODEL_ID=ibm/granite-13b-chat-v2
   ```

### Usage
Run the Watsonx-powered agent:
```bash
python mcp_watsonx.py
```

You will see a demo of available customers and sample queries. Enter a customer email (or use the demo email) and type your support request. The agent will:
- Plan a sequence of tool calls (find customer, check payment, take action, send email)
- Simulate API calls to Shopify, Stripe, and email
- Take immediate action to resolve the issue
- Always send a follow-up email notification
- Respond with a confident, action-oriented message

---

## Running with OpenAI (`mcp_openai.py`)

### Requirements
- Python 3.8+
- OpenAI API key

### Python Dependencies
Listed in `requirements.txt`:
```
openai==1.12.0 
httpx==0.24.1
python-dotenv

```
Install with:
```bash
pip install -r requirements.txt
```

### Setup
1. **Clone the repository** and navigate to the project directory.
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set your OpenAI API key:**
   Export your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

### Usage
Run the main script:
```bash
python mcp_openai.py
```

You will see a demo of available customers and sample queries. Enter a customer email (or use the demo email) and type your support request. The agent will:
- Plan a sequence of tool calls (find customer, check payment, take action, send email)
- Simulate API calls to Shopify, Stripe, and email
- Take immediate action to resolve the issue
- Always send a follow-up email notification
- Respond with a confident, action-oriented message

---

## Customization
- The mock data and tool logic can be extended in `mcp_openai.py` or `mcp_watsonx.py`.
- The agent's planning and synthesis prompts can be modified for different support styles.

## Notes
- This project is for demonstration and prototyping purposes. No real API calls are made; all data is simulated.
- Requires a valid OpenAI or Watsonx API key for chat completions.

## License
MIT License # mcp-demo
# mcp_demo
# mcp_demo
