#!/usr/bin/env python3

import json
import asyncio
import logging
from typing import Dict, List, Any
import os
from datetime import datetime
from dotenv import load_dotenv
import replicate

# Load environment variables from .env file
load_dotenv()

# Setup minimal logging
logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)

def log_summary(title: str, data: Any):
    print(f"\n🔍 {title}")
    print("-" * 50)
    if isinstance(data, dict):
        if "execution_plan" in data:
            for i, step in enumerate(data["execution_plan"], 1):
                print(f"  {i}. {step['tool']} - {step['reasoning']}")
        elif "jsonrpc" in data:
            method = data.get("params", {}).get("name", data.get("method", "unknown"))
            print(f"  Method: {method}")
            if "arguments" in data.get("params", {}):
                print(f"  Args: {data['params']['arguments']}")
        elif "result" in data or "error" in data:
            if "error" in data:
                print(f"  Error: {data['error']['message']}")
            else:
                content = data["result"]["content"][0]["text"]
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and len(parsed) <= 3:
                        print(f"  Result: {parsed}")
                    else:
                        print(f"  Result: {type(parsed).__name__} with {len(str(parsed))} chars")
                except:
                    print(f"  Result: {len(content)} chars")
        else:
            if "method" in data and "url" in data:
                print(f"  {data['method']} {data['url']}")
                if "params" in data:
                    print(f"  Params: {data['params']}")
            elif "status" in data:
                print(f"  Status: {data['status']}")
                if "body" in data and isinstance(data["body"], dict):
                    print(f"  Data: {data['body']}")
    else:
        print(f"  {data}")
    print("-" * 50)

def get_env_var(key: str, default: str = None) -> str:
    value = os.getenv(key)
    if value is None:
        if default is not None:
            return str(default)
        raise ValueError(f"Missing required environment variable: {key}")
    return str(value)

# Replicate LLM Client
class ReplicateLLMClient:
    def __init__(self):
        self.model = "ibm-granite/granite-3.3-8b-instruct"
        self.api_token = get_env_var("REPLICATE_API_TOKEN")
        # replicate uses the environment variable automatically

    async def chat_completions_create(self, messages: list, temperature: float = 0.1):
        prompt = "\n".join([m["content"] for m in messages])
        input_data = {
            "prompt": prompt,
            "max_new_tokens": 2000,
            "min_tokens": 200,
            "temperature": temperature,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }
        import asyncio
        loop = asyncio.get_event_loop()
        # replicate.run is synchronous, so run in executor
        def run_replicate():
            output = replicate.run(self.model, input=input_data)
            if hasattr(output, '__iter__') and not isinstance(output, str):
                return "".join(output)
            return output
        response = await loop.run_in_executor(None, run_replicate)
        return {"choices": [{"message": {"content": response}}]}

# --- Mock data and tool servers (copied from mcp_agent5.py) ---
mock_data = {
    "shopify": {
        "customers": [
            {
                "id": "customer_001", "email": "john@email.com", "first_name": "John", "last_name": "Smith",
                "orders": [{"order_number": "1001", "id": "order_001", "status": "delivered", "product": "Wireless Headphones", "amount": 179.99, "tracking": "1Z999AA1234567890", "shipped_date": "2024-05-20", "delivered_date": "2024-05-22"}]
            },
            {
                "id": "customer_002", "email": "sarah@email.com", "first_name": "Sarah", "last_name": "Johnson",
                "orders": [{"order_number": "1002", "id": "order_002", "status": "shipping_delayed", "product": "Smart Watch", "amount": 299.99, "tracking": "1Z999BB9876543210", "shipped_date": "2024-05-25", "expected_delivery": "2024-06-10", "delay_reason": "Weather conditions affecting shipping hub"}]
            },
            {
                "id": "customer_003", "email": "mike@email.com", "first_name": "Mike", "last_name": "Brown",
                "orders": [{"order_number": "1003", "id": "order_003", "status": "payment_failed", "product": "Gaming Laptop", "amount": 1299.99, "tracking": None, "payment_retry_url": "https://checkout.company.com/retry/1003"}]
            },
            {
                "id": "customer_004", "email": "lisa@email.com", "first_name": "Lisa", "last_name": "Davis",
                "orders": [{"order_number": "1004", "id": "order_004", "status": "lost_in_transit", "product": "Bluetooth Speaker", "amount": 89.99, "tracking": "1Z999CC5432167890", "shipped_date": "2024-05-18", "last_tracking_update": "Package departed carrier facility - May 20, 2024"}]
            },
            {
                "id": "customer_005", "email": "alex@email.com", "first_name": "Alex", "last_name": "Wilson",
                "orders": [{"order_number": "1005", "id": "order_005", "status": "cancelled_by_customer", "product": "Meta Glasses", "amount": 349.99, "cancelled_date": "2024-05-23", "refund_status": "processing"}]
            }
        ]
    },
    "stripe": {
        "customers": [
            {"id": "cus_001", "email": "john@email.com", "payment_methods": [{"id": "pm_001", "type": "card", "last4": "4242", "brand": "visa", "status": "active"}], "charges": [{"id": "ch_001", "amount": 17999, "currency": "usd", "status": "succeeded", "description": "Order #1001", "created": "2024-05-20"}]},
            {"id": "cus_002", "email": "sarah@email.com", "payment_methods": [{"id": "pm_002", "type": "card", "last4": "5555", "brand": "mastercard", "status": "active"}], "charges": [{"id": "ch_002", "amount": 29999, "currency": "usd", "status": "succeeded", "description": "Order #1002", "created": "2024-05-25"}]},
            {"id": "cus_003", "email": "mike@email.com", "payment_methods": [{"id": "pm_003", "type": "card", "last4": "1234", "brand": "visa", "status": "expired"}], "charges": [{"id": "ch_003", "amount": 129999, "currency": "usd", "status": "failed", "description": "Order #1003", "failure_code": "card_declined", "failure_message": "Your card was declined.", "created": "2024-05-28"}]},
            {"id": "cus_004", "email": "lisa@email.com", "payment_methods": [{"id": "pm_004", "type": "card", "last4": "9999", "brand": "amex", "status": "active"}], "charges": [{"id": "ch_004", "amount": 8999, "currency": "usd", "status": "succeeded", "description": "Order #1004", "created": "2024-05-18"}]},
            {"id": "cus_005", "email": "alex@email.com", "payment_methods": [{"id": "pm_005", "type": "card", "last4": "7777", "brand": "visa", "status": "active"}], "charges": [{"id": "ch_005", "amount": 14999, "currency": "usd", "status": "refunded", "description": "Order #1005", "refunded": True, "refund_amount": 14999, "created": "2024-05-23"}]}
        ]
    },
    "email": {"templates": {"order_update": {"subject": "Order Update for {{customer_name}}", "body": "Hi {{customer_name}}! Your order {{order_number}} has been updated."}}, "sent_emails": []}
}

class ShopifyMCPServer:
    def __init__(self):
        self.name = "shopify-server"
        self.version = "1.0.0"

    async def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        request_id = f"req_{datetime.now().timestamp()}"
        mcp_request = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": tool_name, "arguments": args}, "id": request_id}
        log_summary(f"[{self.name}] MCP REQUEST", mcp_request)
        response = {"jsonrpc": "2.0", "id": request_id, "result": None}
        try:
            if tool_name == "find_customer":
                api_request = {"method": "GET", "url": f"https://{self.name}.myshopify.com/admin/api/2023-01/customers.json", "params": {"email": args["email"]}}
                log_summary(f"[{self.name}] API REQUEST", api_request)
                customer = next((c for c in mock_data["shopify"]["customers"] if c["email"] == args["email"]), None)
                api_response = {"status": 200 if customer else 404, "body": {"customers": [customer] if customer else [], "count": 1 if customer else 0}}
                log_summary(f"[{self.name}] API RESPONSE", api_response)
                response["result"] = {"content": [{"type": "text", "text": json.dumps(customer)}]}
            elif tool_name == "get_order_status":
                api_request = {"method": "GET", "url": f"https://{self.name}.myshopify.com/admin/api/2023-01/orders.json", "params": {"name": args["order_number"], "email": args["customer_email"]}}
                log_summary(f"[{self.name}] API REQUEST", api_request)
                customer = next((c for c in mock_data["shopify"]["customers"] if c["email"] == args["customer_email"]), None)
                order = None
                if customer:
                    order = next((o for o in customer["orders"] if o["order_number"] == args["order_number"]), None)
                api_response = {"status": 200 if order else 404, "body": {"orders": [order] if order else [], "count": 1 if order else 0}}
                log_summary(f"[{self.name}] API RESPONSE", api_response)
                response["result"] = {"content": [{"type": "text", "text": json.dumps(order)}]}
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as error:
            response["error"] = {"code": -32603, "message": "Internal error", "data": str(error)}
        return response

class StripeMCPServer:
    def __init__(self):
        self.name = "stripe-server"
        self.version = "1.0.0"

    async def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        request_id = f"req_{datetime.now().timestamp()}"
        mcp_request = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": tool_name, "arguments": args}, "id": request_id}
        log_summary(f"[{self.name}] MCP REQUEST", mcp_request)
        response = {"jsonrpc": "2.0", "id": request_id, "result": None}
        try:
            if tool_name == "get_customer_payments":
                api_request = {"method": "GET", "url": "https://api.stripe.com/v1/customers/search", "params": {"query": f"email:'{args['email']}'", "expand": ["data.payment_methods", "data.charges"]}}
                log_summary(f"[{self.name}] API REQUEST", api_request)
                customer = next((c for c in mock_data["stripe"]["customers"] if c["email"] == args["email"]), None)
                payment_data = {"payment_methods": customer["payment_methods"] if customer else [], "charges": customer["charges"] if customer else []}
                api_response = {"status": 200, "body": {"object": "customer", "payment_methods": payment_data["payment_methods"], "charges": {"object": "list", "data": payment_data["charges"]}}}
                log_summary(f"[{self.name}] API RESPONSE", api_response)
                response["result"] = {"content": [{"type": "text", "text": json.dumps(payment_data)}]}
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as error:
            response["error"] = {"code": -32603, "message": "Internal error", "data": str(error)}
        return response

class ActionMCPServer:
    def __init__(self):
        self.name = "action-server"
        self.version = "1.0.0"

    async def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        request_id = f"req_{datetime.now().timestamp()}"
        mcp_request = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": tool_name, "arguments": args}, "id": request_id}
        log_summary(f"[{self.name}] MCP REQUEST", mcp_request)
        response = {"jsonrpc": "2.0", "id": request_id, "result": None}
        try:
            if tool_name == "process_refund":
                api_request = {"method": "POST", "url": "https://api.stripe.com/v1/refunds", "body": {"charge": args["charge_id"], "amount": args.get("amount"), "reason": args.get("reason", "requested_by_customer")}}
                log_summary(f"[{self.name}] API REQUEST", api_request)
                refund_result = {"refund_id": f"re_{datetime.now().timestamp()}", "amount": args.get("amount", "full"), "status": "processing", "estimated_arrival": "1-2 business days", "expedited": args.get("expedite", False)}
                api_response = {"status": 200, "body": {"object": "refund", "status": "succeeded", **refund_result}}
                log_summary(f"[{self.name}] API RESPONSE", api_response)
                response["result"] = {"content": [{"type": "text", "text": json.dumps(refund_result)}]}
            elif tool_name == "retry_payment":
                api_request = {"method": "POST", "url": "https://api.stripe.com/v1/payment_intents", "body": {"customer": args["customer_id"], "payment_method": args.get("payment_method"), "confirm": True}}
                log_summary(f"[{self.name}] API REQUEST", api_request)
                payment_result = {"payment_id": f"pi_{datetime.now().timestamp()}", "status": "succeeded", "payment_method": args.get("payment_method", "backup_card"), "amount_charged": args.get("amount"), "discount_applied": args.get("discount", 0)}
                api_response = {"status": 200, "body": {"object": "payment_intent", "status": "succeeded", **payment_result}}
                log_summary(f"[{self.name}] API RESPONSE", api_response)
                response["result"] = {"content": [{"type": "text", "text": json.dumps(payment_result)}]}
            elif tool_name == "upgrade_shipping":
                api_request = {"method": "PUT", "url": f"https://api.shopify.com/orders/{args['order_id']}/shipping", "body": {"shipping_method": args["new_method"], "cost_adjustment": 0}}
                log_summary(f"[{self.name}] API REQUEST", api_request)
                shipping_result = {"order_id": args["order_id"], "old_method": args.get("old_method", "standard"), "new_method": args["new_method"], "cost_difference": "waived", "new_delivery_date": args.get("new_delivery_date", "2-3 business days")}
                api_response = {"status": 200, "body": {"success": True, **shipping_result}}
                log_summary(f"[{self.name}] API RESPONSE", api_response)
                response["result"] = {"content": [{"type": "text", "text": json.dumps(shipping_result)}]}
            elif tool_name == "ship_replacement":
                api_request = {"method": "POST", "url": "https://api.shopify.com/orders", "body": {"customer_id": args["customer_id"], "product": args["product"], "shipping_method": "overnight", "reason": args.get("reason", "lost_package")}}
                log_summary(f"[{self.name}] API REQUEST", api_request)
                replacement_result = {"new_order_id": f"repl_{datetime.now().timestamp()}", "original_order": args.get("original_order"), "product": args["product"], "shipping_method": "overnight", "tracking_number": f"1Z999REP{datetime.now().strftime('%Y%m%d')}", "estimated_delivery": "tomorrow by 10 AM"}
                api_response = {"status": 201, "body": {"success": True, **replacement_result}}
                log_summary(f"[{self.name}] API RESPONSE", api_response)
                response["result"] = {"content": [{"type": "text", "text": json.dumps(replacement_result)}]}
            elif tool_name == "apply_credit":
                api_request = {"method": "POST", "url": "https://api.shopify.com/customers/store_credit", "body": {"customer_id": args["customer_id"], "amount": args["amount"], "reason": args.get("reason", "service_recovery")}}
                log_summary(f"[{self.name}] API REQUEST", api_request)
                credit_result = {"credit_id": f"cr_{datetime.now().timestamp()}", "customer_id": args["customer_id"], "amount": args["amount"], "type": args.get("type", "service_credit"), "expires": args.get("expires", "1 year"), "available_immediately": True}
                api_response = {"status": 200, "body": {"success": True, **credit_result}}
                log_summary(f"[{self.name}] API RESPONSE", api_response)
                response["result"] = {"content": [{"type": "text", "text": json.dumps(credit_result)}]}
            elif tool_name == "enable_vip_status":
                api_request = {"method": "PUT", "url": f"https://api.shopify.com/customers/{args['customer_id']}/vip", "body": {"vip_tier": args.get("tier", "gold"), "benefits": ["priority_support", "free_shipping", "early_access"]}}
                log_summary(f"[{self.name}] API REQUEST", api_request)
                vip_result = {"customer_id": args["customer_id"], "vip_tier": args.get("tier", "gold"), "benefits": ["Priority support", "Free shipping on all orders", "Early access to new products"], "effective_immediately": True, "welcome_bonus": "20% off next order"}
                api_response = {"status": 200, "body": {"success": True, **vip_result}}
                log_summary(f"[{self.name}] API RESPONSE", api_response)
                response["result"] = {"content": [{"type": "text", "text": json.dumps(vip_result)}]}
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as error:
            response["error"] = {"code": -32603, "message": "Internal error", "data": str(error)}
        return response

class EmailMCPServer:
    def __init__(self):
        self.name = "email-server"
        self.version = "1.0.0"

    async def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        request_id = f"req_{datetime.now().timestamp()}"
        mcp_request = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": tool_name, "arguments": args}, "id": request_id}
        log_summary(f"[{self.name}] MCP REQUEST", mcp_request)
        response = {"jsonrpc": "2.0", "id": request_id, "result": None}
        try:
            if tool_name == "send_order_update":
                api_request = {"method": "POST", "url": "https://api.sendgrid.com/v3/mail/send", "body": {"personalizations": [{"to": [{"email": args["to"]}], "dynamic_template_data": {"customer_name": args["customer_name"], "order_number": args["order_number"]}}], "template_id": "d-order_update"}}
                log_summary(f"[{self.name}] API REQUEST", api_request)
                update_email = {"to": args["to"], "subject": f"Order Update - {args['order_number']}", "body": f"Hi {args['customer_name']}! Your order {args['order_number']} has been updated.", "sent_at": datetime.now().isoformat()}
                mock_data["email"]["sent_emails"].append(update_email)
                api_response = {"status": 202, "body": {"message": "Order update email queued for delivery", "message_id": f"msg_{request_id}"}}
                log_summary(f"[{self.name}] API RESPONSE", api_response)
                response["result"] = {"content": [{"type": "text", "text": json.dumps({"success": True, "email_id": f"email_{datetime.now().timestamp()}", "message": "Order update email sent"})}]}
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as error:
            response["error"] = {"code": -32603, "message": "Internal error", "data": str(error)}
        return response

class MCPClient:
    def __init__(self):
        self.servers = {"shopify-server": ShopifyMCPServer(), "stripe-server": StripeMCPServer(), "email-server": EmailMCPServer(), "action-server": ActionMCPServer()}

    async def call_tool(self, server_name: str, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if server_name not in self.servers:
            raise ValueError(f"MCP Server not found: {server_name}")
        server = self.servers[server_name]
        response = await server.handle_tool_call(tool_name, args)
        if "error" in response:
            raise Exception(f"MCP Error: {response['error']['message']}")
        return response["result"]

class UnifiedCustomerSupportAgent:
    def __init__(self, llm_client):
        self.mcp_client = MCPClient()
        self.llm_client = llm_client
        print("✅ Enhanced Customer Support Agent (Replicate) initialized successfully")

    def get_available_tools(self) -> Dict[str, Any]:
        return {
            "shopify-server": {
                "find_customer": {"description": "Find customer by email address", "parameters": {"email": "string"}},
                "get_order_status": {"description": "Get order status by order number and customer email", "parameters": {"order_number": "string", "customer_email": "string"}}
            },
            "stripe-server": {"get_customer_payments": {"description": "Get customer payment history and methods", "parameters": {"email": "string"}}},
            "email-server": {"send_order_update": {"description": "Send order update notification", "parameters": {"to": "string", "customer_name": "string", "order_number": "string"}}},
            "action-server": {
                "process_refund": {"description": "Process immediate refund for customer", "parameters": {"charge_id": "string", "amount": "number", "reason": "string"}},
                "retry_payment": {"description": "Retry failed payment with backup method", "parameters": {"customer_id": "string", "payment_method": "string", "amount": "number"}},
                "upgrade_shipping": {"description": "Upgrade shipping method at no charge", "parameters": {"order_id": "string", "new_method": "string"}},
                "ship_replacement": {"description": "Ship replacement item immediately", "parameters": {"customer_id": "string", "product": "string", "original_order": "string"}},
                "apply_credit": {"description": "Apply store credit to customer account", "parameters": {"customer_id": "string", "amount": "string", "reason": "string"}},
                "enable_vip_status": {"description": "Upgrade customer to VIP status", "parameters": {"customer_id": "string", "tier": "string"}}
            }
        }

    def _create_fallback_plan(self, customer_email: str) -> List[Dict[str, Any]]:
        return [
            {"tool": "shopify-server.find_customer", "args": {"email": customer_email}, "reasoning": "Look up customer information"},
            {"tool": "stripe-server.get_customer_payments", "args": {"email": customer_email}, "reasoning": "Check payment status"},
            {"tool": "action-server.apply_credit", "args": {"customer_id": "{{customer_id}}", "amount": "$10", "reason": "proactive_service"}, "reasoning": "Apply proactive service credit"},
            {"tool": "email-server.send_order_update", "args": {"to": customer_email, "customer_name": "{{customer_name}}", "order_number": "{{order_number}}"}, "reasoning": "Send email confirmation"}
        ]

    def _ensure_action_and_email_steps(self, tool_plan: List[Dict[str, Any]], customer_email: str) -> List[Dict[str, Any]]:
        has_email = any(step["tool"].startswith("email-server") for step in tool_plan)
        has_action = any(step["tool"].startswith("action-server") for step in tool_plan)
        if not has_action:
            print("⚠️ Adding proactive service credit...")
            action_step = {"tool": "action-server.apply_credit", "args": {"customer_id": "{{customer_id}}", "amount": "$10", "reason": "excellent_service"}, "reasoning": "Auto-added: Apply proactive service credit"}
            tool_plan.insert(-1 if has_email else len(tool_plan), action_step)
        if not has_email:
            print("⚠️ Adding email notification...")
            email_step = {"tool": "email-server.send_order_update", "args": {"to": customer_email, "customer_name": "{{customer_name}}", "order_number": "{{order_number}}"}, "reasoning": "Auto-added: Send email notification"}
            tool_plan.append(email_step)
        return tool_plan

    def _resolve_placeholders(self, args: Dict[str, Any], execution_results: Dict[str, Any]) -> Dict[str, Any]:
        resolved = args.copy()
        customer = execution_results.get("shopify-server.find_customer")
        if resolved.get("customer_name") == "{{customer_name}}":
            resolved["customer_name"] = customer.get("first_name", "Valued Customer") if customer else "Valued Customer"
        if resolved.get("order_number") in ("{{order_number}}", "unknown", "ORDER_NUMBER_PLACEHOLDER"):
            order = execution_results.get("shopify-server.get_order_status")
            if order:
                resolved["order_number"] = order.get("order_number", "Unknown")
            elif customer and customer.get("orders"):
                resolved["order_number"] = customer["orders"][0].get("order_number", "Unknown")
            else:
                resolved["order_number"] = "General Inquiry"
        if resolved.get("order_id") in ("{{order_id}}", "unknown", "ORDER_NUMBER_PLACEHOLDER"):
            order = execution_results.get("shopify-server.get_order_status")
            if order:
                resolved["order_id"] = order.get("id", "unknown")
            elif customer and customer.get("orders"):
                resolved["order_id"] = customer["orders"][0].get("id", "unknown")
            else:
                resolved["order_id"] = "unknown"
        if resolved.get("customer_id") in ("{{customer_id}}", "unknown", "ORDER_NUMBER_PLACEHOLDER"):
            resolved["customer_id"] = customer.get("id", "unknown") if customer else "unknown"
        return resolved

    async def handle_request(self, customer_email: str, request: str) -> str:
        try:
            print(f"\n🚀 Processing: '{request}' for {customer_email}")
            planning_prompt = f"""You are a proactive customer support AI agent with the power to take immediate action. Analyze this customer request and plan which tools to use.

Available tools: {json.dumps(self.get_available_tools(), indent=2)}

Customer request: \"{request}\"
Customer email: {customer_email}

You are EMPOWERED to take immediate action to solve customer problems. Plan a comprehensive response that includes:
1. Information gathering (shopify-server.find_customer, get_order_status, stripe-server.get_customer_payments)
2. PROACTIVE PROBLEM SOLVING with action-server tools based on order status:
   - For \"payment_failed\": retry_payment, apply_credit  
   - For \"shipping_delayed\": upgrade_shipping, apply_credit
   - For \"lost_in_transit\": ship_replacement, apply_credit
   - For \"cancelled_by_customer\": process_refund
   - For \"delivered\" or happy customers: enable_vip_status, apply_credit
3. Enhanced communication (email-server.send_order_update)

IMPORTANT: First gather customer and order data to determine the correct order status. **After you know the order status, ONLY include the single action-server tool that matches the actual status. Do NOT include all possible actions.**
ALWAYS include at least one action-server tool to proactively solve the customer's problem.
ALWAYS end with email notification.

Respond with a JSON array of tool plans in this exact format:
[{{"tool": "server-name.tool_name", "args": {{"param1": "value1"}}, "reasoning": "Why you chose this tool and what action you're taking"}}]"""
            print(f"\n Sending request to Granite...")
            response = await self.llm_client.chat_completions_create(
                messages=[
                    {"role": "system", "content": "You are a proactive customer support AI that takes immediate action to solve problems. ALWAYS include action-server tools to resolve customer issues. ALWAYS include an email step in every plan."},
                    {"role": "user", "content": planning_prompt}
                ],
                temperature=0.1
            )
            tool_plan_response = response["choices"][0]["message"]["content"]
            print(f"\n🤖 Granite Raw Response: {tool_plan_response}")
            try:
                if not isinstance(tool_plan_response, str):
                    tool_plan_response = str(tool_plan_response)
                if "```json" in tool_plan_response:
                    tool_plan_response = tool_plan_response.split("```json")[1].split("```", 1)[0].strip()
                elif "```" in tool_plan_response:
                    tool_plan_response = tool_plan_response.split("```", 1)[1].split("```", 1)[0].strip()
                tool_plan = json.loads(tool_plan_response)
                print(f"\n🔍 Parsed Tool Plan: {json.dumps(tool_plan, indent=2)}")
                tool_plan = self._ensure_action_and_email_steps(tool_plan, customer_email)
            except json.JSONDecodeError as e:
                print(f"❌ JSON Parse Error: {e}")
                print("🔄 Using comprehensive fallback plan...")
                tool_plan = self._create_fallback_plan(customer_email)
            log_summary("AGENT EXECUTION PLAN", {"execution_plan": tool_plan})
            execution_results = {}
            for i, step in enumerate(tool_plan, 1):
                print(f"\n🔧 Step {i}: {step['reasoning']}")
                try:
                    server_name, tool_name = step["tool"].split(".")
                    resolved_args = self._resolve_placeholders(step["args"], execution_results)
                    result = await self.mcp_client.call_tool(server_name, tool_name, resolved_args)
                    parsed_result = json.loads(result["content"][0]["text"])
                    execution_results[step["tool"]] = parsed_result
                    print(f"✅ Step {i} completed successfully")
                except Exception as error:
                    print(f"❌ Step {i} failed: {error}")
                    execution_results[step["tool"]] = {"error": str(error)}
            email_sent = any(key.startswith("email-server") and "error" not in result for key, result in execution_results.items())
            synthesis_prompt = f"""You are a proactive customer support AI with the power to take immediate action. Based on the data you gathered and actions you took, create a confident, action-oriented response to the customer.

Original customer request: \"{request}\"
Customer email: {customer_email}

Data gathered:
Customer data: {json.dumps(execution_results.get('shopify-server.find_customer'))}
Order data: {json.dumps(execution_results.get('shopify-server.get_order_status'))}
Payment data: {json.dumps(execution_results.get('stripe-server.get_customer_payments'))}

Actions taken:
Refund processed: {json.dumps(execution_results.get('action-server.process_refund'))}
Payment retried: {json.dumps(execution_results.get('action-server.retry_payment'))}
Shipping upgraded: {json.dumps(execution_results.get('action-server.upgrade_shipping'))}
Replacement shipped: {json.dumps(execution_results.get('action-server.ship_replacement'))}
Credit applied: {json.dumps(execution_results.get('action-server.apply_credit'))}
VIP status enabled: {json.dumps(execution_results.get('action-server.enable_vip_status'))}

Email sent: {"Yes" if email_sent else "No"}

RESPONSE STYLE: Be confident, decisive, and action-oriented. Use phrases like:
- "I've immediately taken care of..."
- "I'm processing this right now..."
- "I've already upgraded/applied/processed..."
- "Consider it done..."
- "You'll receive..."

Focus on the ACTIONS you took to solve their problem, not just information. Make them feel like their issue is completely resolved. Include specific details about what you did and when they can expect results."""
            response = await self.llm_client.chat_completions_create(
                messages=[
                    {"role": "system", "content": "You are a powerful, action-oriented customer support representative who takes immediate action to solve problems. Focus on what you DID for the customer, not just what you found. Be confident and decisive."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.3
            )
            final_response = response["choices"][0]["message"]["content"]
            if not isinstance(final_response, str):
                final_response = str(final_response)
            return final_response
        except Exception as error:
            print(f"❌ Critical error in handle_request: {error}")
            return "I apologize, but I encountered an error while processing your request. Please try again later."

class ChatInterface:
    def __init__(self, llm_client):
        self.agent = UnifiedCustomerSupportAgent(llm_client)

    def _display_demo_data(self):
        print("\n📋 DEMO CUSTOMERS & ORDERS")
        print("=" * 60)
        demo_customers = [
            {"email": "john@email.com", "name": "John Smith", "order": "#1001", "product": "Wireless Headphones", "status": "✅ delivered", "scenario": "Happy customer"},
            {"email": "sarah@email.com", "name": "Sarah Johnson", "order": "#1002", "product": "Smart Watch", "status": "⚠️ shipping_delayed", "scenario": "Delayed shipment"},
            {"email": "mike@email.com", "name": "Mike Brown", "order": "#1003", "product": "Gaming Laptop", "status": "❌ payment_failed", "scenario": "Payment issue"},
            {"email": "lisa@email.com", "name": "Lisa Davis", "order": "#1004", "product": "Bluetooth Speaker", "status": "📦 lost_in_transit", "scenario": "Lost package"},
            {"email": "alex@email.com", "name": "Alex Wilson", "order": "#1005", "product": "Wireless Earbuds", "status": "🔄 cancelled/refunded", "scenario": "Refund inquiry"}
        ]
        for customer in demo_customers:
            print(f"📧 {customer['email']}")
            print(f"   👤 {customer['name']} | {customer['order']} - {customer['product']}")
            print(f"   📊 Status: {customer['status']} | 🎭 {customer['scenario']}")
            print()
        print("💡 SAMPLE QUERIES:")
        print("   • 'Check my order status'")
        print("   • 'Where is my [product] order?'") 
        print("   • 'What's happening with order [number]?'")
        print("   • 'I need help with my recent purchase'")
        print("=" * 60)

    async def start_chat(self):
        print("🤖 Enhanced MCP Customer Support Agent (Replicate)")
        print("=" * 50)
        print("✅ Proactive actions and guaranteed email notifications!")
        print("Type 'quit' to exit the chat")
        print("Type 'change email' to switch customer accounts")
        self._display_demo_data()
        customer_email = input("\n👤 Enter your email address: ").strip()
        if not customer_email:
            customer_email = "john@email.com"
            print(f"Using demo email: {customer_email}")
        print(f"\nHello! I'm your AI customer support agent for {customer_email}. How can I help you today?")
        print("(Reference the demo data above for testing different scenarios)")
        print("(Say 'change email' to switch to a different customer)")
        print()
        while True:
            try:
                print(f"💬 You ({customer_email}): ", end="", flush=True)
                user_input = input().strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thanks for using Enhanced MCP Customer Support! Have a great day!")
                    break
                if user_input.lower() in ['change email', 'switch email', 'new email']:
                    new_email = input("\n👤 Enter new email address: ").strip()
                    if new_email:
                        customer_email = new_email
                        print(f"✅ Switched to customer: {customer_email}")
                        print(f"Hello! I'm your AI customer support agent for {customer_email}. How can I help you today?\n")
                    continue
                if not user_input:
                    continue
                response = await self.agent.handle_request(customer_email, user_input)
                print(f"\n💬 Agent: {response}")
                print("\n" + "="*60)
            except KeyboardInterrupt:
                print("\n👋 Thanks for using Enhanced MCP Customer Support! Have a great day!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again.")

async def main():
    print("[DEBUG] Starting Replicate MCP script...")
    llm_client = ReplicateLLMClient()
    try:
        chat = ChatInterface(llm_client)
        await chat.start_chat()
    except Exception as e:
        print(f"❌ Application error: {e}")

if __name__ == "__main__":
    asyncio.run(main())