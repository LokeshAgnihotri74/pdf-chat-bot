"""
Tool-Calling AI Agent Module.
Features:
1. Real-time Weather Tool via Open-Meteo (100% Free, Zero API Key Required).
2. Stock Price Calculator & Lookup Tool via yfinance with resilient fallback.
3. Exponential Backoff and Retries for API Rate Limits.
4. Input validation and graceful error recovery (no agent loop crashes).
5. LangChain Tool Calling with Groq LLaMA models.
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional
import requests
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq

from config import (
    GROQ_API_KEY,
    DEFAULT_MODEL,
    OPEN_METEO_GEO_URL,
    OPEN_METEO_FORECAST_URL,
    RETRY_MAX_ATTEMPTS,
    RETRY_MIN_WAIT,
    RETRY_MAX_WAIT
)

logger = logging.getLogger(__name__)

# WMO Weather Interpretation Codes (WW)
WMO_CODE_MAP = {
    0: "Clear sky ☀️",
    1: "Mainly clear 🌤️",
    2: "Partly cloudy ⛅",
    3: "Overcast ☁️",
    45: "Fog 🌫️",
    48: "Depositing rime fog 🌫️",
    51: "Light drizzle 🌦️",
    53: "Moderate drizzle 🌧️",
    55: "Dense drizzle 🌧️",
    56: "Light freezing drizzle 🌨️",
    57: "Dense freezing drizzle 🌨️",
    61: "Slight rain 🌧️",
    63: "Moderate rain 🌧️",
    65: "Heavy rain 🌧️",
    66: "Light freezing rain 🌨️",
    67: "Heavy freezing rain 🌨️",
    71: "Slight snow fall ❄️",
    73: "Moderate snow fall ❄️",
    75: "Heavy snow fall ❄️",
    77: "Snow grains ❄️",
    80: "Slight rain showers 🌦️",
    81: "Moderate rain showers 🌧️",
    82: "Violent rain showers ⛈️",
    85: "Slight snow showers 🌨️",
    86: "Heavy snow showers 🌨️",
    95: "Thunderstorm ⛈️",
    96: "Thunderstorm with slight hail ⛈️",
    99: "Thunderstorm with heavy hail ⛈️"
}

# Resilience: Retry decorator for rate limits and transient network timeouts
api_retry_decorator = retry(
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    retry=retry_if_exception_type((requests.exceptions.RequestException, TimeoutError)),
    reraise=True
)


class WeatherInput(BaseModel):
    city: str = Field(
        ...,
        description="The name of the city, municipality, or region to look up weather for (e.g. 'Bhopal', 'Tokyo', 'San Francisco')."
    )


class StockInput(BaseModel):
    symbol: str = Field(
        ...,
        description="The official stock ticker symbol to look up (e.g. 'AAPL' for Apple, 'MSFT' for Microsoft, 'GOOGL' for Google, 'TSLA' for Tesla)."
    )
    shares: float = Field(
        default=1.0,
        description="Number of shares for financial calculations. Default is 1.0."
    )
    operation: str = Field(
        default="quote",
        description="Financial calculation type: 'quote' for price, 'cost' for total purchase cost, 'valuation' for PE & market cap."
    )


@api_retry_decorator
def _fetch_open_meteo_data(city: str) -> Dict[str, Any]:
    """Helper to query Open-Meteo with retry logic on network/rate-limit errors."""
    geo_resp = requests.get(
        OPEN_METEO_GEO_URL,
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=10
    )
    if geo_resp.status_code == 429:
        raise requests.exceptions.RequestException("Open-Meteo Rate Limit Exceeded (HTTP 429). Retrying...")
    geo_resp.raise_for_status()
    geo_data = geo_resp.json()

    if not geo_data.get("results"):
        return {"error": f"City '{city}' not found in Open-Meteo geocoding database."}

    location = geo_data["results"][0]
    lat = location["latitude"]
    lon = location["longitude"]
    city_name = location.get("name", city)
    country = location.get("country", "")
    admin1 = location.get("admin1", "")

    forecast_resp = requests.get(
        OPEN_METEO_FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m",
            "timezone": "auto"
        },
        timeout=10
    )
    if forecast_resp.status_code == 429:
        raise requests.exceptions.RequestException("Open-Meteo Rate Limit Exceeded (HTTP 429). Retrying...")
    forecast_resp.raise_for_status()
    forecast_data = forecast_resp.json()

    current = forecast_data.get("current", {})
    weather_code = current.get("weather_code", 0)
    weather_desc = WMO_CODE_MAP.get(weather_code, "Unknown weather condition")

    temp_c = current.get("temperature_2m", 0.0)
    temp_f = round((temp_c * 9 / 5) + 32, 1)
    feels_c = current.get("apparent_temperature", 0.0)
    feels_f = round((feels_c * 9 / 5) + 32, 1)

    return {
        "city": city_name,
        "region": admin1,
        "country": country,
        "latitude": lat,
        "longitude": lon,
        "temperature_celsius": temp_c,
        "temperature_fahrenheit": temp_f,
        "feels_like_celsius": feels_c,
        "feels_like_fahrenheit": feels_f,
        "humidity_percent": current.get("relative_humidity_2m", 0),
        "wind_speed_kmh": current.get("wind_speed_10m", 0.0),
        "precipitation_mm": current.get("precipitation", 0.0),
        "weather_condition": weather_desc,
        "is_day": bool(current.get("is_day", 1)),
        "source": "Open-Meteo Public API (Free, Real-Time)"
    }


@tool("get_realtime_weather", args_schema=WeatherInput)
def get_realtime_weather(city: str) -> str:
    """
    Get live real-time weather information for any city in the world using Open-Meteo.
    Returns temperature (°C and °F), feels-like temp, humidity, wind speed, and conditions.
    """
    if not city or not city.strip():
        return json.dumps({
            "status": "error",
            "message": "City name cannot be empty. Please provide a valid city name (e.g., 'Bhopal', 'London')."
        })

    try:
        data = _fetch_open_meteo_data(city.strip())
        if "error" in data:
            return json.dumps({"status": "not_found", "message": data["error"]})

        return json.dumps({"status": "success", "data": data}, indent=2)

    except requests.exceptions.RequestException as e:
        return json.dumps({
            "status": "error",
            "message": f"Network or rate limit error while querying Open-Meteo: {str(e)}. Please try again."
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Unexpected error in weather tool: {str(e)}"
        })


# Sample / Fallback Stock Data for offline resilience
DEMO_STOCK_DATA = {
    "AAPL": {"name": "Apple Inc.", "price": 235.40, "currency": "USD", "change_pct": 1.25, "pe_ratio": 34.2, "market_cap": "3.62T"},
    "MSFT": {"name": "Microsoft Corporation", "price": 428.15, "currency": "USD", "change_pct": -0.45, "pe_ratio": 36.1, "market_cap": "3.18T"},
    "GOOGL": {"name": "Alphabet Inc.", "price": 182.60, "currency": "USD", "change_pct": 0.88, "pe_ratio": 24.5, "market_cap": "2.25T"},
    "TSLA": {"name": "Tesla, Inc.", "price": 218.90, "currency": "USD", "change_pct": 3.12, "pe_ratio": 62.8, "market_cap": "698B"},
    "NVDA": {"name": "NVIDIA Corporation", "price": 138.25, "currency": "USD", "change_pct": 2.40, "pe_ratio": 54.0, "market_cap": "3.41T"}
}


@tool("stock_lookup_and_calculator", args_schema=StockInput)
def stock_lookup_and_calculator(symbol: str, shares: float = 1.0, operation: str = "quote") -> str:
    """
    Look up stock quotes, calculate total investment costs for a given number of shares,
    and view valuation metrics for any public ticker symbol (e.g. AAPL, MSFT, GOOGL, NVDA, TSLA).
    """
    if not symbol or not symbol.strip():
        return json.dumps({
            "status": "error",
            "message": "Stock symbol cannot be empty. Please provide a ticker like 'AAPL' or 'MSFT'."
        })

    clean_symbol = symbol.strip().upper()
    if shares <= 0:
        return json.dumps({
            "status": "error",
            "message": f"Invalid number of shares ({shares}). Must be greater than 0."
        })

    # Attempt fetching from yfinance first
    is_live = False
    stock_info = None

    try:
        import yfinance as yf
        ticker = yf.Ticker(clean_symbol)
        fast_info = getattr(ticker, "fast_info", None)

        if fast_info and fast_info.last_price is not None:
            last_price = float(fast_info.last_price)
            prev_close = float(fast_info.previous_close) if fast_info.previous_close else last_price
            change_pct = round(((last_price - prev_close) / prev_close) * 100, 2) if prev_close else 0.0

            stock_info = {
                "symbol": clean_symbol,
                "current_price": round(last_price, 2),
                "currency": getattr(fast_info, "currency", "USD"),
                "day_high": round(float(fast_info.day_high), 2) if fast_info.day_high else None,
                "day_low": round(float(fast_info.day_low), 2) if fast_info.day_low else None,
                "change_percent": change_pct,
                "market_cap": getattr(fast_info, "market_cap", None),
                "data_source": "Yahoo Finance (Live Market Feed)"
            }
            is_live = True
    except Exception as e:
        logger.warning(f"yfinance query failed for {clean_symbol}: {e}")

    # Fallback to demo data if live market query fails or rate limited
    if not is_live:
        if clean_symbol in DEMO_STOCK_DATA:
            demo = DEMO_STOCK_DATA[clean_symbol]
            stock_info = {
                "symbol": clean_symbol,
                "company_name": demo["name"],
                "current_price": demo["price"],
                "currency": demo["currency"],
                "change_percent": demo["change_pct"],
                "pe_ratio": demo["pe_ratio"],
                "market_cap": demo["market_cap"],
                "data_source": "Demonstration Market Dataset (Live Yahoo Finance offline / rate limited)"
            }
        else:
            return json.dumps({
                "status": "not_found",
                "message": (
                    f"Ticker '{clean_symbol}' not found on Yahoo Finance or demo data. "
                    "Please verify the ticker symbol (e.g. AAPL, MSFT, GOOGL, TSLA, NVDA)."
                )
            })

    # Financial computations
    price = stock_info["current_price"]
    currency = stock_info.get("currency", "USD")
    total_cost = round(price * shares, 2)

    result_payload = {
        "status": "success",
        "ticker": clean_symbol,
        "details": stock_info,
        "calculation": {
            "requested_shares": shares,
            "unit_price": f"{price} {currency}",
            "total_value": f"{total_cost} {currency}",
            "calculation_formula": f"{shares} shares × {price} {currency} = {total_cost} {currency}"
        }
    }

    return json.dumps(result_payload, indent=2)


AVAILABLE_TOOLS = [get_realtime_weather, stock_lookup_and_calculator]
TOOL_MAP = {t.name: t for t in AVAILABLE_TOOLS}

AGENT_SYSTEM_PROMPT = """You are a highly capable AI Assistant with direct access to specialized Python tools.

When a user's request requires live or specific external data:
1. Intelligently select and call the appropriate tool:
   - For weather queries: call `get_realtime_weather`.
   - For stock prices, portfolio value, share calculations, or market data: call `stock_lookup_and_calculator`.
2. Do NOT invent or make up weather forecasts or current stock prices. Always rely on the tool results.
3. For general knowledge, coding, creative, or conversational questions, answer directly without invoking tools.
4. When presenting tool results to the user:
   - Clearly explain what tool was invoked.
   - Present the key findings in an engaging, well-formatted markdown response.
   - State whether data is live or demo data if applicable.
"""


def run_agent_turn(
    user_message: str,
    conversation_history: List[Dict[str, str]],
    api_key: str,
    model_name: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Executes a single agent turn with tool calling, supporting multiple iterations
    and capturing all tool execution steps for transparent UI rendering.
    """
    if not user_message or not user_message.strip():
        return {
            "final_answer": "Please enter a message or question.",
            "tool_executions": []
        }

    if not api_key:
        return {
            "final_answer": "Error: Groq API Key is not set. Please provide it in .env or the sidebar.",
            "tool_executions": []
        }

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.1
    )
    llm_with_tools = llm.bind_tools(AVAILABLE_TOOLS)

    # Reconstruct message history
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
    for msg in conversation_history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_message))

    tool_executions = []
    max_turns = 5

    try:
        for _ in range(max_turns):
            ai_msg = llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            # Check if LLM requested any tool calls
            tool_calls = getattr(ai_msg, "tool_calls", [])
            if not tool_calls:
                return {
                    "final_answer": ai_msg.content,
                    "tool_executions": tool_executions
                }

            # Execute each requested tool call
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                call_id = tool_call.get("id", "call_default")

                start_time = time.time()
                tool_func = TOOL_MAP.get(tool_name)

                if tool_func:
                    try:
                        raw_result = tool_func.invoke(tool_args)
                    except Exception as err:
                        raw_result = json.dumps({
                            "status": "error",
                            "message": f"Execution error in tool '{tool_name}': {str(err)}"
                        })
                else:
                    raw_result = json.dumps({
                        "status": "error",
                        "message": f"Tool '{tool_name}' is not recognized."
                    })

                exec_time_ms = round((time.time() - start_time) * 1000, 1)

                tool_executions.append({
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "result": raw_result,
                    "duration_ms": exec_time_ms
                })

                messages.append(ToolMessage(
                    content=str(raw_result),
                    tool_call_id=call_id
                ))

        # If loop exited due to max turns, make a final call
        final_ai = llm.invoke(messages)
        return {
            "final_answer": final_ai.content,
            "tool_executions": tool_executions
        }

    except Exception as e:
        return {
            "final_answer": f"Error during agent execution: {str(e)}",
            "tool_executions": tool_executions
        }
