import asyncio
import json
import re
import time
import base64
import hashlib
import random
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple
from urllib.parse import urljoin

try:
    from curl_cffi import requests as curl_requests
    HAS_CURL_CFI = True
except ImportError:
    HAS_CURL_CFI = False
    import requests

try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
except ImportError:
    HAS_CLOUDSCRAPER = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


class TurnstileSolver:
    """دریافت Turnstile Token از صفحه DeepInfra با استفاده از CloudScraper"""

    @staticmethod
    def get_token_sync(model: str = "zai-org/GLM-5.2", timeout: int = 30) -> str:
        """دریافت هم‌زمان Turnstile Token با CloudScraper"""
        if not HAS_CLOUDSCRAPER or not HAS_BS4:
            raise RuntimeError("cloudscraper and beautifulsoup4 are required for Turnstile solving")

        url = f"https://deepinfra.com/{model}"
        scraper = cloudscraper.create_scraper(
            browser={
                "browser": "chrome",
                "platform": "windows",
                "desktop": True,
                "mobile": False,
            },
            delay=10,
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        try:
            response = scraper.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            # جستجوی token در input با name="cf-turnstile-response"
            token_input = soup.find('input', {'name': 'cf-turnstile-response'})
            if token_input and token_input.get('value'):
                return token_input['value']
            # جستجوی token در اسکریپت‌ها
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string:
                    match = re.search(r'cf-turnstile-response["\']?\s*[:=]\s*["\']([^"\']+)["\']', script.string)
                    if match:
                        return match.group(1)
            # اگر token پیدا نشد، یک token ساختگی برمی‌گردانیم (شاید کار نکند)
            return ""
        except Exception as e:
            raise RuntimeError(f"Failed to get Turnstile token: {e}")

    @staticmethod
    async def get_token_async(model: str = "zai-org/GLM-5.2") -> str:
        """دریافت ناهم‌زمان Turnstile Token با استفاده از ThreadPoolExecutor"""
        import concurrent.futures
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, TurnstileSolver.get_token_sync, model)


class DeepInfraClient:
    """کلاینت DeepInfra با پشتیبانی از Function Calling بدون API Key"""

    BASE_URL = "https://api.deepinfra.com/v1/openai"
    MODELS = ["zai-org/GLM-5.2", "meta-llama/Llama-3.3-70B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]

    def __init__(self, model: str = "zai-org/GLM-5.2", turnstile_token: Optional[str] = None):
        self.model = model
        self._turnstile_token = turnstile_token
        self._tools = []
        self._session = None

    def add_tool(self, tool: Dict[str, Any]) -> None:
        self._tools.append(tool)

    def add_tools(self, tools: List[Dict[str, Any]]) -> None:
        self._tools.extend(tools)

    async def _get_turnstile_token(self) -> str:
        """دریافت Turnstile Token در صورت نیاز"""
        if self._turnstile_token:
            return self._turnstile_token
        try:
            token = await TurnstileSolver.get_token_async(self.model)
            self._turnstile_token = token
            return token
        except Exception as e:
            print(f"⚠️ Failed to get Turnstile token: {e}")
            return ""

    def _get_headers(self, stream: bool = True) -> Dict[str, str]:
        """ساخت هدرهای درخواست"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Origin": "https://deepinfra.com",
            "Referer": "https://deepinfra.com/",
            "X-DeepInfra-Source": "web-page",
        }
        if self._turnstile_token:
            headers["X-DeepInfra-Turnstile"] = self._turnstile_token
        return headers

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        tool_choice: Optional[str] = "auto",
        extra_body: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if self._tools:
            payload["tools"] = self._tools
            if tool_choice:
                if tool_choice in ("auto", "none"):
                    payload["tool_choice"] = tool_choice
                else:
                    payload["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
        if extra_body:
            payload.update(extra_body)
        return payload

    async def _send_request(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Tuple[Any, bool]:
        """ارسال درخواست با curl_cffi (ترجیح) یا requests"""
        url = f"{self.BASE_URL}/chat/completions"
        is_stream = payload.get("stream", False)

        if HAS_CURL_CFI:
            # استفاده از curl_cffi با ایمیت کردن Chrome
            session = curl_requests.AsyncSession(impersonate="chrome")
            response = await session.post(url, json=payload, headers=headers)
            return response, True
        else:
            # Fallback به requests معمولی
            session = requests.Session()
            response = session.post(url, json=payload, headers=headers)
            return response, False

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        tool_choice: Optional[str] = "auto",
        extra_body: Optional[Dict] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """ارسال درخواست چت با پشتیبانی از ابزارها"""
        if model:
            self.model = model

        # دریافت Turnstile Token اگر نداریم
        if not self._turnstile_token:
            self._turnstile_token = await self._get_turnstile_token()
            if not self._turnstile_token:
                print("⚠️ No Turnstile token available. Request may fail.")

        headers = self._get_headers(stream)
        payload = self._build_payload(
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            extra_body=extra_body,
        )

        # ارسال درخواست
        response, is_curl = await self._send_request(payload, headers)

        if response.status_code != 200:
            error_text = response.text if hasattr(response, 'text') else await response.text()
            if "turnstile" in error_text.lower() or "captcha" in error_text.lower():
                # تلاش مجدد با Turnstile جدید
                self._turnstile_token = await self._get_turnstile_token()
                headers = self._get_headers(stream)
                response, is_curl = await self._send_request(payload, headers)
                if response.status_code != 200:
                    raise RuntimeError(f"Request failed after retry: {response.status_code} - {error_text}")
            else:
                raise RuntimeError(f"Request failed: {response.status_code} - {error_text}")

        if not stream:
            data = response.json() if hasattr(response, 'json') else json.loads(response.text)
            yield data
            return

        # پردازش پاسخ جریانی (SSE)
        buffer = ""
        async for chunk in self._iter_response(response, is_curl):
            buffer += chunk
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                if not line:
                    continue
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        continue
                    try:
                        data = json.loads(data_str)
                        yield data
                    except json.JSONDecodeError:
                        continue

    async def _iter_response(self, response, is_curl: bool):
        """پیمایش پاسخ جریانی"""
        if is_curl:
            # curl_cffi پاسخ را به صورت AsyncIterator ارائه می‌دهد
            async for chunk in response.iter_content():
                if chunk:
                    yield chunk.decode('utf-8', errors='ignore')
        else:
            # requests معمولی
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk.decode('utf-8', errors='ignore')

    @staticmethod
    def extract_tool_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """استخراج Tool Calls از پاسخ"""
        choices = response.get('choices', [])
        if not choices:
            return []
        message = choices[0].get('message', {})
        return message.get('tool_calls', [])

    @staticmethod
    def create_function_tool(name: str, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }


class WeatherTools:
    """ابزارهای هواشناسی نمونه"""

    @staticmethod
    def get_tools() -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The unit of temperature"
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather_forecast",
                    "description": "Get weather forecast for a given location and time period",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name"
                            },
                            "days": {
                                "type": "integer",
                                "description": "Number of days for forecast (1-7)",
                                "minimum": 1,
                                "maximum": 7
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

    @staticmethod
    def execute_tool(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        func_name = tool_call['function']['name']
        args = json.loads(tool_call['function']['arguments'])
        if func_name == "get_current_weather":
            return WeatherTools._get_current_weather(args)
        elif func_name == "get_weather_forecast":
            return WeatherTools._get_weather_forecast(args)
        else:
            return {"error": f"Unknown tool: {func_name}"}

    @staticmethod
    def _get_current_weather(args: Dict[str, Any]) -> Dict[str, Any]:
        location = args.get('location', 'Unknown')
        unit = args.get('unit', 'celsius')
        temp = 22 if unit == "celsius" else 72
        return {
            "location": location,
            "temperature": temp,
            "unit": unit,
            "condition": "Partly cloudy",
            "humidity": 45,
            "wind_speed": "12 km/h"
        }

    @staticmethod
    def _get_weather_forecast(args: Dict[str, Any]) -> Dict[str, Any]:
        location = args.get('location', 'Unknown')
        days = min(args.get('days', 3), 7)
        forecast = []
        conditions = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy"]
        for i in range(days):
            forecast.append({
                "day": i + 1,
                "condition": conditions[i % len(conditions)],
                "high": 20 + (i * 3),
                "low": 15 + (i * 2)
            })
        return {
            "location": location,
            "forecast": forecast,
            "days": days
        }


async def main():
    print("=" * 60)
    print("🌤️  DeepInfra + Function Calling Example")
    print("=" * 60)

    # بررسی پیش‌نیازها
    if not HAS_CURL_CFI:
        print("⚠️ curl_cffi not installed. Using requests (may fail).")
    if not HAS_CLOUDSCRAPER:
        print("⚠️ cloudscraper not installed. Turnstile solving may fail.")

    # ایجاد کلاینت
    client = DeepInfraClient(model="zai-org/GLM-5.2")
    print("\n✅ Client created (no API key required)")

    # افزودن ابزارها
    tools = WeatherTools.get_tools()
    client.add_tools(tools)
    print(f"✅ {len(tools)} tools added")
    for tool in tools:
        print(f"   - {tool['function']['name']}: {tool['function']['description']}")

    # پیام‌ها
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can provide weather information."},
        {"role": "user", "content": "What's the weather like in Tehran today? Also, could you give me a 3-day forecast for Isfahan?"}
    ]

    print("\n" + "=" * 60)
    print("📨 Sending request...")
    print(f"   Messages: {messages[-1]['content']}")
    print("=" * 60)

    full_response = ""
    tool_calls_found = []
    tool_results = []

    try:
        async for chunk in client.chat_completion(
            messages=messages,
            stream=True,
            temperature=0.7,
            tool_choice="auto"
        ):
            if 'choices' in chunk:
                choice = chunk['choices'][0]
                delta = choice.get('delta', {})
                if 'content' in delta and delta['content']:
                    full_response += delta['content']
                    print(delta['content'], end='', flush=True)
                if 'tool_calls' in delta:
                    for tc in delta['tool_calls']:
                        tool_calls_found.append(tc)
            if 'choices' in chunk and chunk['choices'][0].get('finish_reason') == 'tool_calls':
                print("\n\n🔧 Tool Calls detected!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return

    print("\n" + "=" * 60)

    # اجرای ابزارها
    if tool_calls_found:
        print("\n🔧 Executing tools...")
        for tool_call in tool_calls_found:
            print(f"\n   📞 Calling: {tool_call['function']['name']}")
            print(f"   📋 Arguments: {tool_call['function']['arguments']}")
            result = WeatherTools.execute_tool(tool_call)
            print(f"   ✅ Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
            tool_results.append({
                "tool_call_id": tool_call['id'],
                "result": result
            })

        # آماده‌سازی پیام‌ها برای دور دوم
        for tc in tool_calls_found:
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [tc]
            })
        for tr in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": tr['tool_call_id'],
                "content": json.dumps(tr['result'], ensure_ascii=False)
            })

        # دریافت پاسخ نهایی
        print("\n" + "=" * 60)
        print("📝 Final answer from model...")
        print("=" * 60)

        try:
            async for chunk in client.chat_completion(
                messages=messages,
                stream=True,
                temperature=0.7
            ):
                if 'choices' in chunk:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta and delta['content']:
                        print(delta['content'], end='', flush=True)
        except Exception as e:
            print(f"\n❌ Error in final response: {e}")

    print("\n" + "=" * 60)
    print("✅ Done")


if __name__ == "__main__":
    asyncio.run(main())