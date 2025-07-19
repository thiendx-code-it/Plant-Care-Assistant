import requests
import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional
from openai import AsyncAzureOpenAI
from config.settings import settings

class APIHelper:
    """Helper class for making API calls to external services"""
    
    @staticmethod
    async def call_plant_id_api(image_base64: str, include_health: bool = True) -> Dict[str, Any]:
        """Call Plant.id identification API with optional health assessment (v3 format)"""
        try:
            url = settings.PLANT_ID_API_URL
            headers = {
                "Content-Type": "application/json",
                "Api-Key": settings.PLANT_ID_API_KEY
            }
            
            # API v3 format payload with health assessment
            payload = {
                "images": [image_base64],
                "similar_images": True,
                "classification_level": "all"
            }
            
            # Add health assessment if requested
            if include_health:
                payload["health"] = "all"  # Include both plant.id and plant.health results
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status in [200, 201]:  # Accept both 200 OK and 201 Created
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"Plant.id API error: {response.status} - {error_text}")
                        
        except Exception as e:
            raise
    
    @staticmethod
    async def call_plant_health_api(image_base64: str) -> Dict[str, Any]:
        """Call Plant.id health assessment API (v3 format)"""
        try:
            url = settings.PLANT_ID_HEALTH_URL
            headers = {
                "Content-Type": "application/json",
                "Api-Key": settings.PLANT_ID_API_KEY
            }
            
            # API v3 format payload - simplified without modifiers
            payload = {
                "images": [image_base64],
                "similar_images": True,
                "health": "all"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status in [200, 201]:  # Accept both 200 OK and 201 Created
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"Plant.id Health API error: {response.status} - {error_text}")
                        
        except Exception as e:
            raise
    
    @staticmethod
    async def call_openweather_api(city: str) -> Dict[str, Any]:
        """Call OpenWeather API for current weather"""
        params = {
            "q": city,
            "appid": settings.OPENWEATHER_API_KEY,
            "units": "metric"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(settings.OPENWEATHER_API_URL, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"OpenWeather API error: {response.status} - {await response.text()}")
    
    @staticmethod
    async def call_tavily_search_api(query: str, max_results: int = 5) -> Dict[str, Any]:
        """Call Tavily Search API for web search"""
        payload = {
            "api_key": settings.TAVILY_API_KEY,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": True,
            "include_raw_content": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(settings.TAVILY_API_URL, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Tavily API error: {response.status} - {await response.text()}")
    
    @staticmethod
    async def call_azure_openai_chat(messages: list, temperature: float = 0.3, max_tokens: int = 500) -> Dict[str, Any]:
        """Call Azure OpenAI Chat API for LLM operations like keyword extraction"""
        try:
            # Initialize Azure OpenAI client
            client = AsyncAzureOpenAI(
                api_key=settings.AZURE_OPENAI_API_KEY_CHAT,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT_CHAT
            )
            
            # Make the chat completion call
            response = await client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT_CHAT,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            # Convert response to dictionary format for compatibility
            return {
                "choices": [
                    {
                        "message": {
                            "content": response.choices[0].message.content,
                            "role": response.choices[0].message.role
                        },
                        "finish_reason": response.choices[0].finish_reason
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }
                        
        except Exception as e:
            raise Exception(f"Azure OpenAI API error: {str(e)}")
    
    @staticmethod
    async def extract_search_keywords(user_query: str, plant_name: str = "Unknown") -> Dict[str, Any]:
        """Extract optimized search keywords from user query using LLM"""
        try:
            # Validate configuration first
            if not settings.AZURE_OPENAI_API_KEY_CHAT or not settings.AZURE_OPENAI_ENDPOINT_CHAT:
                raise Exception("Azure OpenAI configuration missing: API key or endpoint not set")
            
            system_prompt = """
You are a plant care expert assistant. Your task is to analyze user queries about plants and extract the most relevant search keywords that would help find accurate information in a plant care knowledge base.

Given a user query and optionally a plant name, extract:
1. Primary search terms (most important keywords)
2. Secondary search terms (supporting keywords)
3. Plant care categories (watering, light, soil, fertilizer, pruning, diseases, pests, etc.)
4. Specific issues or symptoms mentioned

IMPORTANT: If the provided plant name is "Unknown", try to identify any plant names mentioned in the user query itself. Look for:
- Common plant names (rose, tomato, orchid, etc.)
- Scientific names (Rosa, Solanum lycopersicum, etc.)
- Plant types (succulent, herb, tree, etc.)
- Specific varieties or cultivars

Return your response as a JSON object with the following structure:
{
  "primary_keywords": ["keyword1", "keyword2"],
  "secondary_keywords": ["keyword3", "keyword4"],
  "care_categories": ["category1", "category2"],
  "issues_symptoms": ["issue1", "symptom1"],
  "optimized_query": "A refined search query combining the most relevant terms",
  "detected_plant_name": "plant name if found in query, otherwise null"
}

Focus on plant care terminology and be specific about plant problems, care requirements, and botanical terms.
"""
            
            user_prompt = f"""
User Query: "{user_query}"
Plant Name: "{plant_name}"

Please extract the most relevant search keywords and create an optimized search query for finding plant care information.
"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            print(f"DEBUG - Calling Azure OpenAI for keyword extraction...")
            response = await APIHelper.call_azure_openai_chat(messages, temperature=0.1, max_tokens=300)
            print(f"DEBUG - Azure OpenAI response received: {response.get('choices', [{}])[0].get('message', {}).get('content', '')[:100]}...")
            
            # Extract the content from the response
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not content:
                raise Exception("Empty response from Azure OpenAI")
            
            # Clean the content - remove markdown code blocks if present
            cleaned_content = content.strip()
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content[7:]  # Remove ```json
            if cleaned_content.startswith('```'):
                cleaned_content = cleaned_content[3:]   # Remove ```
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]  # Remove trailing ```
            cleaned_content = cleaned_content.strip()
            
            print(f"DEBUG - Cleaned content for JSON parsing: {cleaned_content[:200]}...")
            
            # Try to parse as JSON
            try:
                keywords_data = json.loads(cleaned_content)
                print(f"DEBUG - Successfully parsed JSON keywords: {keywords_data}")
                return {
                    "success": True,
                    "data": keywords_data
                }
            except json.JSONDecodeError as json_error:
                print(f"DEBUG - JSON parsing failed: {json_error}. Original content: {content}")
                print(f"DEBUG - Cleaned content: {cleaned_content}")
                # Fallback: create a simple keyword extraction
                return {
                    "success": True,
                    "data": {
                        "primary_keywords": [plant_name, user_query],
                        "secondary_keywords": [],
                        "care_categories": [],
                        "issues_symptoms": [],
                        "optimized_query": f"{plant_name} {user_query}"
                    }
                }
                
        except Exception as e:
            print(f"DEBUG - Keyword extraction error: {type(e).__name__}: {str(e)}")
            # Fallback in case of API failure
            return {
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "data": {
                    "primary_keywords": [plant_name, user_query],
                    "secondary_keywords": [],
                    "care_categories": [],
                    "issues_symptoms": [],
                    "optimized_query": f"{plant_name} {user_query}"
                }
            }

class WeatherHelper:
    """Helper class for weather-related operations"""
    
    @staticmethod
    def should_skip_watering(weather_data: Dict[str, Any]) -> bool:
        """Determine if watering should be skipped based on weather"""
        try:
            # Check for rain in current conditions
            weather_main = weather_data.get('weather', [{}])[0].get('main', '').lower()
            if 'rain' in weather_main:
                return True
            
            # Check humidity levels
            humidity = weather_data.get('main', {}).get('humidity', 0)
            if humidity > 80:
                return True
            
            return False
        except Exception:
            return False
    
    @staticmethod
    def get_weather_adjustment_factor(weather_data: Dict[str, Any]) -> float:
        """Get watering frequency adjustment based on weather"""
        try:
            temp = weather_data.get('main', {}).get('temp', 20)
            humidity = weather_data.get('main', {}).get('humidity', 50)
            
            # Base adjustment factor
            factor = 1.0
            
            # Temperature adjustments
            if temp > 30:  # Hot weather
                factor *= 1.3
            elif temp < 10:  # Cold weather
                factor *= 0.7
            
            # Humidity adjustments
            if humidity > 70:  # High humidity
                factor *= 0.8
            elif humidity < 30:  # Low humidity
                factor *= 1.2
            
            return max(0.5, min(2.0, factor))  # Clamp between 0.5 and 2.0
        except Exception:
            return 1.0