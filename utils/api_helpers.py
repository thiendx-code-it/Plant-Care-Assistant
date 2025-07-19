import requests
import asyncio
import aiohttp
from typing import Dict, Any, Optional
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