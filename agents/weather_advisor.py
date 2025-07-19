import asyncio
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent
from utils.api_helpers import APIHelper, WeatherHelper
from datetime import datetime, timedelta

class WeatherAdvisorAgent(BaseAgent):
    """Agent responsible for fetching weather data and adjusting care plans"""
    
    def __init__(self):
        super().__init__(
            name="Weather Advisor Agent",
            description="Fetches current weather using OpenWeather API to modify care plans."
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch weather data and provide weather-adjusted care recommendations"""
        try:
            # Validate input
            if not self.validate_input(input_data, ["location"]):
                return self.create_error_response("Missing required field: location")
            
            location = input_data["location"]
            plant_name = input_data.get("plant_name", "Unknown plant")
            care_schedule = input_data.get("care_schedule", {})
            
            # Fetch current weather
            self.logger.info(f"Fetching weather data for {location}")
            weather_data = await APIHelper.call_openweather_api(location)
            
            # Analyze weather impact on plant care
            weather_analysis = self._analyze_weather_impact(weather_data, plant_name)
            
            # Adjust care schedule based on weather
            adjusted_schedule = self._adjust_care_schedule(care_schedule, weather_data)
            
            # Generate weather-specific recommendations
            weather_recommendations = self._generate_weather_recommendations(weather_data, plant_name)
            
            result = {
                "location": location,
                "plant_name": plant_name,
                "current_weather": self._format_weather_data(weather_data),
                "weather_analysis": weather_analysis,
                "adjusted_schedule": adjusted_schedule,
                "weather_recommendations": weather_recommendations,
                "last_updated": datetime.now().isoformat()
            }
            
            return self.create_success_response(result)
            
        except Exception as e:
            self.logger.error(f"Error in weather analysis: {e}")
            return self.create_error_response(str(e), "WEATHER_ANALYSIS_ERROR")
    
    def _format_weather_data(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format weather data for display"""
        try:
            main = weather_data.get("main", {})
            weather = weather_data.get("weather", [{}])[0]
            wind = weather_data.get("wind", {})
            
            return {
                "temperature": main.get("temp", 0),
                "feels_like": main.get("feels_like", 0),
                "humidity": main.get("humidity", 0),
                "pressure": main.get("pressure", 0),
                "description": weather.get("description", "Unknown"),
                "main_condition": weather.get("main", "Unknown"),
                "wind_speed": wind.get("speed", 0),
                "wind_direction": wind.get("deg", 0),
                "visibility": weather_data.get("visibility", 0) / 1000,  # Convert to km
                "uv_index": weather_data.get("uvi", "N/A")
            }
        except Exception as e:
            self.logger.warning(f"Error formatting weather data: {e}")
            return {"error": "Could not format weather data"}
    
    def _analyze_weather_impact(self, weather_data: Dict[str, Any], plant_name: str) -> Dict[str, Any]:
        """Analyze how current weather affects plant care"""
        try:
            main = weather_data.get("main", {})
            weather = weather_data.get("weather", [{}])[0]
            
            temp = main.get("temp", 20)
            humidity = main.get("humidity", 50)
            condition = weather.get("main", "").lower()
            
            analysis = {
                "temperature_impact": self._analyze_temperature_impact(temp),
                "humidity_impact": self._analyze_humidity_impact(humidity),
                "precipitation_impact": self._analyze_precipitation_impact(condition),
                "overall_stress_level": self._calculate_plant_stress_level(temp, humidity, condition),
                "care_adjustments_needed": []
            }
            
            # Determine specific care adjustments
            adjustments = []
            
            if temp > 30:
                adjustments.append("Increase watering frequency due to high temperature")
                adjustments.append("Provide shade during peak sun hours")
                adjustments.append("Increase humidity around plant")
            elif temp < 10:
                adjustments.append("Reduce watering frequency due to low temperature")
                adjustments.append("Protect from cold drafts")
                adjustments.append("Consider moving indoors if possible")
            
            if humidity > 80:
                adjustments.append("Improve air circulation to prevent fungal issues")
                adjustments.append("Reduce watering frequency")
            elif humidity < 30:
                adjustments.append("Increase humidity around plant")
                adjustments.append("Consider using humidity tray")
            
            if "rain" in condition:
                adjustments.append("Skip scheduled watering")
                adjustments.append("Ensure proper drainage")
            
            analysis["care_adjustments_needed"] = adjustments
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing weather impact: {e}")
            return {"error": "Could not analyze weather impact"}
    
    def _analyze_temperature_impact(self, temp: float) -> Dict[str, Any]:
        """Analyze temperature impact on plant"""
        if temp > 35:
            return {
                "level": "extreme_heat",
                "description": "Extreme heat stress - immediate action needed",
                "severity": "critical"
            }
        elif temp > 30:
            return {
                "level": "high_heat",
                "description": "High temperature - increased care needed",
                "severity": "high"
            }
        elif temp < 5:
            return {
                "level": "extreme_cold",
                "description": "Extreme cold - protect plant immediately",
                "severity": "critical"
            }
        elif temp < 10:
            return {
                "level": "cold_stress",
                "description": "Cold conditions - reduce watering",
                "severity": "moderate"
            }
        else:
            return {
                "level": "optimal",
                "description": "Temperature within acceptable range",
                "severity": "low"
            }
    
    def _analyze_humidity_impact(self, humidity: float) -> Dict[str, Any]:
        """Analyze humidity impact on plant"""
        if humidity > 90:
            return {
                "level": "very_high",
                "description": "Very high humidity - risk of fungal issues",
                "severity": "high"
            }
        elif humidity > 70:
            return {
                "level": "high",
                "description": "High humidity - good for most plants",
                "severity": "low"
            }
        elif humidity < 20:
            return {
                "level": "very_low",
                "description": "Very low humidity - plant may suffer",
                "severity": "high"
            }
        elif humidity < 40:
            return {
                "level": "low",
                "description": "Low humidity - consider increasing",
                "severity": "moderate"
            }
        else:
            return {
                "level": "optimal",
                "description": "Humidity within good range",
                "severity": "low"
            }
    
    def _analyze_precipitation_impact(self, condition: str) -> Dict[str, Any]:
        """Analyze precipitation impact on plant care"""
        if "rain" in condition or "drizzle" in condition:
            return {
                "level": "wet",
                "description": "Precipitation present - adjust watering",
                "action": "skip_watering"
            }
        elif "snow" in condition:
            return {
                "level": "snow",
                "description": "Snow conditions - protect plant",
                "action": "protect_from_cold"
            }
        else:
            return {
                "level": "dry",
                "description": "No precipitation - normal watering",
                "action": "normal_care"
            }
    
    def _calculate_plant_stress_level(self, temp: float, humidity: float, condition: str) -> str:
        """Calculate overall plant stress level based on weather"""
        stress_points = 0
        
        # Temperature stress
        if temp > 35 or temp < 5:
            stress_points += 3
        elif temp > 30 or temp < 10:
            stress_points += 2
        elif temp > 25 or temp < 15:
            stress_points += 1
        
        # Humidity stress
        if humidity > 90 or humidity < 20:
            stress_points += 2
        elif humidity > 80 or humidity < 30:
            stress_points += 1
        
        # Weather condition stress
        if "storm" in condition or "extreme" in condition:
            stress_points += 2
        elif "rain" in condition and temp < 10:
            stress_points += 1
        
        if stress_points >= 5:
            return "critical"
        elif stress_points >= 3:
            return "high"
        elif stress_points >= 1:
            return "moderate"
        else:
            return "low"
    
    def _adjust_care_schedule(self, care_schedule: Dict[str, Any], weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust care schedule based on weather conditions"""
        try:
            if not care_schedule:
                return {"note": "No care schedule provided to adjust"}
            
            adjusted_schedule = care_schedule.copy()
            
            # Get weather adjustment factor
            adjustment_factor = WeatherHelper.get_weather_adjustment_factor(weather_data)
            
            # Check if watering should be skipped
            skip_watering = WeatherHelper.should_skip_watering(weather_data)
            
            # Adjust watering schedule
            if "watering" in adjusted_schedule:
                if skip_watering:
                    adjusted_schedule["watering"]["next_watering"] = "Skip due to rain/high humidity"
                    adjusted_schedule["watering"]["status"] = "postponed"
                else:
                    # Adjust frequency based on weather
                    if "frequency_days" in adjusted_schedule["watering"]:
                        original_freq = adjusted_schedule["watering"]["frequency_days"]
                        new_freq = max(1, int(original_freq / adjustment_factor))
                        adjusted_schedule["watering"]["frequency_days"] = new_freq
                        adjusted_schedule["watering"]["weather_adjusted"] = True
                        adjusted_schedule["watering"]["adjustment_factor"] = adjustment_factor
            
            # Add weather-specific notes
            adjusted_schedule["weather_notes"] = self._generate_schedule_notes(weather_data)
            adjusted_schedule["last_weather_update"] = datetime.now().isoformat()
            
            return adjusted_schedule
            
        except Exception as e:
            self.logger.error(f"Error adjusting care schedule: {e}")
            return care_schedule
    
    def _generate_weather_recommendations(self, weather_data: Dict[str, Any], plant_name: str) -> List[str]:
        """Generate specific recommendations based on current weather"""
        recommendations = []
        
        try:
            main = weather_data.get("main", {})
            weather = weather_data.get("weather", [{}])[0]
            
            temp = main.get("temp", 20)
            humidity = main.get("humidity", 50)
            condition = weather.get("main", "").lower()
            
            # Temperature-based recommendations
            if temp > 30:
                recommendations.extend([
                    "Move plant away from direct sunlight during peak hours",
                    "Increase watering frequency and check soil moisture daily",
                    "Consider misting leaves in the morning (if plant tolerates it)",
                    "Ensure good air circulation around the plant"
                ])
            elif temp < 10:
                recommendations.extend([
                    "Protect plant from cold drafts and sudden temperature changes",
                    "Reduce watering frequency as plant growth slows",
                    "Consider moving plant to a warmer location",
                    "Avoid fertilizing during cold periods"
                ])
            
            # Humidity-based recommendations
            if humidity > 80:
                recommendations.extend([
                    "Improve air circulation to prevent fungal diseases",
                    "Reduce watering frequency",
                    "Monitor for signs of root rot or leaf spot"
                ])
            elif humidity < 30:
                recommendations.extend([
                    "Increase humidity around plant with a pebble tray",
                    "Group plants together to create a microclimate",
                    "Consider using a humidifier nearby"
                ])
            
            # Weather condition recommendations
            if "rain" in condition:
                recommendations.extend([
                    "Skip today's watering schedule",
                    "Ensure plant has proper drainage",
                    "Check for water accumulation in saucers"
                ])
            elif "clear" in condition or "sun" in condition:
                recommendations.extend([
                    "Great day for photosynthesis - ensure plant gets adequate light",
                    "Monitor soil moisture as sunny conditions increase evaporation"
                ])
            
            # Seasonal recommendations
            current_month = datetime.now().month
            if current_month in [12, 1, 2]:  # Winter
                recommendations.append("Winter care: Reduce watering and fertilizing frequency")
            elif current_month in [6, 7, 8]:  # Summer
                recommendations.append("Summer care: Monitor for heat stress and increase humidity")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating weather recommendations: {e}")
            return ["Monitor plant closely and adjust care based on current conditions"]
    
    def _generate_schedule_notes(self, weather_data: Dict[str, Any]) -> List[str]:
        """Generate notes for schedule adjustments"""
        notes = []
        
        try:
            main = weather_data.get("main", {})
            weather = weather_data.get("weather", [{}])[0]
            
            temp = main.get("temp", 20)
            humidity = main.get("humidity", 50)
            condition = weather.get("main", "").lower()
            
            if "rain" in condition:
                notes.append("Watering postponed due to rain")
            
            if temp > 30:
                notes.append("Increased watering frequency due to high temperature")
            elif temp < 10:
                notes.append("Reduced watering frequency due to low temperature")
            
            if humidity > 80:
                notes.append("Watering reduced due to high humidity")
            elif humidity < 30:
                notes.append("Consider additional humidity measures")
            
            return notes
            
        except Exception:
            return ["Weather-based adjustments applied"]
    
    async def get_weather_forecast(self, location: str, days: int = 5) -> Dict[str, Any]:
        """Get weather forecast for planning ahead (placeholder for future implementation)"""
        try:
            # This would require the forecast API endpoint
            # For now, return current weather with a note
            current_weather = await APIHelper.call_openweather_api(location)
            
            return {
                "location": location,
                "current_weather": self._format_weather_data(current_weather),
                "forecast_note": "Forecast feature not yet implemented - showing current weather",
                "recommendation": "Check weather regularly and adjust plant care accordingly"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting weather forecast: {e}")
            return self.create_error_response(str(e), "WEATHER_FORECAST_ERROR")