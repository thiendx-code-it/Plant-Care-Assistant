import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta
from .base_agent import BaseAgent
from utils.api_helpers import WeatherHelper

class ScheduleManagerAgent(BaseAgent):
    """Agent responsible for generating and managing plant care schedules"""
    
    def __init__(self):
        super().__init__(
            name="Schedule Manager Agent",
            description="Generates watering/feeding schedules adjusted by plant needs and weather."
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive care schedule for plant"""
        try:
            # Validate input
            if not self.validate_input(input_data, ["plant_name"]):
                return self.create_error_response("Missing required field: plant_name")
            
            plant_name = input_data["plant_name"]
            care_info = input_data.get("care_info", {})
            weather_data = input_data.get("weather_data", {})
            health_status = input_data.get("health_status", {})
            user_preferences = input_data.get("user_preferences", {})
            
            # Generate base schedule from care information
            base_schedule = self._generate_base_schedule(plant_name, care_info)
            
            # Adjust schedule based on weather
            weather_adjusted_schedule = self._adjust_for_weather(base_schedule, weather_data)
            
            # Adjust schedule based on plant health
            health_adjusted_schedule = self._adjust_for_health(weather_adjusted_schedule, health_status)
            
            # Apply user preferences
            final_schedule = self._apply_user_preferences(health_adjusted_schedule, user_preferences)
            
            # Generate calendar events
            calendar_events = self._generate_calendar_events(final_schedule, plant_name)
            
            # Create reminders
            reminders = self._create_reminders(final_schedule, plant_name)
            
            result = {
                "plant_name": plant_name,
                "schedule": final_schedule,
                "calendar_events": calendar_events,
                "reminders": reminders,
                "schedule_summary": self._create_schedule_summary(final_schedule),
                "next_actions": self._get_next_actions(final_schedule),
                "created_at": datetime.now().isoformat(),
                "valid_until": (datetime.now() + timedelta(days=30)).isoformat()
            }
            
            return self.create_success_response(result)
            
        except Exception as e:
            self.logger.error(f"Error generating schedule: {e}")
            return self.create_error_response(str(e), "SCHEDULE_GENERATION_ERROR")
    
    def _generate_base_schedule(self, plant_name: str, care_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate base care schedule from plant care information"""
        try:
            # Extract care requirements
            watering_info = care_info.get("watering", "")
            fertilizing_info = care_info.get("soil_fertilization", "")
            
            # Parse watering frequency
            watering_frequency = self._parse_watering_frequency(watering_info)
            
            # Parse fertilizing frequency
            fertilizing_frequency = self._parse_fertilizing_frequency(fertilizing_info)
            
            # Generate base schedule
            base_schedule = {
                "watering": {
                    "frequency_days": watering_frequency,
                    "amount": self._determine_watering_amount(care_info),
                    "method": self._determine_watering_method(care_info),
                    "time_of_day": "morning",
                    "next_watering": self._calculate_next_date(watering_frequency),
                    "notes": self._extract_watering_notes(watering_info)
                },
                "fertilizing": {
                    "frequency_days": fertilizing_frequency,
                    "type": self._determine_fertilizer_type(fertilizing_info),
                    "strength": "diluted",
                    "next_fertilizing": self._calculate_next_date(fertilizing_frequency),
                    "notes": self._extract_fertilizing_notes(fertilizing_info)
                },
                "monitoring": {
                    "frequency_days": 3,  # Check every 3 days
                    "checks": ["soil moisture", "leaf condition", "growth", "pests"],
                    "next_check": self._calculate_next_date(3)
                },
                "maintenance": {
                    "pruning": {
                        "frequency_days": 30,
                        "next_pruning": self._calculate_next_date(30),
                        "type": "deadheading and light pruning"
                    },
                    "repotting": {
                        "frequency_days": 365,  # Yearly
                        "next_repotting": self._calculate_next_date(365),
                        "season": "spring"
                    }
                }
            }
            
            return base_schedule
            
        except Exception as e:
            self.logger.error(f"Error generating base schedule: {e}")
            return self._get_default_schedule()
    
    def _parse_watering_frequency(self, watering_info: str) -> int:
        """Parse watering frequency from care information"""
        try:
            watering_lower = watering_info.lower()
            
            # Look for specific frequency indicators
            if "daily" in watering_lower or "every day" in watering_lower:
                return 1
            elif "every other day" in watering_lower or "every 2 days" in watering_lower:
                return 2
            elif "twice a week" in watering_lower:
                return 3
            elif "weekly" in watering_lower or "once a week" in watering_lower:
                return 7
            elif "every 10 days" in watering_lower:
                return 10
            elif "bi-weekly" in watering_lower or "every 2 weeks" in watering_lower:
                return 14
            elif "monthly" in watering_lower:
                return 30
            elif "when dry" in watering_lower or "when soil is dry" in watering_lower:
                return 5  # Default to 5 days for "when dry"
            
            # Try to extract numbers
            import re
            numbers = re.findall(r'\d+', watering_info)
            if numbers:
                return min(int(numbers[0]), 30)  # Cap at 30 days
            
            # Default frequency
            return 7
            
        except Exception:
            return 7  # Default to weekly
    
    def _parse_fertilizing_frequency(self, fertilizing_info: str) -> int:
        """Parse fertilizing frequency from care information"""
        try:
            fertilizing_lower = fertilizing_info.lower()
            
            if "weekly" in fertilizing_lower:
                return 7
            elif "bi-weekly" in fertilizing_lower or "every 2 weeks" in fertilizing_lower:
                return 14
            elif "monthly" in fertilizing_lower or "once a month" in fertilizing_lower:
                return 30
            elif "every 6 weeks" in fertilizing_lower:
                return 42
            elif "every 2 months" in fertilizing_lower:
                return 60
            elif "quarterly" in fertilizing_lower or "every 3 months" in fertilizing_lower:
                return 90
            elif "growing season" in fertilizing_lower:
                return 21  # Every 3 weeks during growing season
            
            # Default to monthly
            return 30
            
        except Exception:
            return 30  # Default to monthly
    
    def _determine_watering_amount(self, care_info: Dict[str, Any]) -> str:
        """Determine appropriate watering amount"""
        watering_info = care_info.get("watering", "").lower()
        
        if "thoroughly" in watering_info or "deep" in watering_info:
            return "thorough"
        elif "lightly" in watering_info or "mist" in watering_info:
            return "light"
        elif "moderate" in watering_info:
            return "moderate"
        else:
            return "moderate"  # Default
    
    def _determine_watering_method(self, care_info: Dict[str, Any]) -> str:
        """Determine appropriate watering method"""
        watering_info = care_info.get("watering", "").lower()
        
        if "bottom" in watering_info:
            return "bottom watering"
        elif "mist" in watering_info or "spray" in watering_info:
            return "misting"
        elif "drip" in watering_info:
            return "drip irrigation"
        else:
            return "top watering"  # Default
    
    def _determine_fertilizer_type(self, fertilizing_info: str) -> str:
        """Determine appropriate fertilizer type"""
        fertilizing_lower = fertilizing_info.lower()
        
        if "liquid" in fertilizing_lower:
            return "liquid fertilizer"
        elif "granular" in fertilizing_lower or "pellet" in fertilizing_lower:
            return "granular fertilizer"
        elif "organic" in fertilizing_lower:
            return "organic fertilizer"
        elif "balanced" in fertilizing_lower:
            return "balanced liquid fertilizer"
        else:
            return "balanced liquid fertilizer"  # Default
    
    def _calculate_next_date(self, frequency_days: int) -> str:
        """Calculate next scheduled date"""
        next_date = datetime.now() + timedelta(days=frequency_days)
        return next_date.strftime("%Y-%m-%d")
    
    def _extract_watering_notes(self, watering_info: str) -> List[str]:
        """Extract important watering notes"""
        notes = []
        watering_lower = watering_info.lower()
        
        if "top inch" in watering_lower or "1 inch" in watering_lower:
            notes.append("Check top inch of soil before watering")
        if "drainage" in watering_lower:
            notes.append("Ensure good drainage")
        if "overwater" in watering_lower:
            notes.append("Avoid overwatering")
        if "morning" in watering_lower:
            notes.append("Water in the morning")
        if "room temperature" in watering_lower:
            notes.append("Use room temperature water")
        
        return notes
    
    def _extract_fertilizing_notes(self, fertilizing_info: str) -> List[str]:
        """Extract important fertilizing notes"""
        notes = []
        fertilizing_lower = fertilizing_info.lower()
        
        if "dilute" in fertilizing_lower or "half strength" in fertilizing_lower:
            notes.append("Use diluted fertilizer")
        if "growing season" in fertilizing_lower:
            notes.append("Fertilize only during growing season")
        if "winter" in fertilizing_lower and "stop" in fertilizing_lower:
            notes.append("Stop fertilizing in winter")
        
        return notes
    
    def _adjust_for_weather(self, schedule: Dict[str, Any], weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust schedule based on current weather conditions"""
        if not weather_data:
            return schedule
        
        try:
            adjusted_schedule = schedule.copy()
            
            # Get weather adjustment factor
            adjustment_factor = WeatherHelper.get_weather_adjustment_factor(weather_data)
            
            # Check if watering should be skipped
            skip_watering = WeatherHelper.should_skip_watering(weather_data)
            
            # Adjust watering schedule
            if "watering" in adjusted_schedule:
                if skip_watering:
                    # Postpone watering
                    current_date = datetime.now()
                    next_date = current_date + timedelta(days=1)
                    adjusted_schedule["watering"]["next_watering"] = next_date.strftime("%Y-%m-%d")
                    adjusted_schedule["watering"]["weather_note"] = "Postponed due to rain/high humidity"
                else:
                    # Adjust frequency
                    original_freq = adjusted_schedule["watering"]["frequency_days"]
                    new_freq = max(1, int(original_freq / adjustment_factor))
                    adjusted_schedule["watering"]["frequency_days"] = new_freq
                    adjusted_schedule["watering"]["weather_adjusted"] = True
                    adjusted_schedule["watering"]["adjustment_factor"] = adjustment_factor
            
            return adjusted_schedule
            
        except Exception as e:
            self.logger.error(f"Error adjusting schedule for weather: {e}")
            return schedule
    
    def _adjust_for_health(self, schedule: Dict[str, Any], health_status: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust schedule based on plant health status"""
        if not health_status:
            return schedule
        
        try:
            adjusted_schedule = schedule.copy()
            
            is_healthy = health_status.get("is_healthy", True)
            health_score = health_status.get("health_score", 1.0)
            diseases = health_status.get("diseases", [])
            
            # Adjust based on health status
            if not is_healthy or health_score < 0.7:
                # Reduce watering frequency for unhealthy plants
                if "watering" in adjusted_schedule:
                    current_freq = adjusted_schedule["watering"]["frequency_days"]
                    adjusted_schedule["watering"]["frequency_days"] = int(current_freq * 1.3)
                    adjusted_schedule["watering"]["health_note"] = "Reduced frequency due to health issues"
                
                # Increase monitoring frequency
                if "monitoring" in adjusted_schedule:
                    adjusted_schedule["monitoring"]["frequency_days"] = 1  # Daily monitoring
                    adjusted_schedule["monitoring"]["health_focus"] = True
            
            # Specific disease adjustments
            if diseases:
                disease_names = [d.get("name", "").lower() for d in diseases]
                
                if any("root rot" in name for name in disease_names):
                    # Significantly reduce watering for root rot
                    if "watering" in adjusted_schedule:
                        current_freq = adjusted_schedule["watering"]["frequency_days"]
                        adjusted_schedule["watering"]["frequency_days"] = int(current_freq * 2)
                        adjusted_schedule["watering"]["disease_note"] = "Reduced watering due to root rot risk"
                
                if any("fungal" in name for name in disease_names):
                    # Adjust watering method for fungal issues
                    if "watering" in adjusted_schedule:
                        adjusted_schedule["watering"]["method"] = "bottom watering"
                        adjusted_schedule["watering"]["fungal_note"] = "Bottom watering to prevent fungal spread"
            
            return adjusted_schedule
            
        except Exception as e:
            self.logger.error(f"Error adjusting schedule for health: {e}")
            return schedule
    
    def _apply_user_preferences(self, schedule: Dict[str, Any], user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Apply user preferences to the schedule"""
        if not user_preferences:
            return schedule
        
        try:
            adjusted_schedule = schedule.copy()
            
            # Preferred watering time
            preferred_time = user_preferences.get("watering_time", "morning")
            if "watering" in adjusted_schedule:
                adjusted_schedule["watering"]["time_of_day"] = preferred_time
            
            # Watering frequency preference
            if "watering_frequency_adjustment" in user_preferences:
                adjustment = user_preferences["watering_frequency_adjustment"]
                if "watering" in adjusted_schedule:
                    current_freq = adjusted_schedule["watering"]["frequency_days"]
                    if adjustment == "more_frequent":
                        adjusted_schedule["watering"]["frequency_days"] = max(1, int(current_freq * 0.8))
                    elif adjustment == "less_frequent":
                        adjusted_schedule["watering"]["frequency_days"] = int(current_freq * 1.2)
            
            # Fertilizing preferences
            if "organic_only" in user_preferences and user_preferences["organic_only"]:
                if "fertilizing" in adjusted_schedule:
                    adjusted_schedule["fertilizing"]["type"] = "organic fertilizer"
            
            return adjusted_schedule
            
        except Exception as e:
            self.logger.error(f"Error applying user preferences: {e}")
            return schedule
    
    def _generate_calendar_events(self, schedule: Dict[str, Any], plant_name: str) -> List[Dict[str, Any]]:
        """Generate calendar events for the care schedule"""
        events = []
        
        try:
            # Watering events
            if "watering" in schedule:
                watering = schedule["watering"]
                next_watering = watering.get("next_watering")
                frequency = watering.get("frequency_days", 7)
                
                # Generate next 4 watering events
                for i in range(4):
                    event_date = datetime.strptime(next_watering, "%Y-%m-%d") + timedelta(days=i * frequency)
                    events.append({
                        "type": "watering",
                        "title": f"Water {plant_name}",
                        "date": event_date.strftime("%Y-%m-%d"),
                        "time": watering.get("time_of_day", "morning"),
                        "description": f"Water {plant_name} - {watering.get('amount', 'moderate')} amount",
                        "notes": watering.get("notes", [])
                    })
            
            # Fertilizing events
            if "fertilizing" in schedule:
                fertilizing = schedule["fertilizing"]
                next_fertilizing = fertilizing.get("next_fertilizing")
                frequency = fertilizing.get("frequency_days", 30)
                
                # Generate next 3 fertilizing events
                for i in range(3):
                    event_date = datetime.strptime(next_fertilizing, "%Y-%m-%d") + timedelta(days=i * frequency)
                    events.append({
                        "type": "fertilizing",
                        "title": f"Fertilize {plant_name}",
                        "date": event_date.strftime("%Y-%m-%d"),
                        "description": f"Fertilize {plant_name} with {fertilizing.get('type', 'balanced fertilizer')}",
                        "notes": fertilizing.get("notes", [])
                    })
            
            # Monitoring events
            if "monitoring" in schedule:
                monitoring = schedule["monitoring"]
                next_check = monitoring.get("next_check")
                frequency = monitoring.get("frequency_days", 3)
                
                # Generate next 5 monitoring events
                for i in range(5):
                    event_date = datetime.strptime(next_check, "%Y-%m-%d") + timedelta(days=i * frequency)
                    events.append({
                        "type": "monitoring",
                        "title": f"Check {plant_name}",
                        "date": event_date.strftime("%Y-%m-%d"),
                        "description": f"Monitor {plant_name} health and growth",
                        "checks": monitoring.get("checks", [])
                    })
            
            # Sort events by date
            events.sort(key=lambda x: x["date"])
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error generating calendar events: {e}")
            return []
    
    def _create_reminders(self, schedule: Dict[str, Any], plant_name: str) -> List[Dict[str, Any]]:
        """Create reminders for upcoming care tasks"""
        reminders = []
        
        try:
            current_date = datetime.now().date()
            
            # Check for tasks due today or tomorrow
            for event in self._generate_calendar_events(schedule, plant_name):
                event_date = datetime.strptime(event["date"], "%Y-%m-%d").date()
                days_until = (event_date - current_date).days
                
                if days_until <= 1:  # Due today or tomorrow
                    urgency = "urgent" if days_until == 0 else "upcoming"
                    reminders.append({
                        "type": event["type"],
                        "title": event["title"],
                        "due_date": event["date"],
                        "urgency": urgency,
                        "message": self._create_reminder_message(event, days_until),
                        "action_required": True
                    })
            
            return reminders
            
        except Exception as e:
            self.logger.error(f"Error creating reminders: {e}")
            return []
    
    def _create_reminder_message(self, event: Dict[str, Any], days_until: int) -> str:
        """Create a user-friendly reminder message"""
        if days_until == 0:
            return f"🌱 {event['title']} is due today!"
        elif days_until == 1:
            return f"📅 {event['title']} is due tomorrow."
        else:
            return f"📋 {event['title']} is coming up in {days_until} days."
    
    def _create_schedule_summary(self, schedule: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the care schedule"""
        try:
            summary = {
                "watering_frequency": f"Every {schedule.get('watering', {}).get('frequency_days', 7)} days",
                "fertilizing_frequency": f"Every {schedule.get('fertilizing', {}).get('frequency_days', 30)} days",
                "monitoring_frequency": f"Every {schedule.get('monitoring', {}).get('frequency_days', 3)} days",
                "next_watering": schedule.get('watering', {}).get('next_watering', 'Not scheduled'),
                "next_fertilizing": schedule.get('fertilizing', {}).get('next_fertilizing', 'Not scheduled'),
                "special_notes": []
            }
            
            # Add special notes
            if schedule.get('watering', {}).get('weather_adjusted'):
                summary["special_notes"].append("Watering schedule adjusted for weather")
            
            if schedule.get('watering', {}).get('health_note'):
                summary["special_notes"].append("Schedule modified due to plant health")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error creating schedule summary: {e}")
            return {"error": "Could not create summary"}
    
    def _get_next_actions(self, schedule: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get the next 3 actions to take"""
        try:
            actions = []
            current_date = datetime.now().date()
            
            # Check watering
            if "watering" in schedule:
                next_watering = datetime.strptime(schedule["watering"]["next_watering"], "%Y-%m-%d").date()
                days_until = (next_watering - current_date).days
                actions.append({
                    "action": "watering",
                    "description": "Water the plant",
                    "due_in_days": days_until,
                    "priority": "high" if days_until <= 0 else "medium"
                })
            
            # Check fertilizing
            if "fertilizing" in schedule:
                next_fertilizing = datetime.strptime(schedule["fertilizing"]["next_fertilizing"], "%Y-%m-%d").date()
                days_until = (next_fertilizing - current_date).days
                actions.append({
                    "action": "fertilizing",
                    "description": "Fertilize the plant",
                    "due_in_days": days_until,
                    "priority": "medium" if days_until <= 7 else "low"
                })
            
            # Check monitoring
            if "monitoring" in schedule:
                next_check = datetime.strptime(schedule["monitoring"]["next_check"], "%Y-%m-%d").date()
                days_until = (next_check - current_date).days
                actions.append({
                    "action": "monitoring",
                    "description": "Check plant health",
                    "due_in_days": days_until,
                    "priority": "high" if days_until <= 0 else "low"
                })
            
            # Sort by priority and due date
            priority_order = {"high": 0, "medium": 1, "low": 2}
            actions.sort(key=lambda x: (priority_order.get(x["priority"], 3), x["due_in_days"]))
            
            return actions[:3]  # Return top 3 actions
            
        except Exception as e:
            self.logger.error(f"Error getting next actions: {e}")
            return []
    
    def _get_default_schedule(self) -> Dict[str, Any]:
        """Get default schedule when generation fails"""
        return {
            "watering": {
                "frequency_days": 7,
                "amount": "moderate",
                "method": "top watering",
                "time_of_day": "morning",
                "next_watering": self._calculate_next_date(7),
                "notes": ["Check soil moisture before watering"]
            },
            "fertilizing": {
                "frequency_days": 30,
                "type": "balanced liquid fertilizer",
                "strength": "diluted",
                "next_fertilizing": self._calculate_next_date(30),
                "notes": ["Use during growing season only"]
            },
            "monitoring": {
                "frequency_days": 3,
                "checks": ["soil moisture", "leaf condition"],
                "next_check": self._calculate_next_date(3)
            }
        }