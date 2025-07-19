import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from .base_agent import BaseAgent
from utils.image_utils import encode_image_to_base64, resize_image

class GrowthTrackerAgent(BaseAgent):
    """Agent responsible for tracking plant growth and development over time"""
    
    def __init__(self):
        super().__init__(
            name="Growth Tracker Agent",
            description="Tracks plant growth, analyzes development patterns, and provides insights."
        )
        self.growth_data_file = "data/growth_tracking.json"
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track plant growth and analyze development patterns"""
        try:
            action = input_data.get("action", "analyze_growth")
            
            if action == "record_measurement":
                return await self._record_measurement(input_data)
            elif action == "analyze_growth":
                return await self._analyze_growth(input_data)
            elif action == "compare_images":
                return await self._compare_images(input_data)
            elif action == "get_growth_report":
                return await self._get_growth_report(input_data)
            elif action == "predict_growth":
                return await self._predict_growth(input_data)
            else:
                return self.create_error_response(f"Unknown action: {action}")
                
        except Exception as e:
            self.logger.error(f"Error in growth tracking: {e}")
            return self.create_error_response(str(e), "GROWTH_TRACKING_ERROR")
    
    async def _record_measurement(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Record new growth measurement"""
        try:
            # Validate input
            if not self.validate_input(input_data, ["plant_id"]):
                return self.create_error_response("Missing required field: plant_id")
            
            plant_id = input_data["plant_id"]
            measurement_data = {
                "timestamp": datetime.now().isoformat(),
                "plant_id": plant_id,
                "measurements": input_data.get("measurements", {}),
                "image_data": input_data.get("image_data"),
                "notes": input_data.get("notes", ""),
                "environmental_conditions": input_data.get("environmental_conditions", {}),
                "care_actions": input_data.get("care_actions", [])
            }
            
            # Process image if provided
            if measurement_data["image_data"]:
                processed_image = await self._process_growth_image(measurement_data["image_data"])
                measurement_data["processed_image"] = processed_image
            
            # Save measurement
            saved_measurement = self._save_measurement(measurement_data)
            
            # Analyze recent growth
            growth_analysis = await self._analyze_recent_growth(plant_id)
            
            result = {
                "measurement_id": saved_measurement["id"],
                "recorded_at": saved_measurement["timestamp"],
                "plant_id": plant_id,
                "measurements": saved_measurement["measurements"],
                "growth_analysis": growth_analysis,
                "recommendations": self._generate_growth_recommendations(growth_analysis)
            }
            
            return self.create_success_response(result)
            
        except Exception as e:
            self.logger.error(f"Error recording measurement: {e}")
            return self.create_error_response(str(e), "MEASUREMENT_RECORDING_ERROR")
    
    async def _analyze_growth(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze growth patterns for a plant"""
        try:
            if not self.validate_input(input_data, ["plant_id"]):
                return self.create_error_response("Missing required field: plant_id")
            
            plant_id = input_data["plant_id"]
            time_period = input_data.get("time_period", 30)  # days
            
            # Get growth data
            growth_data = self._get_growth_data(plant_id, time_period)
            
            if not growth_data:
                return self.create_error_response("No growth data found for this plant")
            
            # Analyze growth patterns
            growth_analysis = {
                "plant_id": plant_id,
                "analysis_period": f"{time_period} days",
                "total_measurements": len(growth_data),
                "growth_trends": self._analyze_growth_trends(growth_data),
                "growth_rate": self._calculate_growth_rate(growth_data),
                "health_indicators": self._analyze_health_indicators(growth_data),
                "milestone_progress": self._track_milestones(growth_data),
                "environmental_correlations": self._analyze_environmental_correlations(growth_data),
                "care_effectiveness": self._analyze_care_effectiveness(growth_data)
            }
            
            # Generate insights using LLM
            llm_insights = await self._generate_growth_insights(growth_analysis)
            growth_analysis["ai_insights"] = llm_insights
            
            return self.create_success_response(growth_analysis)
            
        except Exception as e:
            self.logger.error(f"Error analyzing growth: {e}")
            return self.create_error_response(str(e), "GROWTH_ANALYSIS_ERROR")
    
    async def _compare_images(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare plant images to track visual growth changes"""
        try:
            if not self.validate_input(input_data, ["plant_id"]):
                return self.create_error_response("Missing required field: plant_id")
            
            plant_id = input_data["plant_id"]
            current_image = input_data.get("current_image")
            comparison_period = input_data.get("comparison_period", 7)  # days
            
            # Get historical images
            historical_images = self._get_historical_images(plant_id, comparison_period)
            
            if not historical_images:
                return self.create_error_response("No historical images found for comparison")
            
            # Analyze visual changes using LLM
            visual_analysis = await self._analyze_visual_changes(
                current_image, historical_images, plant_id
            )
            
            result = {
                "plant_id": plant_id,
                "comparison_period": f"{comparison_period} days",
                "images_compared": len(historical_images) + (1 if current_image else 0),
                "visual_changes": visual_analysis,
                "growth_indicators": self._extract_visual_growth_indicators(visual_analysis),
                "recommendations": self._generate_visual_recommendations(visual_analysis)
            }
            
            return self.create_success_response(result)
            
        except Exception as e:
            self.logger.error(f"Error comparing images: {e}")
            return self.create_error_response(str(e), "IMAGE_COMPARISON_ERROR")
    
    async def _get_growth_report(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive growth report"""
        try:
            if not self.validate_input(input_data, ["plant_id"]):
                return self.create_error_response("Missing required field: plant_id")
            
            plant_id = input_data["plant_id"]
            report_period = input_data.get("report_period", 90)  # days
            
            # Get comprehensive growth data
            growth_data = self._get_growth_data(plant_id, report_period)
            
            if not growth_data:
                return self.create_error_response("Insufficient data for growth report")
            
            # Generate comprehensive report
            growth_report = {
                "plant_id": plant_id,
                "report_period": f"{report_period} days",
                "generated_at": datetime.now().isoformat(),
                "summary": self._generate_growth_summary(growth_data),
                "detailed_analysis": {
                    "growth_metrics": self._calculate_detailed_metrics(growth_data),
                    "trend_analysis": self._analyze_detailed_trends(growth_data),
                    "health_progression": self._analyze_health_progression(growth_data),
                    "care_impact": self._analyze_care_impact(growth_data)
                },
                "milestones": self._identify_growth_milestones(growth_data),
                "predictions": await self._generate_growth_predictions(growth_data),
                "recommendations": self._generate_comprehensive_recommendations(growth_data),
                "charts_data": self._prepare_chart_data(growth_data)
            }
            
            return self.create_success_response(growth_report)
            
        except Exception as e:
            self.logger.error(f"Error generating growth report: {e}")
            return self.create_error_response(str(e), "GROWTH_REPORT_ERROR")
    
    async def _predict_growth(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future growth based on current patterns"""
        try:
            if not self.validate_input(input_data, ["plant_id"]):
                return self.create_error_response("Missing required field: plant_id")
            
            plant_id = input_data["plant_id"]
            prediction_period = input_data.get("prediction_period", 30)  # days
            
            # Get recent growth data
            growth_data = self._get_growth_data(plant_id, 60)  # Use 60 days for prediction
            
            if len(growth_data) < 3:
                return self.create_error_response("Insufficient data for growth prediction")
            
            # Generate predictions
            predictions = await self._generate_detailed_predictions(
                growth_data, prediction_period
            )
            
            result = {
                "plant_id": plant_id,
                "prediction_period": f"{prediction_period} days",
                "based_on_data_points": len(growth_data),
                "predictions": predictions,
                "confidence_level": self._calculate_prediction_confidence(growth_data),
                "factors_considered": [
                    "Historical growth rate",
                    "Seasonal patterns",
                    "Care consistency",
                    "Environmental conditions",
                    "Plant health trends"
                ],
                "recommendations": self._generate_prediction_recommendations(predictions)
            }
            
            return self.create_success_response(result)
            
        except Exception as e:
            self.logger.error(f"Error predicting growth: {e}")
            return self.create_error_response(str(e), "GROWTH_PREDICTION_ERROR")
    
    async def _process_growth_image(self, image_data: Any) -> Dict[str, Any]:
        """Process and analyze growth image"""
        try:
            # Encode image
            if hasattr(image_data, 'read'):
                image_data.seek(0)
                encoded_image = encode_image_to_base64(image_data)
            else:
                encoded_image = image_data
            
            # Resize for consistency
            resized_image = resize_image(image_data, max_size=(800, 800))
            
            # Extract visual features using LLM
            visual_features = await self._extract_visual_features(encoded_image)
            
            return {
                "encoded_image": encoded_image,
                "visual_features": visual_features,
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing growth image: {e}")
            return {"error": str(e)}
    
    async def _extract_visual_features(self, encoded_image: str) -> Dict[str, Any]:
        """Extract visual features from plant image using LLM"""
        try:
            system_message = self.create_system_message(
                "You are an expert plant growth analyst. Analyze the plant image and extract key visual features for growth tracking."
            )
            
            human_message = self.create_human_message(
                "Analyze this plant image and provide detailed observations about:\n"
                "1. Overall plant size and structure\n"
                "2. Leaf count, size, and condition\n"
                "3. Stem thickness and height\n"
                "4. New growth indicators\n"
                "5. Color and health indicators\n"
                "6. Any flowers, buds, or fruits\n"
                "7. Overall growth stage\n\n"
                "Provide specific, measurable observations where possible.",
                image_data=encoded_image
            )
            
            response = await self.call_llm([system_message, human_message])
            
            # Parse response into structured format
            return {
                "analysis": response,
                "extracted_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting visual features: {e}")
            return {"error": str(e)}
    
    def _save_measurement(self, measurement_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save measurement to growth tracking file"""
        try:
            # Generate unique ID
            measurement_id = f"{measurement_data['plant_id']}_{int(datetime.now().timestamp())}"
            measurement_data["id"] = measurement_id
            
            # Load existing data
            try:
                with open(self.growth_data_file, 'r') as f:
                    all_data = json.load(f)
            except FileNotFoundError:
                all_data = {"measurements": []}
            
            # Add new measurement
            all_data["measurements"].append(measurement_data)
            
            # Save updated data
            import os
            os.makedirs(os.path.dirname(self.growth_data_file), exist_ok=True)
            with open(self.growth_data_file, 'w') as f:
                json.dump(all_data, f, indent=2)
            
            return measurement_data
            
        except Exception as e:
            self.logger.error(f"Error saving measurement: {e}")
            raise
    
    def _get_growth_data(self, plant_id: str, days: int) -> List[Dict[str, Any]]:
        """Get growth data for a plant within specified days"""
        try:
            with open(self.growth_data_file, 'r') as f:
                all_data = json.load(f)
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            filtered_data = []
            for measurement in all_data.get("measurements", []):
                if measurement["plant_id"] == plant_id:
                    measurement_date = datetime.fromisoformat(measurement["timestamp"])
                    if measurement_date >= cutoff_date:
                        filtered_data.append(measurement)
            
            # Sort by timestamp
            filtered_data.sort(key=lambda x: x["timestamp"])
            
            return filtered_data
            
        except FileNotFoundError:
            return []
        except Exception as e:
            self.logger.error(f"Error getting growth data: {e}")
            return []
    
    def _analyze_growth_trends(self, growth_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze growth trends from measurement data"""
        try:
            if len(growth_data) < 2:
                return {"trend": "insufficient_data"}
            
            trends = {}
            
            # Analyze height trend
            heights = [m.get("measurements", {}).get("height") for m in growth_data if m.get("measurements", {}).get("height")]
            if len(heights) >= 2:
                height_change = heights[-1] - heights[0]
                trends["height"] = {
                    "change": height_change,
                    "trend": "increasing" if height_change > 0 else "stable" if height_change == 0 else "decreasing",
                    "rate_per_week": (height_change / len(heights)) * 7 if len(heights) > 1 else 0
                }
            
            # Analyze leaf count trend
            leaf_counts = [m.get("measurements", {}).get("leaf_count") for m in growth_data if m.get("measurements", {}).get("leaf_count")]
            if len(leaf_counts) >= 2:
                leaf_change = leaf_counts[-1] - leaf_counts[0]
                trends["leaf_count"] = {
                    "change": leaf_change,
                    "trend": "increasing" if leaf_change > 0 else "stable" if leaf_change == 0 else "decreasing",
                    "rate_per_week": (leaf_change / len(leaf_counts)) * 7 if len(leaf_counts) > 1 else 0
                }
            
            # Analyze overall growth stage progression
            stages = [m.get("measurements", {}).get("growth_stage") for m in growth_data if m.get("measurements", {}).get("growth_stage")]
            if stages:
                trends["growth_stage"] = {
                    "current_stage": stages[-1],
                    "progression": len(set(stages)),
                    "stages_observed": list(set(stages))
                }
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing growth trends: {e}")
            return {"error": str(e)}
    
    def _calculate_growth_rate(self, growth_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate various growth rates"""
        try:
            if len(growth_data) < 2:
                return {"status": "insufficient_data"}
            
            first_measurement = growth_data[0]
            last_measurement = growth_data[-1]
            
            first_date = datetime.fromisoformat(first_measurement["timestamp"])
            last_date = datetime.fromisoformat(last_measurement["timestamp"])
            days_elapsed = (last_date - first_date).days
            
            if days_elapsed == 0:
                return {"status": "no_time_elapsed"}
            
            growth_rates = {}
            
            # Height growth rate
            first_height = first_measurement.get("measurements", {}).get("height")
            last_height = last_measurement.get("measurements", {}).get("height")
            if first_height and last_height:
                height_rate = (last_height - first_height) / days_elapsed
                growth_rates["height_per_day"] = height_rate
                growth_rates["height_per_week"] = height_rate * 7
            
            # Leaf growth rate
            first_leaves = first_measurement.get("measurements", {}).get("leaf_count")
            last_leaves = last_measurement.get("measurements", {}).get("leaf_count")
            if first_leaves and last_leaves:
                leaf_rate = (last_leaves - first_leaves) / days_elapsed
                growth_rates["leaves_per_day"] = leaf_rate
                growth_rates["leaves_per_week"] = leaf_rate * 7
            
            growth_rates["measurement_period_days"] = days_elapsed
            growth_rates["total_measurements"] = len(growth_data)
            
            return growth_rates
            
        except Exception as e:
            self.logger.error(f"Error calculating growth rate: {e}")
            return {"error": str(e)}
    
    def _analyze_health_indicators(self, growth_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze health indicators from growth data"""
        try:
            health_indicators = {
                "overall_trend": "stable",
                "concerns": [],
                "positive_signs": []
            }
            
            # Check for consistent growth
            heights = [m.get("measurements", {}).get("height") for m in growth_data if m.get("measurements", {}).get("height")]
            if len(heights) >= 3:
                recent_growth = heights[-1] - heights[-2] if len(heights) >= 2 else 0
                if recent_growth > 0:
                    health_indicators["positive_signs"].append("Consistent height growth")
                elif recent_growth < 0:
                    health_indicators["concerns"].append("Height decrease detected")
            
            # Check leaf health trends
            leaf_counts = [m.get("measurements", {}).get("leaf_count") for m in growth_data if m.get("measurements", {}).get("leaf_count")]
            if len(leaf_counts) >= 2:
                leaf_trend = leaf_counts[-1] - leaf_counts[0]
                if leaf_trend > 0:
                    health_indicators["positive_signs"].append("Increasing leaf count")
                elif leaf_trend < -2:  # Significant leaf loss
                    health_indicators["concerns"].append("Significant leaf loss")
            
            # Check for notes about health issues
            recent_notes = [m.get("notes", "") for m in growth_data[-3:]]  # Last 3 measurements
            health_keywords = ["yellow", "brown", "wilting", "pest", "disease", "drooping"]
            for note in recent_notes:
                if any(keyword in note.lower() for keyword in health_keywords):
                    health_indicators["concerns"].append("Health issues noted in recent observations")
                    break
            
            # Determine overall trend
            if len(health_indicators["concerns"]) > len(health_indicators["positive_signs"]):
                health_indicators["overall_trend"] = "declining"
            elif len(health_indicators["positive_signs"]) > 0:
                health_indicators["overall_trend"] = "improving"
            
            return health_indicators
            
        except Exception as e:
            self.logger.error(f"Error analyzing health indicators: {e}")
            return {"error": str(e)}
    
    def _track_milestones(self, growth_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track growth milestones"""
        try:
            milestones = {
                "achieved": [],
                "upcoming": [],
                "progress": {}
            }
            
            # Define common milestones
            height_milestones = [10, 20, 30, 50, 100]  # cm
            leaf_milestones = [5, 10, 20, 50, 100]
            
            # Get current measurements
            if growth_data:
                latest = growth_data[-1]
                current_height = latest.get("measurements", {}).get("height", 0)
                current_leaves = latest.get("measurements", {}).get("leaf_count", 0)
                
                # Check height milestones
                for milestone in height_milestones:
                    if current_height >= milestone:
                        milestones["achieved"].append(f"Reached {milestone}cm height")
                    elif current_height < milestone:
                        milestones["upcoming"].append(f"Reach {milestone}cm height")
                        break
                
                # Check leaf milestones
                for milestone in leaf_milestones:
                    if current_leaves >= milestone:
                        milestones["achieved"].append(f"Reached {milestone} leaves")
                    elif current_leaves < milestone:
                        milestones["upcoming"].append(f"Reach {milestone} leaves")
                        break
                
                # Calculate progress to next milestone
                next_height_milestone = next((m for m in height_milestones if m > current_height), None)
                if next_height_milestone:
                    progress = (current_height / next_height_milestone) * 100
                    milestones["progress"]["height"] = {
                        "current": current_height,
                        "target": next_height_milestone,
                        "progress_percent": min(progress, 100)
                    }
                
                next_leaf_milestone = next((m for m in leaf_milestones if m > current_leaves), None)
                if next_leaf_milestone:
                    progress = (current_leaves / next_leaf_milestone) * 100
                    milestones["progress"]["leaves"] = {
                        "current": current_leaves,
                        "target": next_leaf_milestone,
                        "progress_percent": min(progress, 100)
                    }
            
            return milestones
            
        except Exception as e:
            self.logger.error(f"Error tracking milestones: {e}")
            return {"error": str(e)}
    
    def _analyze_environmental_correlations(self, growth_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlations between environmental conditions and growth"""
        try:
            correlations = {
                "temperature_impact": "unknown",
                "humidity_impact": "unknown",
                "light_impact": "unknown",
                "observations": []
            }
            
            # Analyze temperature correlation
            temp_growth_pairs = []
            for measurement in growth_data:
                temp = measurement.get("environmental_conditions", {}).get("temperature")
                height = measurement.get("measurements", {}).get("height")
                if temp and height:
                    temp_growth_pairs.append((temp, height))
            
            if len(temp_growth_pairs) >= 3:
                # Simple correlation analysis
                temps = [pair[0] for pair in temp_growth_pairs]
                heights = [pair[1] for pair in temp_growth_pairs]
                
                if max(temps) - min(temps) > 5:  # Significant temperature variation
                    avg_temp = sum(temps) / len(temps)
                    high_temp_growth = [h for t, h in temp_growth_pairs if t > avg_temp]
                    low_temp_growth = [h for t, h in temp_growth_pairs if t <= avg_temp]
                    
                    if high_temp_growth and low_temp_growth:
                        avg_high = sum(high_temp_growth) / len(high_temp_growth)
                        avg_low = sum(low_temp_growth) / len(low_temp_growth)
                        
                        if avg_high > avg_low:
                            correlations["temperature_impact"] = "positive"
                            correlations["observations"].append("Higher temperatures correlate with better growth")
                        else:
                            correlations["temperature_impact"] = "negative"
                            correlations["observations"].append("Lower temperatures correlate with better growth")
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error analyzing environmental correlations: {e}")
            return {"error": str(e)}
    
    def _analyze_care_effectiveness(self, growth_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze effectiveness of care actions"""
        try:
            care_analysis = {
                "watering_effectiveness": "unknown",
                "fertilizing_effectiveness": "unknown",
                "care_recommendations": []
            }
            
            # Analyze growth after care actions
            for i, measurement in enumerate(growth_data[:-1]):
                care_actions = measurement.get("care_actions", [])
                if care_actions and i < len(growth_data) - 1:
                    next_measurement = growth_data[i + 1]
                    
                    current_height = measurement.get("measurements", {}).get("height", 0)
                    next_height = next_measurement.get("measurements", {}).get("height", 0)
                    growth_response = next_height - current_height
                    
                    for action in care_actions:
                        if "water" in action.lower() and growth_response > 0:
                            care_analysis["care_recommendations"].append("Watering shows positive growth response")
                        elif "fertiliz" in action.lower() and growth_response > 0:
                            care_analysis["care_recommendations"].append("Fertilizing shows positive growth response")
            
            return care_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing care effectiveness: {e}")
            return {"error": str(e)}
    
    async def _analyze_recent_growth(self, plant_id: str) -> Dict[str, Any]:
        """Analyze recent growth for a plant"""
        try:
            recent_data = self._get_growth_data(plant_id, 14)  # Last 2 weeks
            
            if len(recent_data) < 2:
                return {"status": "insufficient_recent_data"}
            
            analysis = {
                "recent_trend": self._analyze_growth_trends(recent_data),
                "growth_rate": self._calculate_growth_rate(recent_data),
                "health_status": self._analyze_health_indicators(recent_data)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing recent growth: {e}")
            return {"error": str(e)}
    
    def _generate_growth_recommendations(self, growth_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on growth analysis"""
        recommendations = []
        
        try:
            # Check growth trends
            trends = growth_analysis.get("recent_trend", {})
            
            if trends.get("height", {}).get("trend") == "decreasing":
                recommendations.append("Consider adjusting care routine - height growth has decreased")
            
            if trends.get("leaf_count", {}).get("trend") == "decreasing":
                recommendations.append("Monitor for stress factors - leaf count is declining")
            
            # Check health indicators
            health = growth_analysis.get("health_status", {})
            if health.get("concerns"):
                recommendations.append("Address health concerns noted in recent observations")
            
            # Check growth rate
            growth_rate = growth_analysis.get("growth_rate", {})
            if growth_rate.get("height_per_week", 0) < 0.1:
                recommendations.append("Consider optimizing growing conditions for better growth rate")
            
            if not recommendations:
                recommendations.append("Continue current care routine - growth appears healthy")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating growth recommendations: {e}")
            return ["Unable to generate recommendations due to analysis error"]
    
    async def _generate_growth_insights(self, growth_analysis: Dict[str, Any]) -> str:
        """Generate AI insights about growth patterns"""
        try:
            system_message = self.create_system_message(
                "You are an expert plant growth analyst. Provide insights about plant growth patterns and development."
            )
            
            analysis_summary = json.dumps(growth_analysis, indent=2)
            
            human_message = self.create_human_message(
                f"Based on this growth analysis data, provide expert insights about:\n"
                f"1. Overall growth health and patterns\n"
                f"2. Key trends and what they indicate\n"
                f"3. Potential concerns or positive indicators\n"
                f"4. Recommendations for optimizing growth\n\n"
                f"Growth Analysis Data:\n{analysis_summary}"
            )
            
            response = await self.call_llm([system_message, human_message])
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating growth insights: {e}")
            return "Unable to generate insights due to analysis error"
    
    def _get_historical_images(self, plant_id: str, days: int) -> List[Dict[str, Any]]:
        """Get historical images for comparison"""
        try:
            growth_data = self._get_growth_data(plant_id, days)
            images = []
            
            for measurement in growth_data:
                if measurement.get("processed_image"):
                    images.append({
                        "timestamp": measurement["timestamp"],
                        "image_data": measurement["processed_image"],
                        "measurements": measurement.get("measurements", {})
                    })
            
            return images
            
        except Exception as e:
            self.logger.error(f"Error getting historical images: {e}")
            return []
    
    async def _analyze_visual_changes(self, current_image: Any, historical_images: List[Dict[str, Any]], plant_id: str) -> Dict[str, Any]:
        """Analyze visual changes between images"""
        try:
            if not historical_images:
                return {"status": "no_historical_images"}
            
            # Compare with most recent historical image
            latest_historical = historical_images[-1]
            
            system_message = self.create_system_message(
                "You are an expert plant growth analyst. Compare these plant images to identify growth changes and development."
            )
            
            comparison_prompt = (
                "Compare these two plant images and identify:\n"
                "1. Changes in overall size and structure\n"
                "2. New leaf growth or leaf changes\n"
                "3. Stem development\n"
                "4. Color changes\n"
                "5. New buds, flowers, or fruits\n"
                "6. Overall health changes\n"
                "7. Growth stage progression\n\n"
                "Provide specific observations about what has changed between the images."
            )
            
            if current_image:
                # Encode current image if needed
                if hasattr(current_image, 'read'):
                    current_image.seek(0)
                    current_encoded = encode_image_to_base64(current_image)
                else:
                    current_encoded = current_image
                
                human_message = self.create_human_message(
                    f"{comparison_prompt}\n\nCurrent image (newer):",
                    image_data=current_encoded
                )
            else:
                human_message = self.create_human_message(comparison_prompt)
            
            response = await self.call_llm([system_message, human_message])
            
            return {
                "comparison_analysis": response,
                "images_compared": len(historical_images) + (1 if current_image else 0),
                "time_span": self._calculate_time_span(historical_images),
                "analyzed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing visual changes: {e}")
            return {"error": str(e)}
    
    def _calculate_time_span(self, historical_images: List[Dict[str, Any]]) -> str:
        """Calculate time span of historical images"""
        try:
            if not historical_images:
                return "0 days"
            
            oldest = min(historical_images, key=lambda x: x["timestamp"])
            newest = max(historical_images, key=lambda x: x["timestamp"])
            
            oldest_date = datetime.fromisoformat(oldest["timestamp"])
            newest_date = datetime.fromisoformat(newest["timestamp"])
            
            days = (newest_date - oldest_date).days
            return f"{days} days"
            
        except Exception as e:
            self.logger.error(f"Error calculating time span: {e}")
            return "unknown"
    
    def _extract_visual_growth_indicators(self, visual_analysis: Dict[str, Any]) -> List[str]:
        """Extract growth indicators from visual analysis"""
        indicators = []
        
        try:
            analysis_text = visual_analysis.get("comparison_analysis", "").lower()
            
            # Look for growth indicators
            if "larger" in analysis_text or "bigger" in analysis_text:
                indicators.append("Overall size increase")
            
            if "new leaf" in analysis_text or "more leaves" in analysis_text:
                indicators.append("New leaf development")
            
            if "taller" in analysis_text or "height" in analysis_text:
                indicators.append("Height increase")
            
            if "flower" in analysis_text or "bud" in analysis_text:
                indicators.append("Flowering development")
            
            if "greener" in analysis_text or "healthier" in analysis_text:
                indicators.append("Improved health appearance")
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error extracting visual growth indicators: {e}")
            return []
    
    def _generate_visual_recommendations(self, visual_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on visual analysis"""
        recommendations = []
        
        try:
            analysis_text = visual_analysis.get("comparison_analysis", "").lower()
            
            if "yellow" in analysis_text or "brown" in analysis_text:
                recommendations.append("Monitor for potential stress or nutrient deficiency")
            
            if "wilting" in analysis_text or "drooping" in analysis_text:
                recommendations.append("Check watering schedule and soil moisture")
            
            if "pest" in analysis_text or "damage" in analysis_text:
                recommendations.append("Inspect for pests and consider treatment if needed")
            
            if "healthy" in analysis_text and "growth" in analysis_text:
                recommendations.append("Continue current care routine - plant showing healthy development")
            
            if not recommendations:
                recommendations.append("Continue monitoring plant development")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating visual recommendations: {e}")
            return ["Unable to generate recommendations"]
    
    # Additional helper methods for comprehensive reporting
    def _generate_growth_summary(self, growth_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate growth summary"""
        try:
            if not growth_data:
                return {"status": "no_data"}
            
            first_measurement = growth_data[0]
            last_measurement = growth_data[-1]
            
            summary = {
                "total_measurements": len(growth_data),
                "tracking_period": self._calculate_tracking_period(growth_data),
                "initial_state": first_measurement.get("measurements", {}),
                "current_state": last_measurement.get("measurements", {}),
                "overall_progress": self._calculate_overall_progress(growth_data)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating growth summary: {e}")
            return {"error": str(e)}
    
    def _calculate_tracking_period(self, growth_data: List[Dict[str, Any]]) -> str:
        """Calculate total tracking period"""
        try:
            if len(growth_data) < 2:
                return "0 days"
            
            first_date = datetime.fromisoformat(growth_data[0]["timestamp"])
            last_date = datetime.fromisoformat(growth_data[-1]["timestamp"])
            
            days = (last_date - first_date).days
            return f"{days} days"
            
        except Exception as e:
            return "unknown"
    
    def _calculate_overall_progress(self, growth_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall progress metrics"""
        try:
            if len(growth_data) < 2:
                return {"status": "insufficient_data"}
            
            first = growth_data[0].get("measurements", {})
            last = growth_data[-1].get("measurements", {})
            
            progress = {}
            
            # Height progress
            if first.get("height") and last.get("height"):
                height_change = last["height"] - first["height"]
                height_percent = (height_change / first["height"]) * 100 if first["height"] > 0 else 0
                progress["height"] = {
                    "change": height_change,
                    "percent_change": height_percent
                }
            
            # Leaf count progress
            if first.get("leaf_count") and last.get("leaf_count"):
                leaf_change = last["leaf_count"] - first["leaf_count"]
                leaf_percent = (leaf_change / first["leaf_count"]) * 100 if first["leaf_count"] > 0 else 0
                progress["leaf_count"] = {
                    "change": leaf_change,
                    "percent_change": leaf_percent
                }
            
            return progress
            
        except Exception as e:
            self.logger.error(f"Error calculating overall progress: {e}")
            return {"error": str(e)}
    
    async def _generate_detailed_predictions(self, growth_data: List[Dict[str, Any]], prediction_period: int) -> Dict[str, Any]:
        """Generate detailed growth predictions"""
        try:
            system_message = self.create_system_message(
                "You are an expert plant growth analyst. Based on historical growth data, predict future growth patterns."
            )
            
            data_summary = self._create_prediction_data_summary(growth_data)
            
            human_message = self.create_human_message(
                f"Based on this growth data, predict the plant's development over the next {prediction_period} days:\n"
                f"1. Expected height growth\n"
                f"2. Leaf development\n"
                f"3. Overall health trajectory\n"
                f"4. Potential milestones\n"
                f"5. Care recommendations\n\n"
                f"Historical Data Summary:\n{data_summary}"
            )
            
            response = await self.call_llm([system_message, human_message])
            
            return {
                "prediction_text": response,
                "prediction_period": prediction_period,
                "based_on_measurements": len(growth_data),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating detailed predictions: {e}")
            return {"error": str(e)}
    
    def _create_prediction_data_summary(self, growth_data: List[Dict[str, Any]]) -> str:
        """Create summary of growth data for predictions"""
        try:
            if not growth_data:
                return "No growth data available"
            
            summary_parts = []
            
            # Basic stats
            summary_parts.append(f"Total measurements: {len(growth_data)}")
            summary_parts.append(f"Tracking period: {self._calculate_tracking_period(growth_data)}")
            
            # Growth trends
            trends = self._analyze_growth_trends(growth_data)
            if trends.get("height"):
                summary_parts.append(f"Height trend: {trends['height']['trend']}")
            if trends.get("leaf_count"):
                summary_parts.append(f"Leaf count trend: {trends['leaf_count']['trend']}")
            
            # Growth rates
            rates = self._calculate_growth_rate(growth_data)
            if rates.get("height_per_week"):
                summary_parts.append(f"Height growth rate: {rates['height_per_week']:.2f} cm/week")
            if rates.get("leaves_per_week"):
                summary_parts.append(f"Leaf growth rate: {rates['leaves_per_week']:.2f} leaves/week")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error creating prediction data summary: {e}")
            return "Error creating data summary"
    
    def _calculate_prediction_confidence(self, growth_data: List[Dict[str, Any]]) -> str:
        """Calculate confidence level for predictions"""
        try:
            data_points = len(growth_data)
            
            if data_points >= 10:
                return "high"
            elif data_points >= 5:
                return "medium"
            elif data_points >= 3:
                return "low"
            else:
                return "very_low"
                
        except Exception:
            return "unknown"
    
    def _generate_prediction_recommendations(self, predictions: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on predictions"""
        recommendations = [
            "Continue regular monitoring and measurements",
            "Maintain consistent care routine",
            "Document any changes in environmental conditions",
            "Take photos regularly for visual progress tracking"
        ]
        
        return recommendations
    
    def _prepare_chart_data(self, growth_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare data for charts and visualizations"""
        try:
            chart_data = {
                "height_over_time": [],
                "leaf_count_over_time": [],
                "measurement_dates": []
            }
            
            for measurement in growth_data:
                date = measurement["timestamp"][:10]  # Extract date part
                chart_data["measurement_dates"].append(date)
                
                height = measurement.get("measurements", {}).get("height")
                chart_data["height_over_time"].append(height if height else None)
                
                leaf_count = measurement.get("measurements", {}).get("leaf_count")
                chart_data["leaf_count_over_time"].append(leaf_count if leaf_count else None)
            
            return chart_data
            
        except Exception as e:
            self.logger.error(f"Error preparing chart data: {e}")
            return {"error": str(e)}