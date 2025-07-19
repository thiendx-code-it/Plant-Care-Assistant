import asyncio
from typing import Dict, Any
from .base_agent import BaseAgent
from utils.api_helpers import APIHelper
from config.settings import settings

class PlantIdentifierAgent(BaseAgent):
    """Agent responsible for identifying plant species from images"""
    
    def __init__(self):
        super().__init__(
            name="Plant Identifier Agent",
            description="Uses Plant.id API v3 to detect plant species from images or descriptions."
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify plant from image"""
        try:
            # Validate input
            if not self.validate_input(input_data, ["image_base64"]):
                return self.create_error_response("Missing required field: image_base64")
            
            image_base64 = input_data["image_base64"]
            
            # Call Plant.id API
            self.logger.info("Calling Plant.id API for plant identification")
            api_response = await APIHelper.call_plant_id_api(image_base64)
            
            # Process response
            result = self._process_identification_response(api_response)
            
            return self.create_success_response(result)
            
        except Exception as e:
            self.logger.error(f"Error in plant identification: {e}")
            return self.create_error_response(str(e), "PLANT_ID_ERROR")
    
    def _process_identification_response(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """Process Plant.id API v3 response and extract relevant information including health data"""
        try:
            # API v3 response structure
            result_data = api_response.get("result", {})
            
            # Check if it's a plant
            is_plant = result_data.get("is_plant", {}).get("binary", False)
            if not is_plant:
                return {
                    "identified": False,
                    "confidence": 0.0,
                    "message": "Image does not appear to contain a plant"
                }
            
            # Get classification suggestions
            classification = result_data.get("classification", {})
            suggestions = classification.get("suggestions", [])
            
            if not suggestions:
                return {
                    "identified": False,
                    "confidence": 0.0,
                    "message": "No plant species identified"
                }
            
            # Get the top suggestion
            top_suggestion = suggestions[0]
            confidence = top_suggestion.get("probability", 0.0)
            
            # Check if confidence meets minimum threshold
            if confidence < settings.MIN_PLANT_ID_CONFIDENCE:
                return {
                    "identified": False,
                    "confidence": confidence,
                    "message": f"Plant identification confidence ({confidence:.2%}) below threshold ({settings.MIN_PLANT_ID_CONFIDENCE:.2%})",
                    "suggestions": self._format_suggestions_v3(suggestions[:3])  # Top 3 suggestions
                }
            
            # Extract plant details from v3 format
            details = top_suggestion.get("details", {})
            
            # Process health assessment data
            health_data = self._process_health_assessment(result_data)
            
            result = {
                "identified": True,
                "confidence": confidence,
                "plant_name": top_suggestion.get("name", "Unknown"),
                "scientific_name": top_suggestion.get("name", ""),
                "common_names": details.get("common_names", []),
                "family": self._extract_family_name_v3(details.get("taxonomy", {})),
                "description": details.get("description", {}).get("value", ""),
                "url": details.get("url", ""),
                "gbif_id": details.get("gbif_id"),
                "similar_images": self._extract_similar_images_v3(top_suggestion),
                "all_suggestions": self._format_suggestions_v3(suggestions[:5]),  # Top 5 for reference
                "health_assessment": health_data  # Add health assessment data
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing identification response: {e}")
            raise
    
    def _extract_family_name_v3(self, taxonomy: Dict[str, Any]) -> str:
        """Extract family name from taxonomy data (API v3 format)"""
        try:
            if isinstance(taxonomy, dict):
                return taxonomy.get("family", "")
            return ""
        except Exception:
            return ""
    
    def _extract_similar_images_v3(self, suggestion: Dict[str, Any]) -> list:
        """Extract similar images from suggestion (API v3 format)"""
        try:
            similar_images = suggestion.get("similar_images", [])
            return [{
                "url": img.get("url", ""),
                "license_name": img.get("license_name", ""),
                "license_url": img.get("license_url", ""),
                "citation": img.get("citation", "")
            } for img in similar_images[:3]]  # Limit to 3 images
        except Exception:
            return []
    
    def _format_suggestions_v3(self, suggestions: list) -> list:
        """Format suggestions for display (API v3 format)"""
        formatted = []
        for suggestion in suggestions:
            try:
                details = suggestion.get("details", {})
                formatted.append({
                    "plant_name": suggestion.get("name", "Unknown"),
                    "scientific_name": suggestion.get("name", ""),
                    "probability": suggestion.get("probability", 0.0),
                    "common_names": details.get("common_names", [])[:3]  # Limit to 3 names
                })
            except Exception:
                continue
        return formatted
    
    def _process_health_assessment(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process health assessment data from Plant.id API v3 response"""
        try:
            health_data = {
                "is_healthy": True,
                "health_confidence": 1.0,
                "diseases": [],
                "health_suggestions": []
            }
            
            # Check if health assessment data is present
            is_healthy_data = result_data.get("is_healthy", {})
            if is_healthy_data:
                health_data["is_healthy"] = is_healthy_data.get("binary", True)
                health_data["health_confidence"] = is_healthy_data.get("probability", 1.0)
            
            # Process disease data if present
            disease_data = result_data.get("disease", {})
            if disease_data:
                disease_suggestions = disease_data.get("suggestions", [])
                
                for disease in disease_suggestions[:3]:  # Top 3 diseases
                    disease_info = {
                        "name": disease.get("name", "Unknown Disease"),
                        "probability": disease.get("probability", 0.0),
                        "description": "",
                        "treatment": ""
                    }
                    
                    # Extract disease details if available
                    disease_details = disease.get("details", {})
                    if disease_details:
                        disease_info["description"] = disease_details.get("description", {}).get("value", "")
                        disease_info["treatment"] = disease_details.get("treatment", {}).get("value", "")
                    
                    health_data["diseases"].append(disease_info)
                
                # If diseases found, mark as unhealthy
                if disease_suggestions and disease_suggestions[0].get("probability", 0) > 0.3:
                    health_data["is_healthy"] = False
            
            return health_data
            
        except Exception as e:
            self.logger.error(f"Error processing health assessment: {e}")
            return {
                "is_healthy": True,
                "health_confidence": 0.0,
                "diseases": [],
                "health_suggestions": [],
                "error": str(e)
            }
    
    async def identify_by_description(self, description: str) -> Dict[str, Any]:
        """Identify plant by text description using LLM"""
        try:
            system_prompt = """
            You are a plant identification expert. Based on the user's description, 
            identify the most likely plant species. Provide:
            1. Scientific name
            2. Common names
            3. Plant family
            4. Confidence level (0-1)
            5. Key identifying characteristics mentioned
            
            Format your response as JSON.
            """
            
            human_message = f"Please identify this plant based on the description: {description}"
            
            messages = [
                self.create_system_message(system_prompt),
                self.create_human_message(human_message)
            ]
            
            response = await self.call_llm(messages)
            
            # Parse LLM response (simplified - in production, would need better parsing)
            return {
                "identified": True,
                "method": "description",
                "llm_response": response,
                "confidence": 0.6  # Lower confidence for description-based identification
            }
            
        except Exception as e:
            self.logger.error(f"Error in description-based identification: {e}")
            return self.create_error_response(str(e), "DESCRIPTION_ID_ERROR")