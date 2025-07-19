import asyncio
from typing import Dict, Any, List
from .base_agent import BaseAgent
from utils.api_helpers import APIHelper

class DiseaseDetectorAgent(BaseAgent):
    """Agent responsible for analyzing plant health and detecting diseases"""
    
    def __init__(self):
        super().__init__(
            name="Disease Detector Agent",
            description="Analyzes uploaded image for plant disease symptoms."
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze plant health from image"""
        try:
            # Validate input
            if not self.validate_input(input_data, ["image_base64"]):
                return self.create_error_response("Missing required field: image_base64")
            
            image_base64 = input_data["image_base64"]
            plant_name = input_data.get("plant_name", "Unknown plant")
            
            # Call Plant.id Health Assessment API
            self.logger.info("Calling Plant.id Health Assessment API")
            api_response = await APIHelper.call_plant_health_api(image_base64)
            
            # Process response
            result = self._process_health_response(api_response, plant_name)
            
            return self.create_success_response(result)
            
        except Exception as e:
            self.logger.error(f"Error in disease detection: {e}")
            return self.create_error_response(str(e), "DISEASE_DETECTION_ERROR")
    
    def _process_health_response(self, api_response: Dict[str, Any], plant_name: str) -> Dict[str, Any]:
        """Process Plant.id health assessment response (API v3 format)"""
        try:
            # API v3 response structure
            result_data = api_response.get("result", {})
            
            # Check if it's a plant
            is_plant = result_data.get("is_plant", {}).get("binary", False)
            if not is_plant:
                return {
                    "plant_name": plant_name,
                    "is_healthy": False,
                    "health_score": 0.0,
                    "diseases": [],
                    "pests": [],
                    "recommendations": ["Image does not appear to contain a plant"],
                    "severity_level": "Unknown"
                }
            
            # Extract health assessment from v3 format
            health_assessment = result_data.get("health_assessment", {})
            
            # Extract overall health status
            is_healthy_prob = health_assessment.get("is_healthy", {}).get("probability", 1.0)
            is_healthy = is_healthy_prob > 0.5
            
            # Extract diseases from v3 format
            diseases = health_assessment.get("diseases", [])
            
            # Extract pests from v3 format (if available)
            pests = health_assessment.get("pests", [])
            
            health_score = self._calculate_health_score_v3(diseases, pests, is_healthy_prob)
            formatted_diseases = self._format_diseases_v3(diseases)
            formatted_pests = self._format_pests_v3(pests)
            
            result = {
                "plant_name": plant_name,
                "is_healthy": is_healthy,
                "health_score": health_score,
                "diseases": formatted_diseases,
                "pests": formatted_pests,
                "recommendations": self._generate_health_recommendations(formatted_diseases, formatted_pests, is_healthy),
                "severity_level": self._determine_severity_level(formatted_diseases, formatted_pests)
            }
            
            # Add tree-specific analysis if this is a tree
            if self._is_tree_species(plant_name):
                result["tree_prognosis"] = self._generate_tree_prognosis(formatted_diseases, formatted_pests, health_score, plant_name)
                result["disease_likelihood_assessment"] = self._calculate_disease_likelihood(formatted_diseases, formatted_pests, plant_name)
                result["is_tree"] = True
            else:
                result["is_tree"] = False
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing health response: {e}")
            raise
    
    def _calculate_health_score_v3(self, diseases: List[Dict], pests: List[Dict], is_healthy_prob: float) -> float:
        """Calculate overall health score (0-1, where 1 is perfectly healthy) for API v3"""
        try:
            if not diseases and not pests:
                return round(is_healthy_prob, 2)
            
            # Calculate disease impact
            disease_impact = 0.0
            for disease in diseases:
                probability = disease.get("probability", 0.0)
                disease_impact += probability * 0.7  # Diseases have higher impact
            
            # Calculate pest impact
            pest_impact = 0.0
            for pest in pests:
                probability = pest.get("probability", 0.0)
                pest_impact += probability * 0.5  # Pests have moderate impact
            
            # Total impact (capped at 1.0)
            total_impact = min(1.0, disease_impact + pest_impact)
            
            # Health score combines is_healthy probability with impact assessment
            health_score = max(0.0, min(is_healthy_prob, 1.0 - total_impact))
            
            return round(health_score, 2)
            
        except Exception:
            return 0.5  # Default moderate health score
    
    def _format_diseases_v3(self, diseases: List[Dict]) -> List[Dict]:
        """Format disease information for display (API v3 format)"""
        formatted_diseases = []
        
        for disease in diseases:
            try:
                # API v3 format uses 'details' instead of 'disease_details'
                disease_details = disease.get("details", {})
                
                formatted_disease = {
                    "name": disease.get("name", "Unknown disease"),
                    "probability": disease.get("probability", 0.0),
                    "common_names": disease_details.get("common_names", []),
                    "description": disease_details.get("description", {}).get("value", ""),
                    "url": disease_details.get("url", ""),
                    "treatment": self._extract_treatment_info(disease_details)
                }
                
                formatted_diseases.append(formatted_disease)
                
            except Exception as e:
                self.logger.warning(f"Error formatting disease: {e}")
                continue
        
        # Sort by probability (highest first)
        formatted_diseases.sort(key=lambda x: x["probability"], reverse=True)
        
        return formatted_diseases
    
    def _format_pests_v3(self, pests: List[Dict]) -> List[Dict]:
        """Format pest information for display (API v3 format)"""
        formatted_pests = []
        
        for pest in pests:
            try:
                # API v3 format uses 'details' instead of 'pest_details'
                pest_details = pest.get("details", {})
                
                formatted_pest = {
                    "name": pest.get("name", "Unknown pest"),
                    "probability": pest.get("probability", 0.0),
                    "common_names": pest_details.get("common_names", []),
                    "description": pest_details.get("description", {}).get("value", ""),
                    "url": pest_details.get("url", ""),
                    "treatment": self._extract_treatment_info(pest_details)
                }
                
                formatted_pests.append(formatted_pest)
                
            except Exception as e:
                self.logger.warning(f"Error formatting pest: {e}")
                continue
        
        # Sort by probability (highest first)
        formatted_pests.sort(key=lambda x: x["probability"], reverse=True)
        
        return formatted_pests
    
    def _extract_treatment_info(self, details: Dict[str, Any]) -> str:
        """Extract treatment information from disease/pest details"""
        try:
            # Look for treatment in description or other fields
            description = details.get("description", {}).get("value", "")
            
            # Simple extraction - in production, would use more sophisticated NLP
            if "treatment" in description.lower():
                return description
            
            return "Consult with a plant specialist for specific treatment recommendations."
            
        except Exception:
            return "Treatment information not available."
    
    def _generate_health_recommendations(self, diseases: List[Dict], pests: List[Dict], is_healthy: bool) -> List[str]:
        """Generate health recommendations based on detected issues"""
        recommendations = []
        
        if is_healthy and not diseases and not pests:
            recommendations.extend([
                "Your plant appears healthy! Continue with regular care.",
                "Monitor regularly for any changes in appearance.",
                "Maintain consistent watering and lighting conditions."
            ])
        else:
            if diseases:
                recommendations.extend([
                    "Disease detected - isolate plant from other plants if possible.",
                    "Remove affected leaves or parts if safe to do so.",
                    "Improve air circulation around the plant.",
                    "Avoid overhead watering to prevent spread."
                ])
            
            if pests:
                recommendations.extend([
                    "Pest activity detected - inspect plant thoroughly.",
                    "Consider using insecticidal soap or neem oil.",
                    "Check surrounding plants for similar issues.",
                    "Quarantine if infestation is severe."
                ])
            
            recommendations.append("Consider consulting with a local plant expert or extension service.")
        
        return recommendations
    
    def _generate_tree_prognosis(self, diseases: List[Dict], pests: List[Dict], health_score: float, plant_name: str) -> Dict[str, Any]:
        """Generate detailed tree prognosis for tree species"""
        try:
            prognosis = {
                "overall_health_outlook": "",
                "expected_lifespan": "",
                "recovery_potential": "",
                "risk_factors": [],
                "monitoring_frequency": ""
            }
            
            # Determine overall health outlook
            if health_score >= 0.8:
                prognosis["overall_health_outlook"] = "Excellent - Tree shows strong vitality with minimal health concerns"
                prognosis["expected_lifespan"] = "Normal species lifespan expected with proper care"
                prognosis["recovery_potential"] = "N/A - Tree is currently healthy"
                prognosis["monitoring_frequency"] = "Annual professional inspection recommended"
            elif health_score >= 0.6:
                prognosis["overall_health_outlook"] = "Good - Minor issues present but manageable with proper care"
                prognosis["expected_lifespan"] = "Near-normal lifespan with appropriate treatment"
                prognosis["recovery_potential"] = "High - Early intervention should resolve current issues"
                prognosis["monitoring_frequency"] = "Bi-annual monitoring recommended"
            elif health_score >= 0.4:
                prognosis["overall_health_outlook"] = "Fair - Moderate health concerns requiring active management"
                prognosis["expected_lifespan"] = "Potentially reduced lifespan without intervention"
                prognosis["recovery_potential"] = "Moderate - Recovery possible with comprehensive treatment"
                prognosis["monitoring_frequency"] = "Quarterly professional assessment needed"
            else:
                prognosis["overall_health_outlook"] = "Poor - Significant health issues requiring immediate attention"
                prognosis["expected_lifespan"] = "Severely compromised without aggressive treatment"
                prognosis["recovery_potential"] = "Low - Extensive intervention required, outcome uncertain"
                prognosis["monitoring_frequency"] = "Monthly monitoring essential"
            
            # Identify risk factors
            if diseases:
                for disease in diseases:
                    if disease.get("probability", 0) > 0.3:
                        prognosis["risk_factors"].append(f"Disease: {disease.get('name', 'Unknown')} ({disease.get('probability', 0):.1%} likelihood)")
            
            if pests:
                for pest in pests:
                    if pest.get("probability", 0) > 0.3:
                        prognosis["risk_factors"].append(f"Pest: {pest.get('name', 'Unknown')} ({pest.get('probability', 0):.1%} likelihood)")
            
            return prognosis
            
        except Exception as e:
            self.logger.error(f"Error generating tree prognosis: {e}")
            return {
                "overall_health_outlook": "Unable to assess - insufficient data",
                "expected_lifespan": "Cannot determine",
                "recovery_potential": "Cannot assess",
                "risk_factors": [],
                "monitoring_frequency": "Consult arborist for professional assessment"
            }
    
    def _is_tree_species(self, plant_name: str) -> bool:
        """Determine if the identified plant is a tree species"""
        if not plant_name:
            return False
            
        plant_name_lower = plant_name.lower()
        
        # Common tree indicators
        tree_keywords = [
            'tree', 'oak', 'maple', 'pine', 'birch', 'cedar', 'elm', 'ash', 'willow',
            'cherry', 'apple', 'pear', 'plum', 'peach', 'citrus', 'lemon', 'orange',
            'palm', 'eucalyptus', 'magnolia', 'dogwood', 'redwood', 'sequoia',
            'spruce', 'fir', 'cypress', 'juniper', 'poplar', 'sycamore', 'hickory',
            'walnut', 'chestnut', 'beech', 'linden', 'basswood', 'tulip tree',
            'ginkgo', 'mimosa', 'acacia', 'jacaranda', 'baobab', 'banyan'
        ]
        
        return any(keyword in plant_name_lower for keyword in tree_keywords)
    
    def _calculate_disease_likelihood(self, diseases: List[Dict], pests: List[Dict], plant_name: str) -> Dict[str, Any]:
        """Calculate detailed disease likelihood assessment"""
        try:
            assessment = {
                "current_disease_probability": 0.0,
                "future_risk_assessment": "",
                "primary_concerns": [],
                "risk_factors": [],
                "confidence_level": "Medium"
            }
            
            # Calculate current disease probability
            if diseases:
                max_disease_prob = max(disease.get("probability", 0) for disease in diseases)
                assessment["current_disease_probability"] = max_disease_prob
                
                # Add primary concerns
                for disease in diseases:
                    if disease.get("probability", 0) > 0.2:
                        assessment["primary_concerns"].append({
                            "disease": disease.get("name", "Unknown"),
                            "probability": disease.get("probability", 0),
                            "severity": "High" if disease.get("probability", 0) > 0.7 else "Moderate" if disease.get("probability", 0) > 0.4 else "Low"
                        })
            
            # Add pest-related risks
            if pests:
                for pest in pests:
                    if pest.get("probability", 0) > 0.2:
                        assessment["risk_factors"].append(f"Pest pressure from {pest.get('name', 'Unknown')} ({pest.get('probability', 0):.1%})")
            
            # Determine future risk assessment
            total_risk = assessment["current_disease_probability"]
            if pests:
                total_risk += max(pest.get("probability", 0) for pest in pests) * 0.5
            
            if total_risk >= 0.7:
                assessment["future_risk_assessment"] = "High risk - Immediate intervention required to prevent spread"
                assessment["confidence_level"] = "High"
            elif total_risk >= 0.4:
                assessment["future_risk_assessment"] = "Moderate risk - Preventive measures recommended"
                assessment["confidence_level"] = "Medium"
            elif total_risk >= 0.2:
                assessment["future_risk_assessment"] = "Low to moderate risk - Monitor closely"
                assessment["confidence_level"] = "Medium"
            else:
                assessment["future_risk_assessment"] = "Low risk - Continue regular monitoring"
                assessment["confidence_level"] = "High"
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error calculating disease likelihood: {e}")
            return {
                "current_disease_probability": 0.0,
                "future_risk_assessment": "Unable to assess",
                "primary_concerns": [],
                "risk_factors": [],
                "confidence_level": "Low"
            }
    
    def _identify_risk_factors(self, diseases, pests):
        """
        Identify environmental and biological risk factors
        """
        risk_factors = []
        
        # Disease-based risk factors
        for disease in diseases:
            if disease.get('probability', 0) > 0.3:
                if 'fungal' in disease.get('name', '').lower():
                    risk_factors.append("High humidity conditions")
                elif 'bacterial' in disease.get('name', '').lower():
                    risk_factors.append("Wound entry points")
                elif 'viral' in disease.get('name', '').lower():
                    risk_factors.append("Insect vector transmission")
        
        # Pest-based risk factors
        for pest in pests:
            if pest.get('probability', 0) > 0.3:
                risk_factors.append(f"Presence of {pest.get('name', 'unknown pest')}")
        
        # Remove duplicates and limit
        return list(set(risk_factors))[:5]
    
    def _suggest_preventive_measures(self, current_risk, primary_concerns):
        """
        Suggest preventive measures based on risk level and concerns
        """
        measures = []
        
        if current_risk == "High":
            measures.extend([
                "Immediate professional arborist consultation recommended",
                "Apply targeted treatment for identified diseases/pests",
                "Improve air circulation around the tree",
                "Monitor daily for symptom progression"
            ])
        elif current_risk == "Medium":
            measures.extend([
                "Regular monitoring every 2-3 days",
                "Maintain proper watering schedule",
                "Ensure adequate nutrition",
                "Remove affected plant material"
            ])
        else:
            measures.extend([
                "Continue regular care routine",
                "Weekly health inspections",
                "Maintain optimal growing conditions",
                "Preventive treatments during high-risk seasons"
            ])
        
        # Add specific measures based on concerns
        if any('fungal' in str(concern).lower() for concern in primary_concerns):
            measures.append("Reduce humidity and improve drainage")
        if any('pest' in str(concern).lower() for concern in primary_concerns):
            measures.append("Apply appropriate pest control measures")
        
        return measures[:6]  # Limit to 6 measures
    
    def _determine_severity_level(self, diseases: List[Dict], pests: List[Dict]) -> str:
        """Determine severity level of health issues"""
        try:
            max_disease_prob = max([d.get("probability", 0.0) for d in diseases], default=0.0)
            max_pest_prob = max([p.get("probability", 0.0) for p in pests], default=0.0)
            
            max_prob = max(max_disease_prob, max_pest_prob)
            
            if max_prob >= 0.8:
                return "Critical"
            elif max_prob >= 0.6:
                return "High"
            elif max_prob >= 0.3:
                return "Moderate"
            elif max_prob > 0.0:
                return "Low"
            else:
                return "Healthy"
                
        except Exception:
            return "Unknown"
    
    async def analyze_symptoms_by_description(self, symptoms_description: str, plant_name: str = "Unknown") -> Dict[str, Any]:
        """Analyze plant health based on text description of symptoms"""
        try:
            system_prompt = """
            You are a plant pathology expert. Based on the described symptoms, 
            analyze the plant's health and provide:
            1. Possible diseases or pests
            2. Severity assessment
            3. Treatment recommendations
            4. Prevention measures
            
            Format your response as a structured analysis.
            """
            
            human_message = f"Plant: {plant_name}\nSymptoms: {symptoms_description}\n\nPlease analyze these symptoms and provide a health assessment."
            
            messages = [
                self.create_system_message(system_prompt),
                self.create_human_message(human_message)
            ]
            
            response = await self.call_llm(messages)
            
            return {
                "method": "description_analysis",
                "plant_name": plant_name,
                "symptoms_described": symptoms_description,
                "analysis": response,
                "confidence": 0.6  # Lower confidence for description-based analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error in symptom analysis: {e}")
            return self.create_error_response(str(e), "SYMPTOM_ANALYSIS_ERROR")