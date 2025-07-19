import asyncio
from typing import Dict, Any, List
from .base_agent import BaseAgent
from utils.vector_db import VectorDBManager
from langchain_core.documents import Document

class CareAdvisorAgent(BaseAgent):
    """Agent responsible for providing plant care advice using RAG"""
    
    def __init__(self):
        super().__init__(
            name="Care Advisor Agent",
            description="Uses LangChain + RAG to retrieve care instructions from Pinecone."
        )
        self.vector_db = VectorDBManager()
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide care advice for identified plant"""
        try:
            # Validate input
            if not self.validate_input(input_data, ["plant_name"]):
                return self.create_error_response("Missing required field: plant_name")
            
            plant_name = input_data["plant_name"]
            specific_query = input_data.get("specific_query", "")
            health_issues = input_data.get("health_issues", [])
            weather_data = input_data.get("weather_data", {})
            image_base64 = input_data.get("image_base64", "")
            image_description = input_data.get("image_description", "")
            
            # Get pre-retrieved knowledge and web search results from workflow
            knowledge_results = input_data.get("knowledge_results", [])
            web_search_results = input_data.get("web_search_results", [])
            plant_identification = input_data.get("plant_identification", {})
            
            # Use provided knowledge results or retrieve from vector database
            if knowledge_results:
                self.logger.info(f"Using pre-retrieved knowledge results for {plant_name}")
                care_docs = knowledge_results
            else:
                self.logger.info(f"Retrieving care information for {plant_name}")
                care_docs = self.vector_db.get_plant_care_info(plant_name, specific_query)
            
            # Analyze image if provided for visual context
            visual_context = ""
            if image_base64:
                visual_context = await self._analyze_plant_image(image_base64, plant_name, image_description)
            
            # Generate comprehensive care advice with all available data
            care_advice = await self._generate_care_advice(
                plant_name, care_docs, specific_query, health_issues, weather_data, 
                visual_context, web_search_results, plant_identification
            )
            
            # Calculate total sources with None checks
            care_docs_count = len(care_docs) if care_docs else 0
            web_search_count = len(web_search_results) if web_search_results else 0
            health_issues_count = len(health_issues) if health_issues else 0
            total_sources = care_docs_count + web_search_count
            
            result = {
                "plant_name": plant_name,
                "care_advice": care_advice,
                "sources_found": total_sources,
                "knowledge_base_sources": care_docs_count,
                "web_search_sources": web_search_count,
                "rag_quality": self._assess_rag_quality(care_docs),
                "needs_web_search": care_docs_count == 0 or self._assess_rag_quality(care_docs) < 0.7,
                "visual_analysis_included": bool(image_base64),
                "plant_identified": bool(plant_identification),
                "health_issues_detected": health_issues_count > 0
            }
            
            return self.create_success_response(result)
            
        except Exception as e:
            self.logger.error(f"Error in care advice generation: {e}")
            return self.create_error_response(str(e), "CARE_ADVICE_ERROR")
    
    async def _analyze_plant_image(self, image_base64: str, plant_name: str, image_description: str = "") -> str:
        """Analyze plant image for visual health assessment"""
        try:
            system_prompt = f"""
You are an expert plant health diagnostician. Analyze the provided image of {plant_name} and provide a detailed visual assessment.

Focus on:
1. Overall plant health and vigor
2. Leaf condition (color, texture, spots, yellowing, browning)
3. Signs of pests or diseases
4. Growth patterns and structure
5. Soil condition (if visible)
6. Environmental stress indicators
7. Watering status indicators

Provide a concise but thorough visual assessment that will help inform care recommendations.
"""
            
            user_prompt = f"Analyze this image of {plant_name} for health assessment."
            if image_description:
                user_prompt += f" Additional context: {image_description}"
            
            # Get visual analysis from LLM
            messages = [
                self.create_system_message(system_prompt),
                self.create_human_message(user_prompt, image_base64=image_base64)
            ]
            
            response = await self.call_llm(messages, max_tokens=500)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error analyzing plant image: {e}")
            return "Visual analysis unavailable due to processing error."
    
    async def _generate_care_advice(self, plant_name: str, care_docs: List[Document], 
                                  specific_query: str, health_issues: List[Dict], 
                                  weather_data: Dict[str, Any], visual_context: str = "",
                                  web_search_results: List[Dict] = None, 
                                  plant_identification: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive care advice using LLM and retrieved documents"""
        try:
            # Initialize default values
            if web_search_results is None:
                web_search_results = []
            if plant_identification is None:
                plant_identification = {}
            
            # Prepare context from retrieved documents
            context = self._prepare_context_from_docs(care_docs)
            
            # Prepare web search context
            web_context = self._prepare_web_search_context(web_search_results)
            
            # Prepare plant identification context
            plant_id_context = self._prepare_plant_identification_context(plant_identification)
            
            # Prepare health context
            health_context = self._prepare_health_context(health_issues)
            
            # Prepare weather context
            weather_context = self._prepare_weather_context(weather_data)
            
            # Determine if this is a tree and adjust prompt accordingly
            is_tree = self._is_tree_species(plant_name, plant_identification)
            
            # Create system prompt with tree-specific requirements
            if is_tree:
                system_prompt = f"""
                You are a certified arborist and tree care specialist with expertise in tree health assessment, 
                disease diagnosis, and treatment planning. Your goal is to provide comprehensive tree care advice for {plant_name}.
                
                Based on the information I've gathered for you:
                
                🌳 **Tree Identification:**
                {plant_id_context}
                
                📚 **Knowledge Base Research:**
                {context}
                
                🌐 **Additional Web Research:**
                {web_context}
                
                🩺 **Health Assessment:**
                {health_context}
                
                🌤️ **Current Weather:**
                {weather_context}
                
                📸 **Visual Analysis:**
                {visual_context}
                
                **CRITICAL REQUIREMENTS - You MUST include these three sections:**
                
                🔮 **TREE PROGNOSIS** - Provide a detailed assessment of the tree's overall health outlook, 
                expected lifespan, recovery potential, and long-term viability. Include specific timeframes 
                and factors that could affect the prognosis.
                
                🦠 **DISEASE LIKELIHOOD ASSESSMENT** - Analyze and quantify the probability of current or 
                potential diseases. Provide percentage estimates where possible, identify risk factors, 
                and explain the reasoning behind your assessment.
                
                🏥 **COMPREHENSIVE TREATMENT PLAN** - Develop a detailed, step-by-step treatment protocol 
                including immediate actions, ongoing treatments, monitoring schedule, and preventive measures. 
                Include specific products, techniques, and timelines.
                
                Additionally, provide comprehensive care covering:
                
                💧 **Watering & Irrigation** - Deep watering schedules, soil moisture management
                ☀️ **Light & Location** - Optimal placement, sun exposure requirements
                🌱 **Soil & Nutrition** - Soil amendments, fertilization programs, root health
                🌡️ **Environmental Factors** - Temperature tolerance, humidity, seasonal considerations
                🚨 **Risk Management** - Structural integrity, safety concerns, monitoring protocols
                🎯 **Immediate Actions** - Urgent interventions needed based on current condition
                📅 **Long-term Care Plan** - Multi-year maintenance strategy, pruning schedules
                
                Use professional terminology while remaining accessible. Include specific recommendations 
                with scientific backing where appropriate.
                """
            else:
                system_prompt = f"""
                You are a friendly and knowledgeable plant care expert who loves helping people take better care of their plants. 
                Your goal is to provide warm, conversational, and practical advice for {plant_name}.
                
                Write your response as if you're talking to a friend who asked for plant care help. Use a warm, encouraging tone 
                and make the advice feel personal and approachable. Include emojis where appropriate to make it more engaging.
                
                Based on the information I've gathered for you:
                
                🌱 **Plant Identification:**
                {plant_id_context}
                
                📚 **Knowledge Base Research:**
                {context}
                
                🌐 **Additional Web Research:**
                {web_context}
                
                🩺 **Health Assessment:**
                {health_context}
                
                🌤️ **Current Weather:**
                {weather_context}
                
                📸 **Visual Analysis:**
                {visual_context}
                
                Please provide a comprehensive but friendly response that covers:
                
                💧 **Watering Care** - When and how to water, signs to watch for
                ☀️ **Light & Placement** - Best lighting conditions and where to place the plant
                🌱 **Soil & Nutrition** - Soil type, fertilizing schedule, and feeding tips
                🌡️ **Environment** - Temperature, humidity, and seasonal adjustments
                🚨 **Problem Prevention** - Common issues and how to avoid them
                🎯 **Immediate Actions** - Any urgent care needed based on current condition
                📅 **Ongoing Care** - Long-term maintenance and seasonal tips
                
                Start your response with a warm greeting and acknowledgment of their question. Use natural language, 
                personal pronouns (you, your), and practical examples. Make it feel like advice from a caring friend 
                who happens to be a plant expert.
                
                Format as natural, flowing text with clear sections using emojis as headers. Avoid JSON format - 
                write as a conversational response that feels human and caring.
                """
            
            # Create human message
            human_message = f"Please provide comprehensive care advice for {plant_name}."
            if specific_query:
                human_message += f" Specific question: {specific_query}"
            
            messages = [
                self.create_system_message(system_prompt),
                self.create_human_message(human_message)
            ]
            
            # Get LLM response
            llm_response = await self.call_llm(messages)
            
            # Parse and structure the response
            structured_advice = self._structure_care_advice(llm_response, plant_name)
            
            return structured_advice
            
        except Exception as e:
            self.logger.error(f"Error generating care advice: {e}")
            return self._get_fallback_care_advice(plant_name)
    
    def _prepare_context_from_docs(self, docs: List[Document]) -> str:
        """Prepare context string from retrieved documents"""
        if not docs:
            return "No specific care information found in knowledge base."
        
        context_parts = []
        for i, doc in enumerate(docs[:5]):  # Limit to top 5 documents
            if hasattr(doc, 'page_content'):
                context_parts.append(f"Source {i+1}: {doc.page_content}")
            elif isinstance(doc, dict) and 'content' in doc:
                context_parts.append(f"Source {i+1}: {doc['content']}")
            else:
                context_parts.append(f"Source {i+1}: {str(doc)}")
        
        return "\n\n".join(context_parts)
    
    def _prepare_web_search_context(self, web_results: List[Dict]) -> str:
        """Prepare context string from web search results"""
        if not web_results:
            return "No additional web research available."
        
        context_parts = []
        for i, result in enumerate(web_results[:3]):  # Limit to top 3 web results
            title = result.get('title', 'Unknown title')
            snippet = result.get('snippet', result.get('content', 'No content available'))
            url = result.get('url', 'No URL')
            context_parts.append(f"Web Source {i+1}: {title}\n{snippet}\nURL: {url}")
        
        return "\n\n".join(context_parts)
    
    def _prepare_plant_identification_context(self, plant_id: Dict[str, Any]) -> str:
        """Prepare context string from plant identification data"""
        if not plant_id:
            return "No plant identification data available."
        
        context_parts = []
        
        # Basic identification info
        plant_name = plant_id.get('plant_name', 'Unknown')
        scientific_name = plant_id.get('scientific_name', 'Unknown')
        confidence = plant_id.get('confidence', 0.0)
        
        context_parts.append(f"Identified Plant: {plant_name}")
        context_parts.append(f"Scientific Name: {scientific_name}")
        context_parts.append(f"Identification Confidence: {confidence:.1%}")
        
        # Plant details if available
        if 'plant_details' in plant_id:
            details = plant_id['plant_details']
            if 'common_names' in details:
                context_parts.append(f"Common Names: {', '.join(details['common_names'])}")
            if 'taxonomy' in details:
                taxonomy = details['taxonomy']
                context_parts.append(f"Family: {taxonomy.get('family', 'Unknown')}")
                context_parts.append(f"Genus: {taxonomy.get('genus', 'Unknown')}")
        
        # Health assessment if available
        if 'health_assessment' in plant_id:
            health = plant_id['health_assessment']
            if 'is_healthy' in health:
                health_status = "Healthy" if health['is_healthy'] else "Health issues detected"
                context_parts.append(f"Health Status: {health_status}")
        
        return "\n".join(context_parts)
    
    def _prepare_health_context(self, health_issues: List[Dict]) -> str:
        """Prepare health context from detected issues"""
        if not health_issues:
            return "Plant appears healthy with no detected issues."
        
        health_parts = []
        for issue in health_issues:
            issue_type = issue.get("type", "unknown")
            name = issue.get("name", "Unknown issue")
            probability = issue.get("probability", 0.0)
            health_parts.append(f"{issue_type.title()}: {name} (confidence: {probability:.1%})")
        
        return "Detected health issues: " + "; ".join(health_parts)
    
    def _prepare_weather_context(self, weather_data: Dict[str, Any]) -> str:
        """Prepare weather context for care adjustments"""
        if not weather_data:
            return "No current weather data available."
        
        try:
            temp = weather_data.get("main", {}).get("temp", "N/A")
            humidity = weather_data.get("main", {}).get("humidity", "N/A")
            weather_desc = weather_data.get("weather", [{}])[0].get("description", "N/A")
            
            return f"Current weather: {weather_desc}, Temperature: {temp}°C, Humidity: {humidity}%"
        except Exception:
            return "Weather data available but could not be parsed."
    
    def _is_tree_species(self, plant_name: str, plant_identification: Dict[str, Any]) -> bool:
        """Determine if the plant is a tree species"""
        try:
            # Check plant name for tree indicators
            tree_keywords = [
                'tree', 'oak', 'maple', 'pine', 'birch', 'cedar', 'fir', 'spruce', 
                'willow', 'elm', 'ash', 'cherry', 'apple', 'pear', 'citrus', 'palm',
                'eucalyptus', 'magnolia', 'dogwood', 'redwood', 'sequoia', 'cypress',
                'juniper', 'poplar', 'sycamore', 'hickory', 'walnut', 'pecan', 'chestnut'
            ]
            
            plant_name_lower = plant_name.lower()
            if any(keyword in plant_name_lower for keyword in tree_keywords):
                return True
            
            # Check plant identification data
            if plant_identification:
                # Check scientific name and taxonomy
                scientific_name = plant_identification.get('scientific_name', '').lower()
                if any(keyword in scientific_name for keyword in tree_keywords):
                    return True
                
                # Check plant details and taxonomy
                plant_details = plant_identification.get('plant_details', {})
                taxonomy = plant_details.get('taxonomy', {})
                
                # Common tree families
                tree_families = [
                    'fagaceae', 'pinaceae', 'rosaceae', 'salicaceae', 'betulaceae',
                    'aceraceae', 'oleaceae', 'cupressaceae', 'ulmaceae', 'juglandaceae'
                ]
                
                family = taxonomy.get('family', '').lower()
                if any(tree_family in family for tree_family in tree_families):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error determining if plant is tree: {e}")
            return False
    
    def _structure_care_advice(self, llm_response: str, plant_name: str) -> str:
        """Return the natural language response directly"""
        try:
            # Return the natural language response as-is
            # The LLM is already instructed to provide conversational, human-friendly text
            return llm_response
            
        except Exception as e:
            self.logger.warning(f"Error processing care advice: {e}")
            return f"I apologize, but I encountered an issue while processing the care advice for {plant_name}. Please try asking again."
    
    def _extract_section(self, text: str, keywords: List[str]) -> str:
        """Extract relevant section from text based on keywords"""
        try:
            lines = text.split('\n')
            relevant_lines = []
            
            for line in lines:
                if any(keyword.lower() in line.lower() for keyword in keywords):
                    relevant_lines.append(line.strip())
            
            return '\n'.join(relevant_lines) if relevant_lines else "No specific information found."
            
        except Exception:
            return "Could not extract section information."
    
    def _assess_rag_quality(self, docs: List[Document]) -> float:
        """Assess the quality of retrieved documents"""
        if not docs:
            return 0.0
        
        # Simple quality assessment based on number and length of documents
        quality_score = min(1.0, len(docs) / 3.0)  # Prefer 3+ documents
        
        # Adjust based on content length
        total_length = 0
        for doc in docs:
            if hasattr(doc, 'page_content'):
                total_length += len(doc.page_content)
            elif isinstance(doc, dict) and 'content' in doc:
                total_length += len(doc['content'])
            else:
                total_length += len(str(doc))
        
        avg_length = total_length / len(docs) if docs else 0
        if avg_length > 200:  # Prefer longer, more detailed content
            quality_score *= 1.2
        
        return min(1.0, quality_score)
    
    def _get_fallback_care_advice(self, plant_name: str) -> str:
        """Provide fallback care advice when LLM fails"""
        is_tree = self._is_tree_species(plant_name, {})
        
        if is_tree:
            return f"""I'd be happy to help with general care advice for your {plant_name}! 🌳

Since I don't have specific details about your tree's current condition, here's some general guidance:

💧 **Watering & Care**: Trees generally prefer deep, infrequent watering rather than frequent shallow watering. This encourages deep root growth and better drought tolerance.

🔍 **Health Monitoring**: Keep an eye out for any changes in leaf color, bark condition, or overall structure. Early detection of issues is key for tree health.

⚠️ **Professional Assessment**: For the most accurate prognosis, disease likelihood assessment, and treatment planning, I'd recommend consulting with a certified arborist who can evaluate your tree in person.

🌱 **General Maintenance**: Ensure good drainage around the tree, avoid soil compaction, and maintain proper mulching practices.

If you can share a photo or more specific details about any concerns you have, I'd be happy to provide more targeted advice!"""
        else:
            return f"""I'd love to help you care for your {plant_name}! 🌱

Here's some general care guidance to get you started:

💧 **Watering**: Most houseplants prefer to dry out slightly between waterings. Check the top 1-2 inches of soil - if it feels dry, it's usually time to water.

☀️ **Light**: Bright, indirect light works well for most plants. Avoid placing them in direct sunlight which can scorch the leaves.

🌱 **Soil & Nutrition**: Use well-draining potting mix and fertilize monthly during the growing season (spring and summer).

🌡️ **Environment**: Most plants thrive in temperatures between 65-75°F (18-24°C) with moderate humidity.

For more specific advice tailored to your plant's needs, feel free to share a photo or ask about any particular concerns you might have! I'm here to help. 😊"""
    
    async def get_specific_care_advice(self, plant_name: str, question: str) -> Dict[str, Any]:
        """Get advice for specific plant care question"""
        try:
            input_data = {
                "plant_name": plant_name,
                "specific_query": question
            }
            
            return await self.execute(input_data)
            
        except Exception as e:
            self.logger.error(f"Error getting specific care advice: {e}")
            return self.create_error_response(str(e), "SPECIFIC_ADVICE_ERROR")
    
    async def update_care_knowledge(self, plant_name: str, new_care_info: str, source: str = "user_input") -> bool:
        """Update the knowledge base with new care information"""
        try:
            success = self.vector_db.update_knowledge_base(plant_name, new_care_info, source)
            
            if success:
                self.logger.info(f"Updated care knowledge for {plant_name}")
            else:
                self.logger.warning(f"Failed to update care knowledge for {plant_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating care knowledge: {e}")
            return False