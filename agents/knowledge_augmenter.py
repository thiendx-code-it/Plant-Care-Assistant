import asyncio
from typing import Dict, Any, List
from .base_agent import BaseAgent
from utils.api_helpers import APIHelper
from utils.vector_db import VectorDBManager

class KnowledgeAugmenterAgent(BaseAgent):
    """Agent responsible for augmenting knowledge base using web search"""
    
    def __init__(self):
        super().__init__(
            name="Knowledge Augmenter Agent",
            description="Uses Tavily API to search the web for missing care information and updates Pinecone."
        )
        self.vector_db = VectorDBManager()
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Search web for plant care information and update knowledge base"""
        try:
            # Validate input
            if not self.validate_input(input_data, ["plant_name"]):
                return self.create_error_response("Missing required field: plant_name")
            
            plant_name = input_data["plant_name"]
            specific_query = input_data.get("specific_query", "")
            search_topics = input_data.get("search_topics", ["care", "watering", "light", "soil"])
            
            # Perform web searches for different care topics
            self.logger.info(f"Searching web for {plant_name} care information")
            search_results = await self._perform_comprehensive_search(plant_name, specific_query, search_topics)
            
            # Process and filter search results
            processed_info = await self._process_search_results(search_results, plant_name)
            
            # Update knowledge base with new information
            update_success = await self._update_knowledge_base(plant_name, processed_info)
            
            result = {
                "plant_name": plant_name,
                "searches_performed": len(search_results),
                "information_found": len(processed_info.get("care_sections", [])),
                "knowledge_base_updated": update_success,
                "search_summary": processed_info.get("summary", ""),
                "sources": processed_info.get("sources", []),
                "care_information": processed_info.get("care_sections", {})
            }
            
            return self.create_success_response(result)
            
        except Exception as e:
            self.logger.error(f"Error in knowledge augmentation: {e}")
            return self.create_error_response(str(e), "KNOWLEDGE_AUGMENTATION_ERROR")
    
    async def _perform_comprehensive_search(self, plant_name: str, specific_query: str, search_topics: List[str]) -> List[Dict[str, Any]]:
        """Perform multiple web searches for comprehensive plant information"""
        search_results = []
        
        try:
            # Base search queries
            base_queries = [
                f"{plant_name} care guide",
                f"{plant_name} growing conditions",
                f"how to care for {plant_name}"
            ]
            
            # Topic-specific queries
            topic_queries = []
            for topic in search_topics:
                topic_queries.append(f"{plant_name} {topic} requirements")
            
            # Specific query if provided
            if specific_query:
                topic_queries.append(f"{plant_name} {specific_query}")
            
            # Combine all queries
            all_queries = base_queries + topic_queries
            
            # Perform searches (limit to avoid rate limiting)
            for query in all_queries[:5]:  # Limit to 5 searches
                try:
                    self.logger.info(f"Searching: {query}")
                    result = await APIHelper.call_tavily_search_api(query, max_results=3)
                    search_results.append({
                        "query": query,
                        "results": result
                    })
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    self.logger.warning(f"Search failed for query '{query}': {e}")
                    continue
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error performing comprehensive search: {e}")
            return []
    
    async def _process_search_results(self, search_results: List[Dict[str, Any]], plant_name: str) -> Dict[str, Any]:
        """Process and synthesize search results into structured care information"""
        try:
            # Extract all content from search results
            all_content = []
            sources = []
            
            for search_result in search_results:
                query = search_result.get("query", "")
                results = search_result.get("results", {})
                
                # Extract results from Tavily response
                tavily_results = results.get("results", [])
                
                for result in tavily_results:
                    content = result.get("content", "")
                    url = result.get("url", "")
                    title = result.get("title", "")
                    
                    if content and len(content) > 100:  # Filter out very short content
                        all_content.append({
                            "content": content,
                            "source_url": url,
                            "title": title,
                            "query": query
                        })
                        
                        sources.append({
                            "url": url,
                            "title": title,
                            "query": query
                        })
            
            # Use LLM to synthesize information
            synthesized_info = await self._synthesize_care_information(all_content, plant_name)
            
            return {
                "care_sections": synthesized_info,
                "sources": sources,
                "summary": f"Found {len(all_content)} relevant sources for {plant_name} care information",
                "raw_content_count": len(all_content)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing search results: {e}")
            return {"error": "Failed to process search results"}
    
    async def _synthesize_care_information(self, content_list: List[Dict[str, Any]], plant_name: str) -> Dict[str, Any]:
        """Use LLM to synthesize web search results into structured care information"""
        try:
            if not content_list:
                return {"note": "No content to synthesize"}
            
            # Prepare content for LLM
            content_text = "\n\n".join([
                f"Source: {item['title']}\nURL: {item['source_url']}\nContent: {item['content'][:1000]}..."  # Limit content length
                for item in content_list[:10]  # Limit to 10 sources
            ])
            
            system_prompt = f"""
            You are a plant care expert. Synthesize the following web search results into comprehensive, 
            structured care information for {plant_name}.
            
            Create detailed care instructions organized into these categories:
            1. Watering requirements and schedule
            2. Light and placement needs
            3. Soil and fertilization requirements
            4. Temperature and humidity preferences
            5. Common problems and solutions
            6. Propagation methods (if mentioned)
            7. Seasonal care variations
            8. Special care notes
            
            Ensure the information is:
            - Accurate and consistent across sources
            - Practical and actionable
            - Specific to {plant_name}
            - Well-organized and easy to follow
            
            Format as structured JSON with clear categories.
            
            Web search content:
            {content_text}
            """
            
            human_message = f"Please synthesize this information into comprehensive care instructions for {plant_name}."
            
            messages = [
                self.create_system_message(system_prompt),
                self.create_human_message(human_message)
            ]
            
            llm_response = await self.call_llm(messages)
            
            # Try to parse as JSON, fallback to structured text
            try:
                import json
                return json.loads(llm_response)
            except json.JSONDecodeError:
                return self._structure_text_response(llm_response, plant_name)
            
        except Exception as e:
            self.logger.error(f"Error synthesizing care information: {e}")
            return {"error": "Failed to synthesize care information", "raw_response": str(e)}
    
    def _structure_text_response(self, text_response: str, plant_name: str) -> Dict[str, Any]:
        """Structure text response into organized care information"""
        try:
            sections = {
                "watering": self._extract_section_info(text_response, ["water", "irrigation", "moisture"]),
                "lighting": self._extract_section_info(text_response, ["light", "sun", "shade", "bright"]),
                "soil_fertilization": self._extract_section_info(text_response, ["soil", "fertiliz", "nutrient", "feed"]),
                "temperature_humidity": self._extract_section_info(text_response, ["temperature", "humidity", "climate"]),
                "common_problems": self._extract_section_info(text_response, ["problem", "disease", "pest", "issue"]),
                "propagation": self._extract_section_info(text_response, ["propagat", "cutting", "division"]),
                "seasonal_care": self._extract_section_info(text_response, ["season", "winter", "summer", "dormant"]),
                "special_notes": self._extract_section_info(text_response, ["special", "note", "important", "tip"])
            }
            
            # Add metadata
            sections["plant_name"] = plant_name
            sections["source"] = "web_search_synthesis"
            sections["synthesis_method"] = "llm_structured"
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error structuring text response: {e}")
            return {"general_info": text_response, "plant_name": plant_name}
    
    def _extract_section_info(self, text: str, keywords: List[str]) -> str:
        """Extract information related to specific keywords from text"""
        try:
            lines = text.split('\n')
            relevant_lines = []
            
            for line in lines:
                line_lower = line.lower()
                if any(keyword.lower() in line_lower for keyword in keywords):
                    # Include this line and potentially the next few lines for context
                    relevant_lines.append(line.strip())
            
            return '\n'.join(relevant_lines) if relevant_lines else "No specific information found."
            
        except Exception:
            return "Could not extract section information."
    
    async def _update_knowledge_base(self, plant_name: str, processed_info: Dict[str, Any]) -> bool:
        """Update the vector database with new care information"""
        try:
            care_sections = processed_info.get("care_sections", {})
            
            if not care_sections or "error" in care_sections:
                self.logger.warning(f"No valid care information to update for {plant_name}")
                return False
            
            # Create comprehensive care document
            care_document = self._create_care_document(plant_name, care_sections)
            
            # Update vector database
            success = self.vector_db.update_knowledge_base(
                plant_name, 
                care_document, 
                source="web_search"
            )
            
            if success:
                self.logger.info(f"Successfully updated knowledge base for {plant_name}")
            else:
                self.logger.warning(f"Failed to update knowledge base for {plant_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating knowledge base: {e}")
            return False
    
    def _create_care_document(self, plant_name: str, care_sections: Dict[str, Any]) -> str:
        """Create a comprehensive care document from structured sections"""
        try:
            document_parts = [f"Comprehensive Care Guide for {plant_name}"]
            document_parts.append("=" * 50)
            
            # Add each care section
            section_order = [
                ("watering", "Watering Requirements"),
                ("lighting", "Light and Placement"),
                ("soil_fertilization", "Soil and Fertilization"),
                ("temperature_humidity", "Temperature and Humidity"),
                ("common_problems", "Common Problems and Solutions"),
                ("propagation", "Propagation Methods"),
                ("seasonal_care", "Seasonal Care"),
                ("special_notes", "Special Care Notes")
            ]
            
            for section_key, section_title in section_order:
                if section_key in care_sections and care_sections[section_key]:
                    content = care_sections[section_key]
                    if content and content != "No specific information found.":
                        document_parts.append(f"\n{section_title}:")
                        document_parts.append(content)
            
            # Add metadata
            document_parts.append("\nSource: Web search synthesis")
            document_parts.append(f"Updated: {asyncio.get_event_loop().time()}")
            
            return "\n".join(document_parts)
            
        except Exception as e:
            self.logger.error(f"Error creating care document: {e}")
            return f"Basic care information for {plant_name} from web search."
    
    async def search_specific_topic(self, plant_name: str, topic: str) -> Dict[str, Any]:
        """Search for specific care topic information"""
        try:
            input_data = {
                "plant_name": plant_name,
                "specific_query": topic,
                "search_topics": [topic]
            }
            
            return await self.execute(input_data)
            
        except Exception as e:
            self.logger.error(f"Error searching specific topic: {e}")
            return self.create_error_response(str(e), "SPECIFIC_SEARCH_ERROR")
    
    async def validate_care_information(self, plant_name: str, care_info: str) -> Dict[str, Any]:
        """Validate care information against web sources"""
        try:
            # Search for validation information
            validation_query = f"{plant_name} care facts verification"
            search_result = await APIHelper.call_tavily_search_api(validation_query, max_results=3)
            
            # Use LLM to compare and validate
            system_prompt = f"""
            You are a plant care expert. Compare the provided care information for {plant_name} 
            against the web search results and validate its accuracy.
            
            Provide:
            1. Accuracy assessment (0-100%)
            2. Any contradictions found
            3. Missing important information
            4. Recommendations for improvement
            """
            
            human_message = f"Care info to validate: {care_info}\n\nWeb sources: {search_result}"
            
            messages = [
                self.create_system_message(system_prompt),
                self.create_human_message(human_message)
            ]
            
            validation_result = await self.call_llm(messages)
            
            return {
                "plant_name": plant_name,
                "validation_result": validation_result,
                "sources_checked": len(search_result.get("results", [])),
                "validation_method": "web_comparison"
            }
            
        except Exception as e:
            self.logger.error(f"Error validating care information: {e}")
            return self.create_error_response(str(e), "VALIDATION_ERROR")