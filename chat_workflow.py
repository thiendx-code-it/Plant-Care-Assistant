"""LangGraph workflow for AI-powered plant care chat system."""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from agents.plant_identifier import PlantIdentifierAgent
from agents.care_advisor import CareAdvisorAgent
from agents.weather_advisor import WeatherAdvisorAgent
from agents.knowledge_augmenter import KnowledgeAugmenterAgent
from utils.vector_db import VectorDBManager
from utils.image_utils import prepare_image_for_api


@dataclass
class ChatState:
    """State for the chat workflow."""
    # Input data
    user_query: str = ""
    image_base64: Optional[str] = None
    image_description: str = ""
    location: str = ""
    
    # Workflow data
    identified_plant: Dict[str, Any] = None
    knowledge_results: List[Dict[str, Any]] = None
    weather_data: Dict[str, Any] = None
    web_search_results: List[Dict[str, Any]] = None
    
    # Output data
    final_response: str = ""
    confidence_score: float = 0.0
    sources_used: Dict[str, Any] = None
    
    # Feedback data
    user_feedback_score: Optional[int] = None
    feedback_comments: str = ""
    knowledge_updated: bool = False
    
    # Workflow control
    current_step: str = "start"
    error_message: str = ""
    needs_web_search: bool = False


class PlantCareWorkflow:
    """LangGraph workflow for plant care chat system."""
    
    def __init__(self, vector_db: VectorDBManager):
        self.vector_db = vector_db
        self.plant_identifier = PlantIdentifierAgent()
        self.care_advisor = CareAdvisorAgent()
        self.weather_advisor = WeatherAdvisorAgent()
        self.knowledge_augmenter = KnowledgeAugmenterAgent()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("identify_plant", self._identify_plant_node)
        workflow.add_node("search_knowledge", self._search_knowledge_node)
        workflow.add_node("get_weather", self._get_weather_node)
        workflow.add_node("web_search", self._web_search_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("collect_feedback", self._collect_feedback_node)
        workflow.add_node("update_knowledge", self._update_knowledge_node)
        
        # Define the workflow flow
        workflow.set_entry_point("identify_plant")
        
        # Plant identification -> Knowledge search
        workflow.add_edge("identify_plant", "search_knowledge")
        
        # Knowledge search -> Weather (if knowledge found) or Web search (if not)
        workflow.add_conditional_edges(
            "search_knowledge",
            self._should_web_search,
            {
                "web_search": "web_search",
                "weather": "get_weather"
            }
        )
        
        # Web search -> Weather
        workflow.add_edge("web_search", "get_weather")
        
        # Weather -> Generate response
        workflow.add_edge("get_weather", "generate_response")
        
        # Generate response -> Collect feedback
        workflow.add_edge("generate_response", "collect_feedback")
        
        # Collect feedback -> Update knowledge (if score > 70%) or END
        workflow.add_conditional_edges(
            "collect_feedback",
            self._should_update_knowledge,
            {
                "update": "update_knowledge",
                "end": END
            }
        )
        
        # Update knowledge -> END
        workflow.add_edge("update_knowledge", END)
        
        return workflow.compile()
    
    async def _identify_plant_node(self, state: ChatState) -> ChatState:
        """Node to identify the plant from image."""
        state.current_step = "Identifying plant from image..."
        
        try:
            if state.image_base64:
                input_data = {
                    "image_base64": state.image_base64,
                    "description": state.image_description
                }
                
                result = await self.plant_identifier.execute(input_data)
                
                if result.get("success"):
                    state.identified_plant = result["data"]
                    state.confidence_score = result["data"].get("confidence", 0.0)
                else:
                    state.error_message = f"Plant identification failed: {result.get('error', 'Unknown error')}"
                    state.identified_plant = {"plant_name": "Unknown", "confidence": 0.0}
            else:
                # If no image, try to extract plant name from query
                state.identified_plant = {"plant_name": "Unknown", "confidence": 0.0}
                
        except Exception as e:
            state.error_message = f"Error in plant identification: {str(e)}"
        
        return state
    
    async def _search_knowledge_node(self, state: ChatState) -> ChatState:
        """Node to search the knowledge base."""
        state.current_step = "Searching knowledge base..."
        
        try:
            plant_name = state.identified_plant.get("plant_name", "Unknown") if state.identified_plant else "Unknown"
            
            # Search for relevant information
            search_query = f"{plant_name} {state.user_query}"
            results = self.vector_db.similarity_search(search_query, k=5)
            
            # Convert Document objects to dictionaries for easier handling
            knowledge_results = []
            for doc in results:
                # Handle both Document objects and dictionaries
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                    metadata = getattr(doc, 'metadata', {})
                elif isinstance(doc, dict):
                    content = doc.get('content', str(doc))
                    metadata = doc.get('metadata', {})
                else:
                    content = str(doc)
                    metadata = {}
                
                knowledge_results.append({
                    "content": content,
                    "metadata": metadata,
                    "score": 0.8  # Default score since similarity_search doesn't return scores
                })
            
            state.knowledge_results = knowledge_results
            
            # Determine if we have sufficient knowledge
            if not results or len(results) == 0:
                state.needs_web_search = True
            else:
                # If we have results, assume they are relevant enough
                state.needs_web_search = len(results) < 2  # Need at least 2 relevant documents
                
        except Exception as e:
            state.error_message = f"Error searching knowledge base: {str(e)}"
            state.needs_web_search = True
        
        return state
    
    async def _get_weather_node(self, state: ChatState) -> ChatState:
        """Node to get weather information."""
        state.current_step = "Getting weather information..."
        
        try:
            if state.location:
                input_data = {
                    "location": state.location,
                    "plant_name": state.identified_plant.get("plant_name", "Unknown") if state.identified_plant else "Unknown"
                }
                
                result = await self.weather_advisor.execute(input_data)
                
                if result.get("success"):
                    state.weather_data = result["data"]
                else:
                    state.weather_data = {"error": "Weather data unavailable"}
            else:
                state.weather_data = {"error": "Location not provided"}
                
        except Exception as e:
            state.error_message = f"Error getting weather data: {str(e)}"
            state.weather_data = {"error": str(e)}
        
        return state
    
    async def _web_search_node(self, state: ChatState) -> ChatState:
        """Node to perform web search when knowledge base is insufficient."""
        state.current_step = "Searching the web for additional information..."
        
        try:
            plant_name = state.identified_plant.get("plant_name", "Unknown") if state.identified_plant else "Unknown"
            
            input_data = {
                "plant_name": plant_name,
                "search_topic": state.user_query
            }
            
            result = await self.knowledge_augmenter.execute(input_data)
            
            if result.get("success"):
                state.web_search_results = result["data"].get("search_results", [])
            else:
                state.web_search_results = []
                
        except Exception as e:
            state.error_message = f"Error in web search: {str(e)}"
            state.web_search_results = []
        
        return state
    
    async def _generate_response_node(self, state: ChatState) -> ChatState:
        """Node to generate the final response."""
        state.current_step = "Generating comprehensive response..."
        
        try:
            plant_name = state.identified_plant.get("plant_name", "Unknown") if state.identified_plant else "Unknown"
            
            # Extract health issues from plant identification if available
            health_issues = []
            if state.identified_plant and "health_assessment" in state.identified_plant:
                health_data = state.identified_plant["health_assessment"]
                if "diseases" in health_data:
                    for disease in health_data["diseases"]:
                        health_issues.append({
                            "type": "disease",
                            "name": disease.get("name", "Unknown disease"),
                            "probability": disease.get("probability", 0.0)
                        })
                if "pests" in health_data:
                    for pest in health_data["pests"]:
                        health_issues.append({
                            "type": "pest",
                            "name": pest.get("name", "Unknown pest"),
                            "probability": pest.get("probability", 0.0)
                        })
            
            # Prepare comprehensive input for care advisor
            input_data = {
                "plant_name": plant_name,
                "specific_query": state.user_query,
                "health_issues": health_issues,
                "weather_data": state.weather_data,
                "location": state.location,
                "season": "current",  # Could be enhanced to detect season
                "plant_identification": state.identified_plant,
                "knowledge_results": state.knowledge_results,
                "web_search_results": state.web_search_results
            }
            
            # Add image data if available
            if state.image_base64:
                input_data["image_base64"] = state.image_base64
                input_data["image_description"] = state.image_description
            
            result = await self.care_advisor.execute(input_data)
            
            if result.get("success"):
                advice_data = result["data"]
                # Handle both old and new response formats
                if "care_advice" in advice_data:
                    state.final_response = advice_data["care_advice"]
                elif "advice" in advice_data:
                    state.final_response = advice_data["advice"]
                else:
                    state.final_response = str(advice_data)
                
                # Track sources used with detailed step-by-step information
                sources = {
                    "summary": [],
                    "workflow_steps": [],
                    "details": {
                        "plant_identification": {
                            "used": bool(state.identified_plant),
                            "step_name": "🌱 Plant Identification",
                            "status": "completed" if state.identified_plant else "skipped",
                            "result": {
                                "plant_name": state.identified_plant.get("plant_name", "Not identified") if state.identified_plant else None,
                                "scientific_name": state.identified_plant.get("scientific_name", "Not available") if state.identified_plant else None,
                                "confidence": f"{state.identified_plant.get('confidence', 0.0):.1%}" if state.identified_plant else "0%",
                                "health_status": "Healthy" if state.identified_plant and state.identified_plant.get('health_assessment', {}).get('is_healthy') else "Issues detected" if state.identified_plant and 'health_assessment' in state.identified_plant else "Not assessed"
                            } if state.identified_plant else None
                        },
                        "knowledge_base": {
                            "used": bool(state.knowledge_results),
                            "step_name": "📚 Knowledge Base Search",
                            "status": "completed" if state.knowledge_results else "no_results",
                            "count": len(state.knowledge_results) if state.knowledge_results else 0,
                            "sources": [{
                                "content_preview": (item.get('content', str(item))[:150] + "...") if hasattr(item, 'get') or isinstance(item, dict) else str(item)[:150] + "...",
                                "metadata": item.get('metadata', {}) if hasattr(item, 'get') or isinstance(item, dict) else {}
                            } for item in (state.knowledge_results[:3] if state.knowledge_results else [])]
                        },
                        "web_search": {
                            "used": bool(state.web_search_results),
                            "step_name": "🌐 Web Research",
                            "status": "completed" if state.web_search_results else "skipped",
                            "count": len(state.web_search_results) if state.web_search_results else 0,
                            "sources": [{
                                "title": item.get("title", "No title available"),
                                "url": item.get("url", "No URL available"),
                                "snippet": (item.get("snippet", item.get("content", ""))[:200] + "...") if item.get("snippet", item.get("content", "")) else "No content available"
                            } for item in (state.web_search_results[:3] if state.web_search_results else [])]
                        },
                        "weather_analysis": {
                            "used": bool(state.weather_data and "error" not in state.weather_data),
                            "step_name": "🌤️ Weather Analysis",
                            "status": "completed" if state.weather_data and "error" not in state.weather_data else "failed" if state.weather_data else "skipped",
                            "data": {
                                "temperature": f"{state.weather_data.get('current_weather', {}).get('temperature', 'N/A')}°C" if state.weather_data and "error" not in state.weather_data else None,
                                "humidity": f"{state.weather_data.get('current_weather', {}).get('humidity', 'N/A')}%" if state.weather_data and "error" not in state.weather_data else None,
                                "description": state.weather_data.get('current_weather', {}).get('description', 'N/A') if state.weather_data and "error" not in state.weather_data else None
                            } if state.weather_data and "error" not in state.weather_data else None
                        },
                        "image_analysis": {
                            "used": bool(state.image_base64),
                            "step_name": "📸 Image Analysis",
                            "status": "completed" if state.image_base64 else "skipped",
                            "has_description": bool(state.image_description),
                            "description_preview": state.image_description[:100] + "..." if state.image_description and len(state.image_description) > 100 else state.image_description
                        },
                        "care_generation": {
                            "used": True,
                            "step_name": "🤖 Care Advice Generation",
                            "status": "completed",
                            "response_type": "care_advice" if "care_advice" in advice_data else "advice" if "advice" in advice_data else "raw_data",
                            "sources_combined": sum([
                                len(state.knowledge_results) if state.knowledge_results else 0,
                                len(state.web_search_results) if state.web_search_results else 0,
                                1 if state.weather_data and "error" not in state.weather_data else 0,
                                1 if state.image_base64 else 0,
                                1 if state.identified_plant else 0
                            ])
                        }
                    }
                }
                
                # Build workflow steps for detailed display
                workflow_order = ["plant_identification", "knowledge_base", "web_search", "weather_analysis", "image_analysis", "care_generation"]
                
                for step_key in workflow_order:
                    step_info = sources["details"][step_key]
                    if step_info["used"] or step_info["status"] != "skipped":
                        step_data = {
                            "name": step_info["step_name"],
                            "status": step_info["status"],
                            "details": {}
                        }
                        
                        # Add specific details for each step
                        if step_key == "plant_identification" and step_info["result"]:
                            step_data["details"] = {
                                "plant_name": step_info["result"]["plant_name"],
                                "scientific_name": step_info["result"]["scientific_name"],
                                "confidence": step_info["result"]["confidence"],
                                "health_status": step_info["result"]["health_status"]
                            }
                        elif step_key == "knowledge_base" and step_info["used"]:
                            step_data["details"] = {
                                "sources_found": step_info["count"],
                                "preview": [src["content_preview"] for src in step_info["sources"][:2]]
                            }
                        elif step_key == "web_search" and step_info["used"]:
                            step_data["details"] = {
                                "results_found": step_info["count"],
                                "top_sources": [f"{src['title']} - {src['url']}" for src in step_info["sources"][:2]]
                            }
                        elif step_key == "weather_analysis" and step_info["data"]:
                            step_data["details"] = step_info["data"]
                        elif step_key == "image_analysis" and step_info["used"]:
                            step_data["details"] = {
                                "description_available": step_info["has_description"],
                                "preview": step_info["description_preview"] if step_info["has_description"] else "Image processed"
                            }
                        elif step_key == "care_generation":
                            step_data["details"] = {
                                "total_sources_used": step_info["sources_combined"],
                                "response_type": step_info["response_type"]
                            }
                        
                        sources["workflow_steps"].append(step_data)
                
                # Build summary list for display
                if sources["details"]["plant_identification"]["used"] and sources["details"]["plant_identification"]["result"]:
                    plant_name = sources["details"]["plant_identification"]["result"]["plant_name"]
                    sources["summary"].append(f"🌱 Plant ID: {plant_name}")
                if sources["details"]["knowledge_base"]["used"]:
                    sources["summary"].append(f"📚 Knowledge Base ({sources['details']['knowledge_base']['count']} sources)")
                if sources["details"]["web_search"]["used"]:
                    sources["summary"].append(f"🌐 Web Search ({sources['details']['web_search']['count']} results)")
                if sources["details"]["weather_analysis"]["used"]:
                    sources["summary"].append("🌤️ Weather Data")
                if sources["details"]["image_analysis"]["used"]:
                    sources["summary"].append("📸 Image Analysis")
                
                state.sources_used = sources
            else:
                state.final_response = f"Error generating response: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            state.error_message = f"Error generating response: {str(e)}"
            state.final_response = "Sorry, I encountered an error while generating the response."
        
        return state
    
    async def _collect_feedback_node(self, state: ChatState) -> ChatState:
        """Node to collect user feedback (this will be handled by the UI)."""
        state.current_step = "Waiting for user feedback..."
        # This node is mainly a placeholder - actual feedback collection happens in the UI
        return state
    
    async def _update_knowledge_node(self, state: ChatState) -> ChatState:
        """Node to update knowledge base if feedback score is high enough."""
        state.current_step = "Updating knowledge base..."
        
        try:
            if state.user_feedback_score and state.user_feedback_score >= 70:
                # Prepare data for knowledge base update
                plant_name = state.identified_plant.get("plant_name", "Not identified") if state.identified_plant else "Not identified"
                
                knowledge_entry = {
                    "plant_name": plant_name,
                    "query": state.user_query,
                    "response": state.final_response,
                    "sources": state.sources_used.get("summary", []) if state.sources_used else [],
                    "sources_details": state.sources_used.get("details", {}) if state.sources_used else {},
                    "feedback_score": state.user_feedback_score,
                    "feedback_comments": state.feedback_comments
                }
                
                # Add to knowledge base
                content = f"Plant: {plant_name}\nQuery: {state.user_query}\nAdvice: {state.final_response}"
                self.vector_db.add_documents([content], [knowledge_entry])
                
                state.knowledge_updated = True
            
        except Exception as e:
            state.error_message = f"Error updating knowledge base: {str(e)}"
        
        return state
    
    def _should_web_search(self, state: ChatState) -> str:
        """Conditional edge function to determine if web search is needed."""
        return "web_search" if state.needs_web_search else "weather"
    
    def _should_update_knowledge(self, state: ChatState) -> str:
        """Conditional edge function to determine if knowledge should be updated."""
        if state.user_feedback_score and state.user_feedback_score >= 70:
            return "update"
        return "end"
    
    async def run_workflow(self, initial_state: ChatState) -> ChatState:
        """Run the complete workflow."""
        try:
            # Execute the workflow
            result = await self.workflow.ainvoke(initial_state)
            
            # LangGraph returns a dictionary, convert it back to ChatState
            if isinstance(result, dict):
                # Create a new ChatState with the result data
                final_state = ChatState(
                    user_query=result.get('user_query', initial_state.user_query),
                    image_base64=result.get('image_base64', initial_state.image_base64),
                    image_description=result.get('image_description', initial_state.image_description),
                    location=result.get('location', initial_state.location),
                    identified_plant=result.get('identified_plant', {}),
                    knowledge_results=result.get('knowledge_results', []),
                    needs_web_search=result.get('needs_web_search', False),
                    web_search_results=result.get('web_search_results', []),
                    weather_data=result.get('weather_data', {}),
                    final_response=result.get('final_response', ''),
                    sources_used=result.get('sources_used', {}),
                    user_feedback_score=result.get('user_feedback_score'),
                    feedback_comments=result.get('feedback_comments', ''),
                    knowledge_updated=result.get('knowledge_updated', False),
                    current_step=result.get('current_step', ''),
                    error_message=result.get('error_message', '')
                )
                return final_state
            else:
                # If it's already a ChatState object, return as is
                return result
        except Exception as e:
            initial_state.error_message = f"Workflow execution error: {str(e)}"
            return initial_state
    
    async def update_feedback(self, state: ChatState, feedback_score: int, comments: str = "") -> ChatState:
        """Update the state with user feedback and potentially update knowledge base."""
        state.user_feedback_score = feedback_score
        state.feedback_comments = comments
        
        # If score is high enough, update knowledge base
        if feedback_score >= 70:
            state = await self._update_knowledge_node(state)
        
        return state