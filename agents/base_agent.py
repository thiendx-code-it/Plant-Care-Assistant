from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from config.settings import settings
import logging

class BaseAgent(ABC):
    """Base class for all agents in the Plant Care Assistant system"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"agent.{name}")
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self) -> AzureChatOpenAI:
        """Initialize Azure OpenAI LLM"""
        return AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT_CHAT,
            api_key=settings.AZURE_OPENAI_API_KEY_CHAT,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_CHAT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=0.1
        )
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's main functionality"""
        pass
    
    def log_execution(self, input_data: Dict[str, Any], output_data: Dict[str, Any], execution_time: float):
        """Log agent execution for debugging and audit"""
        self.logger.info(f"Agent {self.name} executed in {execution_time:.2f}s")
        self.logger.debug(f"Input: {input_data}")
        self.logger.debug(f"Output: {output_data}")
    
    def create_system_message(self, system_prompt: str) -> SystemMessage:
        """Create a system message for the LLM"""
        return SystemMessage(content=system_prompt)
    
    def create_human_message(self, content: str, image_base64: Optional[str] = None) -> HumanMessage:
        """Create a human message for the LLM with optional image"""
        if image_base64:
            # For now, just include the content without image processing
            # This can be enhanced later to support vision models
            return HumanMessage(content=f"{content}\n[Image provided but not processed in current implementation]")
        return HumanMessage(content=content)
    
    async def call_llm(self, messages: list[BaseMessage], max_tokens: Optional[int] = None) -> str:
        """Call the LLM with messages and return response"""
        try:
            # Create a temporary LLM instance with max_tokens if specified
            if max_tokens:
                temp_llm = AzureChatOpenAI(
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT_CHAT,
                    api_key=settings.AZURE_OPENAI_API_KEY_CHAT,
                    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_CHAT,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    temperature=0.1,
                    max_tokens=max_tokens
                )
                response = await temp_llm.agenerate([messages])
            else:
                response = await self.llm.agenerate([messages])
            return response.generations[0][0].text.strip()
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            raise
    
    def validate_input(self, input_data: Dict[str, Any], required_fields: list[str]) -> bool:
        """Validate that required fields are present in input data"""
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            self.logger.error(f"Missing required fields: {missing_fields}")
            return False
        return True
    
    def create_error_response(self, error_message: str, error_code: str = "AGENT_ERROR") -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "error": {
                "code": error_code,
                "message": error_message,
                "agent": self.name
            }
        }
    
    def create_success_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized success response"""
        return {
            "success": True,
            "agent": self.name,
            "data": data
        }