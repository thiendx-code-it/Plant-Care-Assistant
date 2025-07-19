from typing import Dict, List, Any, Optional, Tuple
import asyncio
from enum import Enum
from dataclasses import dataclass
from .base_agent import BaseAgent
from .plant_identifier import PlantIdentifierAgent
from .care_advisor import CareAdvisorAgent
from .weather_advisor import WeatherAdvisorAgent
from .knowledge_augmenter import KnowledgeAugmenterAgent
from .disease_detector import DiseaseDetectorAgent
from utils.api_helpers import APIHelper

class UserIntent(Enum):
    """Enumeration of possible user intents"""
    PLANT_IDENTIFICATION = "plant_identification"
    DISEASE_DIAGNOSIS = "disease_diagnosis"
    CARE_ADVICE = "care_advice"
    WATERING_SCHEDULE = "watering_schedule"
    GENERAL_INFO = "general_info"
    TROUBLESHOOTING = "troubleshooting"
    SEASONAL_CARE = "seasonal_care"
    UNKNOWN = "unknown"

class ExecutionStrategy(Enum):
    """Different execution strategies for agent coordination"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ITERATIVE = "iterative"
    FALLBACK_CHAIN = "fallback_chain"

@dataclass
class AgentTask:
    """Represents a task for a specific agent"""
    agent_name: str
    priority: int  # Lower number = higher priority
    required: bool  # Whether this agent is mandatory for the intent
    conditions: List[str]  # Conditions that must be met to execute
    parallel_group: Optional[str] = None  # Agents in same group can run in parallel
    fallback_for: Optional[str] = None  # This agent is a fallback for another

@dataclass
class ExecutionPlan:
    """Represents the execution plan for handling a user query"""
    intent: UserIntent
    tasks: List[AgentTask]
    strategy: ExecutionStrategy
    confidence_threshold: float = 0.7
    max_iterations: int = 3

class OrchestrationAgent(BaseAgent):
    """Master orchestration agent that coordinates all sub-agents intelligently"""
    
    def __init__(self, vector_db=None):
        super().__init__(name="OrchestrationAgent", description="Master agent that coordinates all sub-agents intelligently")
        self.vector_db = vector_db
        
        # Initialize all sub-agents
        self.agents = {
            'plant_identifier': PlantIdentifierAgent(),
            'care_advisor': CareAdvisorAgent(),
            'weather_advisor': WeatherAdvisorAgent(),
            'knowledge_augmenter': KnowledgeAugmenterAgent(),
            'disease_detector': DiseaseDetectorAgent()
        }
        
        # Execution context
        self.context = {
            'conversation_history': [],
            'gathered_info': {},
            'confidence_scores': {},
            'failed_agents': set()
        }
        
        # Define execution plans for different intents
        self.execution_plans = self._define_execution_plans()
    
    def _define_execution_plans(self) -> Dict[UserIntent, ExecutionPlan]:
        """Define execution plans for different user intents"""
        return {
            UserIntent.PLANT_IDENTIFICATION: ExecutionPlan(
                intent=UserIntent.PLANT_IDENTIFICATION,
                strategy=ExecutionStrategy.FALLBACK_CHAIN,
                tasks=[
                    AgentTask('plant_identifier', 1, True, ['has_image']),
                    AgentTask('knowledge_augmenter', 2, False, ['no_plant_identified'], fallback_for='plant_identifier'),
                    AgentTask('care_advisor', 3, False, ['plant_identified'])
                ]
            ),
            
            UserIntent.DISEASE_DIAGNOSIS: ExecutionPlan(
                intent=UserIntent.DISEASE_DIAGNOSIS,
                strategy=ExecutionStrategy.SEQUENTIAL,
                tasks=[
                    AgentTask('plant_identifier', 1, False, ['has_image']),
                    AgentTask('disease_detector', 2, True, []),
                    AgentTask('knowledge_augmenter', 3, True, [], parallel_group='search'),
                    AgentTask('care_advisor', 4, True, ['has_disease_info'])
                ]
            ),
            
            UserIntent.CARE_ADVICE: ExecutionPlan(
                intent=UserIntent.CARE_ADVICE,
                strategy=ExecutionStrategy.PARALLEL,
                tasks=[
                    AgentTask('plant_identifier', 1, False, ['has_image']),
                    AgentTask('knowledge_augmenter', 2, True, [], parallel_group='search'),
                    AgentTask('weather_advisor', 2, False, ['has_location'], parallel_group='search'),
                    AgentTask('care_advisor', 3, True, ['has_search_results'])
                ]
            ),
            
            UserIntent.WATERING_SCHEDULE: ExecutionPlan(
                intent=UserIntent.WATERING_SCHEDULE,
                strategy=ExecutionStrategy.CONDITIONAL,
                tasks=[
                    AgentTask('plant_identifier', 1, False, ['has_image']),
                    AgentTask('weather_advisor', 2, True, ['has_location']),
                    AgentTask('knowledge_augmenter', 2, True, [], parallel_group='search'),
                    AgentTask('care_advisor', 3, True, ['has_weather_and_search'])
                ]
            ),
            
            UserIntent.GENERAL_INFO: ExecutionPlan(
                intent=UserIntent.GENERAL_INFO,
                strategy=ExecutionStrategy.PARALLEL,
                tasks=[
                    AgentTask('knowledge_augmenter', 1, True, [], parallel_group='search'),
                    AgentTask('care_advisor', 2, True, ['has_search_results'])
                ]
            )
        }
    
    async def analyze_intent(self, user_query: str, has_image: bool = False, location: str = None) -> Tuple[UserIntent, float]:
        """Analyze user intent using LLM"""
        try:
            system_prompt = """
You are an expert at analyzing user intents for a plant care chatbot. Analyze the user query and classify it into one of these categories:

1. PLANT_IDENTIFICATION - User wants to identify a plant ("What plant is this?", "Can you identify this plant?")
2. DISEASE_DIAGNOSIS - User reports plant health issues ("My plant has yellow leaves", "Plant looks sick")
3. CARE_ADVICE - User wants general care information ("How to care for roses?", "Best soil for succulents")
4. WATERING_SCHEDULE - User asks about watering ("How often to water?", "Watering schedule for winter")
5. GENERAL_INFO - User wants general plant information ("Tell me about orchids", "Succulent varieties")
6. TROUBLESHOOTING - User has specific plant problems ("Why are leaves dropping?", "Plant not growing")
7. SEASONAL_CARE - User asks about seasonal care ("Winter care tips", "Summer plant care")
8. UNKNOWN - Query doesn't fit other categories

Return a JSON object with:
{
  "intent": "category_name",
  "confidence": 0.95,
  "reasoning": "Brief explanation",
  "key_entities": ["plant_name", "symptom", "care_aspect"]
}
"""
            
            user_prompt = f"""
User Query: "{user_query}"
Has Image: {has_image}
Location Provided: {location is not None}

Analyze this query and classify the user's intent.
"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await APIHelper.call_azure_openai_chat(messages, temperature=0.1, max_tokens=200)
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Clean and parse JSON response
            cleaned_content = content.strip()
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.startswith('```'):
                cleaned_content = cleaned_content[3:]
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()
            
            import json
            intent_data = json.loads(cleaned_content)
            
            intent_str = intent_data.get('intent', 'UNKNOWN')
            confidence = intent_data.get('confidence', 0.5)
            
            # Map string to enum
            try:
                intent = UserIntent(intent_str.lower())
            except ValueError:
                intent = UserIntent.UNKNOWN
            
            return intent, confidence
            
        except Exception as e:
            print(f"Intent analysis error: {e}")
            return UserIntent.UNKNOWN, 0.5
    
    def _check_conditions(self, conditions: List[str], context: Dict[str, Any]) -> bool:
        """Check if conditions are met for agent execution"""
        for condition in conditions:
            if condition == 'has_image' and not context.get('image_base64'):
                return False
            elif condition == 'has_location' and not context.get('location'):
                return False
            elif condition == 'plant_identified' and not context.get('identified_plant'):
                return False
            elif condition == 'no_plant_identified' and context.get('identified_plant'):
                return False
            elif condition == 'has_disease_info' and not context.get('disease_info'):
                return False
            elif condition == 'has_search_results' and not (context.get('knowledge_results') or context.get('web_search_results')):
                return False
            elif condition == 'has_weather_and_search' and not (context.get('weather_data') and (context.get('knowledge_results') or context.get('web_search_results'))):
                return False
        return True
    
    async def create_execution_plan(self, intent: UserIntent, context: Dict[str, Any]) -> List[AgentTask]:
        """Create dynamic execution plan based on intent and context"""
        base_plan = self.execution_plans.get(intent, self.execution_plans[UserIntent.GENERAL_INFO])
        
        # Filter tasks based on conditions and failed agents
        executable_tasks = []
        for task in base_plan.tasks:
            if task.agent_name not in self.context['failed_agents']:
                if not task.conditions or self._check_conditions(task.conditions, context):
                    executable_tasks.append(task)
                elif not task.required:
                    # Skip optional tasks that don't meet conditions
                    continue
                else:
                    # Required task can't be executed - might need fallback
                    if task.fallback_for:
                        continue
            
        return executable_tasks
    
    async def execute_agent(self, agent_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific agent with error handling"""
        try:
            agent = self.agents.get(agent_name)
            if not agent:
                return {"success": False, "error": f"Agent {agent_name} not found"}
            
            # Special handling for knowledge search using vector DB
            if agent_name == 'knowledge_augmenter' and self.vector_db:
                # Perform vector DB search first
                plant_name = input_data.get('plant_name', 'Unknown')
                query = input_data.get('specific_query', '')
                search_query = f"{plant_name} {query}".strip()
                
                try:
                    vector_results = self.vector_db.similarity_search(search_query, k=5)
                    knowledge_results = []
                    for doc in vector_results:
                        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                        metadata = getattr(doc, 'metadata', {})
                        knowledge_results.append({
                            "content": content,
                            "metadata": metadata,
                            "score": 0.8
                        })
                    
                    # Add vector DB results to input for web search
                    input_data['knowledge_results'] = knowledge_results
                except Exception as e:
                    print(f"Vector DB search error: {e}")
                    input_data['knowledge_results'] = []
            
            result = await agent.execute(input_data)
            
            if result.get('success'):
                self.context['confidence_scores'][agent_name] = result.get('confidence', 0.8)
            else:
                self.context['failed_agents'].add(agent_name)
            
            return result
            
        except Exception as e:
            self.context['failed_agents'].add(agent_name)
            return {"success": False, "error": str(e)}
    
    async def execute_plan(self, tasks: List[AgentTask], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent plan with appropriate strategy"""
        results = {}
        
        # Group tasks by parallel groups
        parallel_groups = {}
        sequential_tasks = []
        
        for task in sorted(tasks, key=lambda t: t.priority):
            if task.parallel_group:
                if task.parallel_group not in parallel_groups:
                    parallel_groups[task.parallel_group] = []
                parallel_groups[task.parallel_group].append(task)
            else:
                sequential_tasks.append(task)
        
        # Execute sequential tasks
        for task in sequential_tasks:
            if self._check_conditions(task.conditions, context):
                input_data = self._prepare_agent_input(task.agent_name, context)
                result = await self.execute_agent(task.agent_name, input_data)
                results[task.agent_name] = result
                
                # Update context with results
                if result.get('success'):
                    self._update_context_with_result(task.agent_name, result, context)
        
        # Execute parallel groups
        for group_name, group_tasks in parallel_groups.items():
            # Check if any task in the group can be executed
            executable_tasks = [t for t in group_tasks if self._check_conditions(t.conditions, context)]
            
            if executable_tasks:
                # Execute tasks in parallel
                parallel_results = await asyncio.gather(*[
                    self.execute_agent(task.agent_name, self._prepare_agent_input(task.agent_name, context))
                    for task in executable_tasks
                ], return_exceptions=True)
                
                # Process parallel results
                for task, result in zip(executable_tasks, parallel_results):
                    if not isinstance(result, Exception):
                        results[task.agent_name] = result
                        if result.get('success'):
                            self._update_context_with_result(task.agent_name, result, context)
        
        return results
    
    def _prepare_agent_input(self, agent_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input data for specific agent based on context"""
        base_input = {
            'user_query': context.get('user_query', ''),
            'plant_name': context.get('identified_plant', {}).get('plant_name', 'Unknown') if context.get('identified_plant') else 'Unknown'
        }
        
        if agent_name == 'plant_identifier':
            base_input.update({
                'image_base64': context.get('image_base64'),
                'description': context.get('image_description')
            })
        elif agent_name == 'weather_advisor':
            base_input.update({
                'location': context.get('location')
            })
        elif agent_name == 'knowledge_augmenter':
            base_input.update({
                'specific_query': context.get('user_query', '')
            })
        elif agent_name == 'disease_detector':
            base_input.update({
                'image_base64': context.get('image_base64'),
                'image_description': context.get('image_description')
            })
        elif agent_name == 'care_advisor':
            base_input.update({
                'health_issues': context.get('health_issues', []),
                'weather_data': context.get('weather_data'),
                'location': context.get('location'),
                'plant_identification': context.get('identified_plant'),
                'knowledge_results': context.get('knowledge_results', []),
                'web_search_results': context.get('web_search_results', []),
                'image_base64': context.get('image_base64'),
                'image_description': context.get('image_description')
            })
        
        return base_input
    
    def _update_context_with_result(self, agent_name: str, result: Dict[str, Any], context: Dict[str, Any]):
        """Update context with agent execution results"""
        if not result.get('success'):
            return
        
        data = result.get('data', {})
        
        if agent_name == 'plant_identifier':
            context['identified_plant'] = data
        elif agent_name == 'weather_advisor':
            context['weather_data'] = data
        elif agent_name == 'knowledge_augmenter':
            context['web_search_results'] = data.get('sources', [])
        elif agent_name == 'disease_detector':
            context['disease_info'] = data
            context['health_issues'] = data.get('health_issues', [])
        elif agent_name == 'care_advisor':
            context['final_response'] = data.get('care_advice', data.get('advice', str(data)))
    
    def assess_completeness(self, intent: UserIntent, context: Dict[str, Any]) -> Tuple[bool, float]:
        """Assess if gathered information is sufficient for the intent"""
        required_info = {
            UserIntent.PLANT_IDENTIFICATION: ['identified_plant'],
            UserIntent.DISEASE_DIAGNOSIS: ['disease_info', 'final_response'],
            UserIntent.CARE_ADVICE: ['final_response'],
            UserIntent.WATERING_SCHEDULE: ['final_response'],
            UserIntent.GENERAL_INFO: ['final_response']
        }
        
        required = required_info.get(intent, ['final_response'])
        available = sum(1 for req in required if context.get(req))
        completeness = available / len(required) if required else 1.0
        
        # Factor in confidence scores
        avg_confidence = sum(self.context['confidence_scores'].values()) / len(self.context['confidence_scores']) if self.context['confidence_scores'] else 0.5
        
        overall_score = (completeness * 0.7) + (avg_confidence * 0.3)
        
        return completeness >= 0.8, overall_score
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution method for orchestration agent"""
        try:
            # Extract input parameters
            user_query = input_data.get('user_query', '')
            image_base64 = input_data.get('image_base64')
            location = input_data.get('location')
            
            # Initialize execution context
            context = {
                'user_query': user_query,
                'image_base64': image_base64,
                'image_description': input_data.get('image_description'),
                'location': location
            }
            
            # Step 1: Analyze user intent
            intent, intent_confidence = await self.analyze_intent(
                user_query, 
                has_image=bool(image_base64), 
                location=location
            )
            
            print(f"DEBUG - Orchestration: Detected intent {intent.value} with confidence {intent_confidence:.2f}")
            
            # Step 2: Create execution plan
            tasks = await self.create_execution_plan(intent, context)
            print(f"DEBUG - Orchestration: Created plan with {len(tasks)} tasks")
            
            # Step 3: Execute plan
            execution_results = await self.execute_plan(tasks, context)
            
            # Step 4: Assess completeness
            is_complete, completeness_score = self.assess_completeness(intent, context)
            
            print(f"DEBUG - Orchestration: Execution complete. Completeness: {completeness_score:.2f}")
            
            # Prepare final response
            final_response = context.get('final_response', 'I apologize, but I was unable to provide a complete response to your query.')
            
            return {
                'success': True,
                'data': {
                    'response': final_response,
                    'intent': intent.value,
                    'intent_confidence': intent_confidence,
                    'completeness_score': completeness_score,
                    'agents_used': list(execution_results.keys()),
                    'context': context,
                    'execution_summary': {
                        'total_agents': len(execution_results),
                        'successful_agents': len([r for r in execution_results.values() if r.get('success')]),
                        'failed_agents': list(self.context['failed_agents']),
                        'confidence_scores': self.context['confidence_scores']
                    }
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Orchestration error: {str(e)}'
            }