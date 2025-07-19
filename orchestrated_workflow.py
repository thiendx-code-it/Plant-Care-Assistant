from typing import Dict, Any, Optional
import asyncio
from agents.orchestration_agent import OrchestrationAgent
from utils.vector_db import VectorDBManager

class OrchestratedWorkflow:
    """New workflow using the orchestration agent instead of fixed LangGraph"""
    
    def __init__(self, vector_db: Optional[VectorDBManager] = None):
        # Initialize vector database
        if vector_db is None:
            try:
                self.vector_db = VectorDBManager()
                print("Vector database initialized successfully")
            except Exception as e:
                print(f"Warning: Could not load vector database: {e}")
                self.vector_db = None
        else:
            self.vector_db = vector_db
        
        # Initialize orchestration agent
        self.orchestration_agent = OrchestrationAgent(vector_db=self.vector_db)
        
        # Conversation state
        self.conversation_history = []
    
    async def process_query(self, 
                          user_query: str, 
                          image_base64: Optional[str] = None,
                          image_description: Optional[str] = None,
                          location: Optional[str] = None) -> Dict[str, Any]:
        """Process user query using orchestration agent"""
        try:
            # Add to conversation history
            self.conversation_history.append({
                'type': 'user',
                'query': user_query,
                'has_image': bool(image_base64),
                'location': location
            })
            
            # Prepare input for orchestration agent
            input_data = {
                'user_query': user_query,
                'image_base64': image_base64,
                'image_description': image_description,
                'location': location,
                'conversation_history': self.conversation_history
            }
            
            print(f"\n=== ORCHESTRATED WORKFLOW START ===")
            print(f"User Query: {user_query}")
            print(f"Has Image: {bool(image_base64)}")
            print(f"Location: {location}")
            
            # Execute orchestration agent
            result = await self.orchestration_agent.execute(input_data)
            
            if result.get('success'):
                response_data = result.get('data', {})
                
                # Add to conversation history
                self.conversation_history.append({
                    'type': 'assistant',
                    'response': response_data.get('response', ''),
                    'intent': response_data.get('intent'),
                    'agents_used': response_data.get('agents_used', []),
                    'completeness_score': response_data.get('completeness_score', 0.0)
                })
                
                print(f"\n=== ORCHESTRATION SUMMARY ===")
                print(f"Intent: {response_data.get('intent')}")
                print(f"Intent Confidence: {response_data.get('intent_confidence', 0):.2f}")
                print(f"Agents Used: {', '.join(response_data.get('agents_used', []))}")
                print(f"Completeness Score: {response_data.get('completeness_score', 0):.2f}")
                
                execution_summary = response_data.get('execution_summary', {})
                print(f"Successful Agents: {execution_summary.get('successful_agents', 0)}/{execution_summary.get('total_agents', 0)}")
                if execution_summary.get('failed_agents'):
                    print(f"Failed Agents: {', '.join(execution_summary.get('failed_agents', []))}")
                
                print(f"=== ORCHESTRATED WORKFLOW END ===\n")
                
                return {
                    'success': True,
                    'response': response_data.get('response', ''),
                    'metadata': {
                        'intent': response_data.get('intent'),
                        'intent_confidence': response_data.get('intent_confidence'),
                        'agents_used': response_data.get('agents_used', []),
                        'completeness_score': response_data.get('completeness_score'),
                        'execution_summary': execution_summary,
                        'workflow_type': 'orchestrated'
                    }
                }
            else:
                error_msg = result.get('error', 'Unknown orchestration error')
                print(f"Orchestration failed: {error_msg}")
                
                return {
                    'success': False,
                    'response': f"I apologize, but I encountered an error while processing your request: {error_msg}",
                    'metadata': {
                        'error': error_msg,
                        'workflow_type': 'orchestrated'
                    }
                }
                
        except Exception as e:
            error_msg = f"Workflow execution error: {str(e)}"
            print(f"ERROR: {error_msg}")
            
            return {
                'success': False,
                'response': f"I apologize, but I encountered an unexpected error: {str(e)}",
                'metadata': {
                    'error': error_msg,
                    'workflow_type': 'orchestrated'
                }
            }
    
    def get_conversation_history(self) -> list:
        """Get the conversation history"""
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        # Also clear orchestration agent context
        self.orchestration_agent.context = {
            'conversation_history': [],
            'gathered_info': {},
            'confidence_scores': {},
            'failed_agents': set()
        }
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow statistics"""
        total_queries = len([h for h in self.conversation_history if h['type'] == 'user'])
        total_responses = len([h for h in self.conversation_history if h['type'] == 'assistant'])
        
        # Calculate average completeness score
        completeness_scores = [h.get('completeness_score', 0) for h in self.conversation_history if h['type'] == 'assistant']
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
        
        # Count agent usage
        agent_usage = {}
        for h in self.conversation_history:
            if h['type'] == 'assistant' and 'agents_used' in h:
                for agent in h['agents_used']:
                    agent_usage[agent] = agent_usage.get(agent, 0) + 1
        
        return {
            'total_queries': total_queries,
            'total_responses': total_responses,
            'average_completeness': avg_completeness,
            'agent_usage': agent_usage,
            'workflow_type': 'orchestrated'
        }

# Async wrapper for compatibility with existing code
async def run_orchestrated_workflow(user_query: str, 
                                   image_base64: Optional[str] = None,
                                   image_description: Optional[str] = None,
                                   location: Optional[str] = None,
                                   vector_db: Optional[VectorDBManager] = None) -> Dict[str, Any]:
    """Standalone function to run orchestrated workflow"""
    workflow = OrchestratedWorkflow(vector_db)
    return await workflow.process_query(user_query, image_base64, image_description, location)