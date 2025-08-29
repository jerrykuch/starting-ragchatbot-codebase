
import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools for course information.

Tool Usage Guidelines:
- **Course content questions**: Use search_course_content for specific educational materials and lesson content
- **Course outline questions**: Use get_course_outline for course structure, lesson lists, and course metadata
- **Sequential tool usage**: You can make up to 2 tool calls per query in separate rounds to gather comprehensive information
- **Strategic tool chaining**: Use course outline first to understand structure, then search specific content based on that context
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific content questions**: Use search_course_content tool first, then answer
- **Course outline/structure questions**: Use get_course_outline tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "according to the outline"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_rounds: Optional[int] = None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum tool call rounds (defaults to 2)
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            # Use sequential rounds with configurable maximum
            rounds_limit = max_rounds if max_rounds is not None else 2
            return self._execute_rounds(response, api_params, tool_manager, rounds_limit)
        
        # Return direct response
        return response.content[0].text
    
    def _execute_rounds(self, initial_response, base_params: Dict[str, Any], tool_manager, max_rounds: int):
        """
        Execute sequential tool call rounds with Claude.
        
        Args:
            initial_response: The first response containing tool use requests
            base_params: Base API parameters 
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool rounds allowed
            
        Returns:
            Final response text after all rounds complete
        """
        messages = base_params["messages"].copy()
        system_content = base_params["system"]
        tools = base_params.get("tools", [])
        
        current_response = initial_response
        rounds_completed = 0
        
        # Execute tool rounds sequentially
        while (current_response.stop_reason == "tool_use" and 
               rounds_completed < max_rounds):
            
            # Execute single round
            messages, current_response = self._execute_single_round(
                messages, system_content, tools, tool_manager, current_response
            )
            rounds_completed += 1
        
        # Return final response text
        return current_response.content[0].text
    
    def _execute_single_round(self, messages: List, system_content: str, tools: List, 
                             tool_manager, response) -> tuple:
        """
        Execute a single round of tool calls and get Claude's next response.
        
        Args:
            messages: Current conversation messages
            system_content: System prompt content
            tools: Available tools
            tool_manager: Tool execution manager
            response: Claude's response with tool calls
            
        Returns:
            Tuple of (updated_messages, next_response)
        """
        # Add AI's tool use response to conversation
        messages.append({"role": "assistant", "content": response.content})
        
        # Execute all tool calls in this response
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                except Exception as e:
                    tool_result = f"Tool execution error: {str(e)}"
                
                tool_results.append({
                    "type": "tool_result", 
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results to conversation
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Get Claude's next response with tools still available
        next_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
            "tools": tools,
            "tool_choice": {"type": "auto"}
        }
        
        next_response = self.client.messages.create(**next_params)
        return messages, next_response