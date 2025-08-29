import pytest
from unittest.mock import Mock, patch, MagicMock, call
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test AIGenerator functionality and tool calling"""
    
    def test_init(self):
        """Test AIGenerator initialization"""
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
        assert ai_gen.model == "claude-sonnet-4-20250514"
        assert ai_gen.base_params["model"] == "claude-sonnet-4-20250514"
        assert ai_gen.base_params["temperature"] == 0
        assert ai_gen.base_params["max_tokens"] == 800
    
    @patch('anthropic.Anthropic')
    def test_generate_response_simple(self, mock_anthropic):
        """Test simple response generation without tools"""
        # Setup mock client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Simple response")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response("What is Python?")
        
        assert result == "Simple response"
        mock_client.messages.create.assert_called_once()
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Test response generation with conversation context"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with context")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_response
        
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
        ai_gen.client = mock_client
        
        history = "Previous: User asked about basics\nAssistant: Here are the basics..."
        result = ai_gen.generate_response("Follow up question", conversation_history=history)
        
        assert result == "Response with context"
        
        # Verify system content includes history
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation" in system_content
        assert history in system_content
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic):
        """Test response with tools available but not used"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Direct answer without tools")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        mock_tool_manager = Mock()
        
        result = ai_gen.generate_response(
            "What is 2+2?", 
            tools=tools, 
            tool_manager=mock_tool_manager
        )
        
        assert result == "Direct answer without tools"
        
        # Verify tools were included in API call
        call_args = mock_client.messages.create.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tools"] == tools
    
    @patch('anthropic.Anthropic')
    def test_generate_response_with_tool_execution(self, mock_anthropic):
        """Test response generation with tool execution (two-phase process)"""
        mock_client = Mock()
        
        # First response: AI decides to use tools
        mock_tool_use_content = Mock()
        mock_tool_use_content.type = "tool_use"
        mock_tool_use_content.name = "search_course_content"
        mock_tool_use_content.id = "tool_123"
        mock_tool_use_content.input = {"query": "test search"}
        
        initial_response = Mock()
        initial_response.content = [mock_tool_use_content]
        initial_response.stop_reason = "tool_use"
        
        # Second response: AI synthesizes tool results
        final_response = Mock()
        final_response.content = [Mock(text="Answer based on search results")]
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Setup tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results from tool"
        
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514") 
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        result = ai_gen.generate_response(
            "Tell me about Python basics",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Answer based on search results"
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test search"
        )
        
        # Verify two API calls were made (initial + follow-up)
        assert mock_client.messages.create.call_count == 2
    
    @patch('anthropic.Anthropic')
    def test_tool_execution_flow_with_multiple_tools(self, mock_anthropic):
        """Test handling multiple tool calls in single response"""
        mock_client = Mock()
        
        # Create mock tool use blocks
        tool_use_1 = Mock()
        tool_use_1.type = "tool_use"
        tool_use_1.name = "search_course_content"
        tool_use_1.id = "tool_123"
        tool_use_1.input = {"query": "first search"}
        
        tool_use_2 = Mock()
        tool_use_2.type = "tool_use"
        tool_use_2.name = "get_course_outline"
        tool_use_2.id = "tool_456"
        tool_use_2.input = {"course_title": "Test Course"}
        
        initial_response = Mock()
        initial_response.content = [tool_use_1, tool_use_2]
        initial_response.stop_reason = "tool_use"
        
        final_response = Mock()
        final_response.content = [Mock(text="Combined answer from both tools")]
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Setup tool manager for multiple tools
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "First search results",
            "Course outline results"
        ]
        
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
        
        tools = [
            {"name": "search_course_content", "description": "Search tool"},
            {"name": "get_course_outline", "description": "Outline tool"}
        ]
        result = ai_gen.generate_response(
            "Tell me about the course structure and content",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Combined answer from both tools"
        assert mock_tool_manager.execute_tool.call_count == 2
    
    @patch('anthropic.Anthropic')  
    def test_tool_execution_handles_errors(self, mock_anthropic):
        """Test that tool execution errors are properly handled"""
        mock_client = Mock()
        
        # First response: AI decides to use tool
        mock_tool_use_content = Mock()
        mock_tool_use_content.type = "tool_use"
        mock_tool_use_content.name = "search_course_content"
        mock_tool_use_content.id = "tool_123"
        mock_tool_use_content.input = {"query": "test"}
        
        initial_response = Mock()
        initial_response.content = [mock_tool_use_content]
        initial_response.stop_reason = "tool_use"
        
        # Second response: AI handles tool error
        final_response = Mock()
        final_response.content = [Mock(text="Sorry, search failed")]
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Setup tool manager to return error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "No relevant content found"
        
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
        
        result = ai_gen.generate_response(
            "Search for something",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        assert result == "Sorry, search failed"


class TestAIGeneratorSequentialTooling:
    """Test sequential tool calling functionality"""
    
    @patch('anthropic.Anthropic')
    def test_single_tool_call_behavior_preserved(self, mock_anthropic):
        """Test that existing single tool call behavior works unchanged"""
        mock_client = Mock()
        
        # Single round: tool_use → final text response
        tool_use_response = Mock()
        tool_use_response.stop_reason = "tool_use" 
        tool_use_response.content = [Mock(type="tool_use", name="search_course_content", id="t1", input={"query": "test"})]
        
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Single tool result answer")]
        
        mock_client.messages.create.side_effect = [tool_use_response, final_response]
        mock_anthropic.return_value = mock_client
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results"
        
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Search for Python basics",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        assert result == "Single tool result answer"
        assert mock_client.messages.create.call_count == 2  # Initial + follow-up
        mock_tool_manager.execute_tool.assert_called_once()
    
    @patch('anthropic.Anthropic')
    def test_two_sequential_tool_calls_success(self, mock_anthropic):
        """Test successful two-round tool calling sequence"""
        mock_client = Mock()
        
        # Round 1: AI requests first tool
        tool1_block = Mock()
        tool1_block.type = "tool_use"
        tool1_block.name = "get_course_outline"
        tool1_block.id = "t1"
        tool1_block.input = {"course_title": "MCP"}
        
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        round1_response.content = [tool1_block]
        
        # Round 2: AI requests second tool after seeing first results  
        tool2_block = Mock()
        tool2_block.type = "tool_use"
        tool2_block.name = "search_course_content"
        tool2_block.id = "t2"
        tool2_block.input = {"query": "basics", "course_name": "MCP"}
        
        round2_response = Mock()
        round2_response.stop_reason = "tool_use"
        round2_response.content = [tool2_block]
        
        # Round 3: Final synthesis without tools
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Combined answer from both tools")]
        
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        mock_anthropic.return_value = mock_client
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline: Lesson 1: Intro, Lesson 2: Advanced",
            "Search results: MCP basics involve connecting tools..."
        ]
        
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Tell me about MCP course structure and basic concepts",
            tools=[
                {"name": "get_course_outline", "description": "Get course outline"},
                {"name": "search_course_content", "description": "Search content"}
            ],
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        assert result == "Combined answer from both tools"
        assert mock_client.messages.create.call_count == 3  # 2 tool rounds + final
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify tool execution order
        expected_calls = [
            call("get_course_outline", course_title="MCP"),
            call("search_course_content", query="basics", course_name="MCP")
        ]
        mock_tool_manager.execute_tool.assert_has_calls(expected_calls)
    
    @patch('anthropic.Anthropic')
    def test_tool_call_followed_by_direct_response(self, mock_anthropic):
        """Test tool call in first round, direct response in second"""
        mock_client = Mock()
        
        # Round 1: AI uses tool
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        round1_response.content = [Mock(type="tool_use", name="search_course_content", id="t1", input={"query": "Python"})]
        
        # Round 2: AI provides direct answer (no tools)
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Based on the search results, Python is...")]
        
        mock_client.messages.create.side_effect = [round1_response, final_response]
        mock_anthropic.return_value = mock_client
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Python programming content"
        
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "What is Python?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        assert result == "Based on the search results, Python is..."
        assert mock_client.messages.create.call_count == 2
        mock_tool_manager.execute_tool.assert_called_once()
    
    @patch('anthropic.Anthropic')
    def test_error_in_first_round_handled_gracefully(self, mock_anthropic):
        """Test error handling when first tool call fails"""
        mock_client = Mock()
        
        # Round 1: AI uses tool (will fail)
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        round1_response.content = [Mock(type="tool_use", name="search_course_content", id="t1", input={"query": "test"})]
        
        # Round 2: AI handles error and responds
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="I encountered an error searching for that information")]
        
        mock_client.messages.create.side_effect = [round1_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Tool manager raises exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")
        
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Search for something",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        assert result == "I encountered an error searching for that information"
        assert mock_client.messages.create.call_count == 2
        mock_tool_manager.execute_tool.assert_called_once()
    
    @patch('anthropic.Anthropic')
    def test_error_in_second_round_preserves_first_results(self, mock_anthropic):
        """Test that errors in second round don't lose first round results"""
        mock_client = Mock()
        
        # Round 1: Successful tool call
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        round1_response.content = [Mock(type="tool_use", name="get_course_outline", id="t1", input={"course_title": "MCP"})]
        
        # Round 2: Tool call that will fail
        round2_response = Mock()
        round2_response.stop_reason = "tool_use"
        round2_response.content = [Mock(type="tool_use", name="search_course_content", id="t2", input={"query": "advanced"})]
        
        # Round 3: AI synthesizes with partial results
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Based on the outline, MCP course has lessons but search failed")]
        
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        mock_anthropic.return_value = mock_client
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline: Lesson 1, 2, 3",  # First tool succeeds
            Exception("Search service unavailable")  # Second tool fails
        ]
        
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Tell me about MCP course and advanced topics",
            tools=[
                {"name": "get_course_outline"},
                {"name": "search_course_content"}
            ],
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        assert result == "Based on the outline, MCP course has lessons but search failed"
        assert mock_client.messages.create.call_count == 3  # 2 rounds + final
        assert mock_tool_manager.execute_tool.call_count == 2
    
    @patch('anthropic.Anthropic')
    def test_maximum_rounds_enforced(self, mock_anthropic):
        """Test that exactly 2 rounds are enforced as maximum"""
        mock_client = Mock()
        
        # All responses try to use tools (AI keeps wanting more tools)
        persistent_tool_response = Mock()
        persistent_tool_response.stop_reason = "tool_use"
        persistent_tool_response.content = [Mock(type="tool_use", name="search_course_content", id="t1", input={"query": "more"})]
        
        # Final forced response after max rounds
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Maximum tool rounds reached")]
        
        # AI tries tools 3 times but only 2 are allowed
        mock_client.messages.create.side_effect = [
            persistent_tool_response,  # Round 1
            persistent_tool_response,  # Round 2  
            final_response             # Final (forced after max rounds)
        ]
        mock_anthropic.return_value = mock_client
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Keep searching for more information",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        assert result == "Maximum tool rounds reached"
        assert mock_client.messages.create.call_count == 3  # 2 tool attempts + final
        assert mock_tool_manager.execute_tool.call_count == 2  # Only 2 rounds executed
    
    @patch('anthropic.Anthropic')
    def test_conversation_context_preserved_across_rounds(self, mock_anthropic):
        """Test that conversation context builds properly across tool rounds"""
        mock_client = Mock()
        
        # Two sequential tool rounds
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        round1_response.content = [Mock(type="tool_use", name="get_course_outline", id="t1", input={"course_title": "Python"})]
        
        round2_response = Mock()
        round2_response.stop_reason = "tool_use" 
        round2_response.content = [Mock(type="tool_use", name="search_course_content", id="t2", input={"query": "variables"})]
        
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Context preserved answer")]
        
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        mock_anthropic.return_value = mock_client
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Outline result", "Search result"]
        
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Tell me about Python course variables",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        assert result == "Context preserved answer"
        
        # Verify conversation context grows across calls
        calls = mock_client.messages.create.call_args_list
        assert len(calls) == 3
        
        # First call: just user message
        assert len(calls[0][1]["messages"]) == 1
        
        # Second call: user + assistant tool_use + tool_results
        assert len(calls[1][1]["messages"]) >= 3
        
        # Third call: all previous context + round 2 tool exchange
        assert len(calls[2][1]["messages"]) >= 5
    
    @patch('anthropic.Anthropic')
    def test_complex_course_comparison_workflow(self, mock_anthropic):
        """Test realistic sequential workflow for course comparison"""
        mock_client = Mock()
        
        # Round 1: Get course outline to find lesson 4 title
        tool1_block = Mock()
        tool1_block.type = "tool_use"
        tool1_block.name = "get_course_outline"
        tool1_block.id = "t1"
        tool1_block.input = {"course_title": "Course X"}
        
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        round1_response.content = [tool1_block]
        
        # Round 2: Search for courses with similar topic to lesson 4
        tool2_block = Mock()
        tool2_block.type = "tool_use"
        tool2_block.name = "search_course_content"
        tool2_block.id = "t2"
        tool2_block.input = {"query": "advanced neural networks"}
        
        round2_response = Mock()
        round2_response.stop_reason = "tool_use"
        round2_response.content = [tool2_block]
        
        # Final: Provide comparison
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="Course Y also covers neural networks in lesson 3")]
        
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        mock_anthropic.return_value = mock_client
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course X outline: Lesson 4: Advanced Neural Networks",
            "Found Course Y - Lesson 3: Neural Network Fundamentals"
        ]
        
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Search for a course that discusses the same topic as lesson 4 of Course X",
            tools=[
                {"name": "get_course_outline", "description": "Get course structure"},
                {"name": "search_course_content", "description": "Search course content"}
            ],
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        assert result == "Course Y also covers neural networks in lesson 3"
        assert mock_client.messages.create.call_count == 3
        
        # Verify logical tool sequence
        expected_calls = [
            call("get_course_outline", course_title="Course X"),
            call("search_course_content", query="advanced neural networks")
        ]
        mock_tool_manager.execute_tool.assert_has_calls(expected_calls)
    
    @patch('anthropic.Anthropic')
    def test_max_rounds_parameter_respected(self, mock_anthropic):
        """Test that custom max_rounds parameter is respected"""
        mock_client = Mock()
        
        # AI wants to keep using tools
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [Mock(type="tool_use", name="search_course_content", id="t1", input={"query": "test"})]
        
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock(text="One round only")]
        
        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic.return_value = mock_client
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Search for information",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
            max_rounds=1  # Limit to 1 round
        )
        
        assert result == "One round only"
        assert mock_client.messages.create.call_count == 2  # 1 tool round + final
        mock_tool_manager.execute_tool.assert_called_once()
    
    @patch('anthropic.Anthropic') 
    def test_no_tools_available_fallback(self, mock_anthropic):
        """Test behavior when tools requested but none available"""
        mock_client = Mock()
        
        direct_response = Mock()
        direct_response.stop_reason = "end_turn"
        direct_response.content = [Mock(text="Direct answer without tools")]
        
        mock_client.messages.create.return_value = direct_response
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Search for Python basics",
            tools=None,  # No tools available
            tool_manager=None,
            max_rounds=2
        )
        
        assert result == "Direct answer without tools"
        assert mock_client.messages.create.call_count == 1  # Single call only