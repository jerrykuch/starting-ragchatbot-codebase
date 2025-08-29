import pytest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock
from rag_system import RAGSystem


class TestRAGSystem:
    """Test RAGSystem integration and content query handling"""
    
    def test_init_with_config(self, test_config):
        """Test RAG system initialization"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            
            rag = RAGSystem(test_config)
            
            assert rag.config == test_config
            assert hasattr(rag, 'vector_store')
            assert hasattr(rag, 'ai_generator') 
            assert hasattr(rag, 'tool_manager')
            assert hasattr(rag, 'search_tool')
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_with_broken_max_results_config(self, mock_session, mock_ai, mock_doc, mock_vector, test_config):
        """Test query processing with MAX_RESULTS=0 (broken config)"""
        # Setup mocks
        mock_vector_instance = Mock()
        mock_vector.return_value = mock_vector_instance
        
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "No relevant content found."
        mock_ai.return_value = mock_ai_instance
        
        mock_session_instance = Mock()
        mock_session_instance.get_conversation_history.return_value = None
        mock_session.return_value = mock_session_instance
        
        # Create RAG system with broken config
        rag = RAGSystem(test_config)
        
        # Mock tool manager to simulate empty results from search
        rag.tool_manager = Mock()
        rag.tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search tool"}
        ]
        rag.tool_manager.get_last_sources.return_value = []  # No sources due to MAX_RESULTS=0
        
        # Execute query
        response, sources = rag.query("Tell me about Python basics")
        
        # Should get response but no sources due to broken config
        assert "No relevant content found" in response
        assert sources == []
        
        # Verify AI was called with tools
        mock_ai_instance.generate_response.assert_called_once()
        call_args = mock_ai_instance.generate_response.call_args
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_with_working_config(self, mock_session, mock_ai, mock_doc, mock_vector):
        """Test query processing with working MAX_RESULTS=5 config"""
        # Create working config
        class WorkingConfig:
            ANTHROPIC_API_KEY = "test-key"
            ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
            EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            CHUNK_SIZE = 800
            CHUNK_OVERLAP = 100
            MAX_RESULTS = 5  # Fixed config!
            MAX_HISTORY = 2
            MAX_TOOL_ROUNDS = 2
            CHROMA_PATH = "./test_chroma"
        
        working_config = WorkingConfig()
        
        # Setup mocks
        mock_vector_instance = Mock()
        mock_vector.return_value = mock_vector_instance
        
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Python is a programming language..."
        mock_ai.return_value = mock_ai_instance
        
        mock_session_instance = Mock()
        mock_session_instance.get_conversation_history.return_value = None
        mock_session.return_value = mock_session_instance
        
        # Create RAG system with working config
        rag = RAGSystem(working_config)
        
        # Mock tool manager to simulate successful search
        rag.tool_manager = Mock()
        rag.tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search tool"}
        ]
        rag.tool_manager.get_last_sources.return_value = ["Test Course - Lesson 1"]
        
        # Execute query
        response, sources = rag.query("Tell me about Python basics")
        
        # Should get response AND sources with working config
        assert response == "Python is a programming language..."
        assert sources == ["Test Course - Lesson 1"]
        
        # Verify sources were reset after retrieval
        rag.tool_manager.reset_sources.assert_called_once()
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor') 
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_session_management(self, mock_session, mock_ai, mock_doc, mock_vector, test_config):
        """Test conversation history and session management"""
        # Setup mocks
        mock_vector_instance = Mock()
        mock_vector.return_value = mock_vector_instance
        
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Session response"
        mock_ai.return_value = mock_ai_instance
        
        mock_session_instance = Mock()
        mock_session_instance.get_conversation_history.return_value = "Previous conversation"
        mock_session.return_value = mock_session_instance
        
        rag = RAGSystem(test_config)
        rag.tool_manager = Mock()
        rag.tool_manager.get_tool_definitions.return_value = []
        rag.tool_manager.get_last_sources.return_value = []
        
        # Execute query with session
        response, sources = rag.query("Follow up question", session_id="test_session")
        
        # Verify session history was retrieved and used
        mock_session_instance.get_conversation_history.assert_called_once_with("test_session")
        
        # Verify conversation was updated
        mock_session_instance.add_exchange.assert_called_once_with(
            "test_session", 
            "Follow up question", 
            "Session response"
        )
        
        # Verify history was passed to AI
        call_args = mock_ai_instance.generate_response.call_args
        assert call_args[1]["conversation_history"] == "Previous conversation"
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager') 
    def test_query_prompt_formatting(self, mock_session, mock_ai, mock_doc, mock_vector, test_config):
        """Test that user queries are properly formatted for AI"""
        # Setup mocks
        mock_vector_instance = Mock()
        mock_vector.return_value = mock_vector_instance
        
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Formatted response"
        mock_ai.return_value = mock_ai_instance
        
        mock_session_instance = Mock()
        mock_session_instance.get_conversation_history.return_value = None
        mock_session.return_value = mock_session_instance
        
        rag = RAGSystem(test_config)
        rag.tool_manager = Mock()
        rag.tool_manager.get_tool_definitions.return_value = []
        rag.tool_manager.get_last_sources.return_value = []
        
        # Execute query
        user_query = "What is machine learning?"
        response, sources = rag.query(user_query)
        
        # Verify query was formatted correctly for AI
        call_args = mock_ai_instance.generate_response.call_args
        expected_prompt = f"Answer this question about course materials: {user_query}"
        assert call_args[1]["query"] == expected_prompt
    
    def test_get_course_analytics(self, test_config):
        """Test course analytics retrieval"""
        with patch('rag_system.VectorStore') as mock_vector, \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            
            mock_vector_instance = Mock()
            mock_vector_instance.get_course_count.return_value = 4
            mock_vector_instance.get_existing_course_titles.return_value = [
                "Course 1", "Course 2", "Course 3", "Course 4"
            ]
            mock_vector.return_value = mock_vector_instance
            
            rag = RAGSystem(test_config)
            analytics = rag.get_course_analytics()
            
            assert analytics["total_courses"] == 4
            assert len(analytics["course_titles"]) == 4
            assert "Course 1" in analytics["course_titles"]


class TestRAGSystemRealIntegration:
    """Integration tests with real components to test the actual bug"""
    
    def test_real_integration_with_zero_max_results(self, temp_chroma_path, sample_course, sample_chunks):
        """Test real RAG system with MAX_RESULTS=0 to confirm bug"""
        # Create config with the bug
        class BuggyConfig:
            ANTHROPIC_API_KEY = "test-key"
            ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
            EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            CHUNK_SIZE = 800
            CHUNK_OVERLAP = 100
            MAX_RESULTS = 0  # The bug!
            MAX_HISTORY = 2
            MAX_TOOL_ROUNDS = 2
            CHROMA_PATH = temp_chroma_path
        
        with patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager') as mock_session:
            
            # Setup AI generator mock
            mock_ai_instance = Mock()
            mock_ai_instance.generate_response.return_value = "No relevant content found."
            mock_ai.return_value = mock_ai_instance
            
            # Setup session manager mock
            mock_session_instance = Mock()
            mock_session_instance.get_conversation_history.return_value = None
            mock_session.return_value = mock_session_instance
            
            # Create RAG system and add data
            rag = RAGSystem(BuggyConfig())
            rag.vector_store.add_course_metadata(sample_course)
            rag.vector_store.add_course_content(sample_chunks)
            
            # Execute query - should fail due to MAX_RESULTS=0
            response, sources = rag.query("Tell me about basic concepts")
            
            # Verify the bug causes empty sources
            assert sources == []  # No sources due to MAX_RESULTS=0 bug
            
            # Verify the search tool returns error due to MAX_RESULTS=0
            search_result = rag.search_tool.execute("basic concepts")
            assert "Search error: Number of requested results 0" in search_result
    
    def test_real_integration_with_fixed_max_results(self, temp_chroma_path, sample_course, sample_chunks):
        """Test real RAG system with MAX_RESULTS=5 to show fix works"""
        # Create config with fix
        class FixedConfig:
            ANTHROPIC_API_KEY = "test-key"
            ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
            EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            CHUNK_SIZE = 800
            CHUNK_OVERLAP = 100
            MAX_RESULTS = 5  # The fix!
            MAX_HISTORY = 2
            MAX_TOOL_ROUNDS = 2
            CHROMA_PATH = temp_chroma_path
        
        with patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager') as mock_session:
            
            # Setup AI generator mock
            mock_ai_instance = Mock()
            mock_ai_instance.generate_response.return_value = "Python basics include variables, functions..."
            mock_ai.return_value = mock_ai_instance
            
            # Setup session manager mock  
            mock_session_instance = Mock()
            mock_session_instance.get_conversation_history.return_value = None
            mock_session.return_value = mock_session_instance
            
            # Create RAG system and add data
            rag = RAGSystem(FixedConfig())
            rag.vector_store.add_course_metadata(sample_course)
            rag.vector_store.add_course_content(sample_chunks)
            
            # Execute query - should work with MAX_RESULTS=5
            response, sources = rag.query("Tell me about basic concepts")
            
            # Verify the fix allows proper functionality
            # (Sources will still be empty in this test due to mocked AI, 
            # but search tool should return results)
            search_result = rag.search_tool.execute("basic concepts")
            assert "No relevant content found" not in search_result
            assert "Test Course" in search_result


class TestRAGSystemDocumentProcessing:
    """Test document processing functionality"""
    
    def test_add_course_document_success(self, temp_chroma_path):
        """Test adding a single course document"""
        class TestConfig:
            ANTHROPIC_API_KEY = "test-key"
            ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
            EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            CHUNK_SIZE = 800
            CHUNK_OVERLAP = 100
            MAX_RESULTS = 5
            MAX_HISTORY = 2
            MAX_TOOL_ROUNDS = 2
            CHROMA_PATH = temp_chroma_path
        
        with patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager') as mock_session:
            
            mock_ai.return_value = Mock()
            mock_session.return_value = Mock()
            
            rag = RAGSystem(TestConfig())
            
            # Create a test document file
            test_file_content = """Course Title: Test Course
Course Instructor: Test Instructor

Lesson 1: Introduction
This is lesson 1 content about basics.
"""
            test_file_path = os.path.join(temp_chroma_path, "test_course.txt")
            os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
            with open(test_file_path, 'w') as f:
                f.write(test_file_content)
            
            # Add the document
            course, chunk_count = rag.add_course_document(test_file_path)
            
            # Verify course was processed
            assert course is not None
            assert course.title == "Test Course"
            assert course.instructor == "Test Instructor"
            assert chunk_count > 0
    
    def test_add_course_document_file_not_found(self, temp_chroma_path):
        """Test adding non-existent course document"""
        class TestConfig:
            ANTHROPIC_API_KEY = "test-key"
            ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
            EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            CHUNK_SIZE = 800
            CHUNK_OVERLAP = 100
            MAX_RESULTS = 5
            MAX_HISTORY = 2
            MAX_TOOL_ROUNDS = 2
            CHROMA_PATH = temp_chroma_path
        
        with patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager') as mock_session:
            
            mock_ai.return_value = Mock()
            mock_session.return_value = Mock()
            
            rag = RAGSystem(TestConfig())
            
            # Try to add non-existent file
            course, chunk_count = rag.add_course_document("/nonexistent/file.txt")
            
            # Should handle error gracefully
            assert course is None
            assert chunk_count == 0


class TestRAGQueryFlow:
    """Test the complete query flow that's currently failing"""
    
    @patch('anthropic.Anthropic')
    def test_end_to_end_query_flow_with_bug(self, mock_anthropic, temp_chroma_path, sample_course, sample_chunks):
        """Test complete flow from user query to response with MAX_RESULTS=0 bug"""
        # Create config with the bug
        class BuggyConfig:
            ANTHROPIC_API_KEY = "test-key" 
            ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
            EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            CHUNK_SIZE = 800
            CHUNK_OVERLAP = 100
            MAX_RESULTS = 0  # The bug causing failures!
            MAX_HISTORY = 2
            MAX_TOOL_ROUNDS = 2
            CHROMA_PATH = temp_chroma_path
        
        # Setup Anthropic client mock for tool calling flow
        mock_client = Mock()
        
        # First call: AI requests to use search tool
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.id = "tool_123"
        tool_use_block.input = {"query": "Python basics"}
        
        initial_response = Mock()
        initial_response.content = [tool_use_block]
        initial_response.stop_reason = "tool_use"
        
        # Second call: AI responds with "no content found" due to empty search results
        final_response = Mock()
        final_response.content = [Mock(text="I couldn't find any relevant content about that topic.")]
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Create RAG system and populate with data
        rag = RAGSystem(BuggyConfig())
        rag.vector_store.add_course_metadata(sample_course)
        rag.vector_store.add_course_content(sample_chunks)
        
        # Execute query that should find results but doesn't due to bug
        response, sources = rag.query("Tell me about Python basics", session_id="test_session")
        
        # Verify bug causes failure even with relevant content in database
        assert "couldn't find" in response or "No relevant content" in response
        assert sources == []  # No sources returned due to MAX_RESULTS=0
        
        # Verify that search tool was called but returned error
        # (This demonstrates the bug is in configuration, not in tool logic)
        search_result = rag.search_tool.execute("Python basics")
        assert "Search error: Number of requested results 0" in search_result
    
    @patch('anthropic.Anthropic')
    def test_end_to_end_query_flow_with_fix(self, mock_anthropic, temp_chroma_path, sample_course, sample_chunks):
        """Test complete flow shows fix works when MAX_RESULTS > 0"""
        # Create config with the fix
        class FixedConfig:
            ANTHROPIC_API_KEY = "test-key"
            ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
            EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            CHUNK_SIZE = 800
            CHUNK_OVERLAP = 100
            MAX_RESULTS = 5  # The fix!
            MAX_HISTORY = 2
            MAX_TOOL_ROUNDS = 2
            CHROMA_PATH = temp_chroma_path
        
        # Setup Anthropic client mock
        mock_client = Mock()
        
        # First call: AI requests to use search tool
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.id = "tool_123"
        tool_use_block.input = {"query": "basic concepts"}
        
        initial_response = Mock()
        initial_response.content = [tool_use_block]
        initial_response.stop_reason = "tool_use"
        
        # Second call: AI responds with content found
        final_response = Mock()
        final_response.content = [Mock(text="Based on the course content, basic concepts include...")]
        final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [initial_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Create RAG system and populate with data
        rag = RAGSystem(FixedConfig())
        rag.vector_store.add_course_metadata(sample_course)
        rag.vector_store.add_course_content(sample_chunks)
        
        # Execute query 
        response, sources = rag.query("Tell me about basic concepts")
        
        # Verify fix allows proper results
        assert "Based on the course content" in response
        
        # Verify search tool can now return results
        search_result = rag.search_tool.execute("basic concepts")
        assert "No relevant content found" not in search_result
        assert "Test Course" in search_result