import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test CourseSearchTool.execute method with different scenarios"""
    
    def test_get_tool_definition(self, course_search_tool):
        """Test that tool definition is correctly formatted"""
        definition = course_search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["properties"]["query"]["type"] == "string"
        assert "query" in definition["input_schema"]["required"]
    
    def test_execute_with_empty_results_zero_max_results(self, mock_vector_store):
        """Test execute when MAX_RESULTS=0 causes empty results (current bug)"""
        # Configure mock to return empty results (simulating MAX_RESULTS=0 bug)
        mock_vector_store.search.return_value = SearchResults.empty("")
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert "No relevant content found" in result
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=None
        )
    
    def test_execute_with_successful_results(self, mock_vector_store):
        """Test execute with successful search results"""
        # Configure mock to return valid results
        mock_vector_store.search.return_value = SearchResults(
            documents=["This is test content about basic concepts"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert "[Test Course - Lesson 1]" in result
        assert "This is test content about basic concepts" in result
        assert len(tool.last_sources) == 1
        assert "Test Course - Lesson 1|https://example.com/lesson1" in tool.last_sources
    
    def test_execute_with_course_filter(self, mock_vector_store):
        """Test execute with course name filtering"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Filtered content"],
            metadata=[{"course_title": "Specific Course", "lesson_number": 2}],
            distances=[0.2]
        )
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Specific Course")
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Specific Course", 
            lesson_number=None
        )
        assert "[Specific Course - Lesson 2]" in result
    
    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Test execute with lesson number filtering"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Lesson-specific content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 3}],
            distances=[0.15]
        )
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", lesson_number=3)
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=3
        )
        assert "[Test Course - Lesson 3]" in result
    
    def test_execute_with_search_error(self, mock_vector_store):
        """Test execute when vector store returns error"""
        mock_vector_store.search.return_value = SearchResults.empty("Database connection failed")
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert result == "Database connection failed"
    
    def test_execute_with_course_and_lesson_filter(self, mock_vector_store):
        """Test execute with both course and lesson filters"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Highly specific content"],
            metadata=[{"course_title": "Advanced Course", "lesson_number": 5}],
            distances=[0.05]
        )
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Advanced Course", lesson_number=5)
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Advanced Course",
            lesson_number=5
        )
        assert "[Advanced Course - Lesson 5]" in result
    
    def test_sources_tracking(self, mock_vector_store):
        """Test that sources are properly tracked for UI display"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/lesson1",
            "https://example.com/lesson2"
        ]
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert len(tool.last_sources) == 2
        assert "Course A - Lesson 1|https://example.com/lesson1" in tool.last_sources
        assert "Course B - Lesson 2|https://example.com/lesson2" in tool.last_sources


class TestCourseOutlineTool:
    """Test CourseOutlineTool functionality"""
    
    def test_get_tool_definition(self):
        """Test that outline tool definition is correctly formatted"""
        mock_store = Mock()
        tool = CourseOutlineTool(mock_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "get_course_outline"
        assert "course_title" in definition["input_schema"]["required"]
    
    def test_execute_course_not_found(self):
        """Test execute when course is not found"""
        mock_store = Mock()
        mock_store._resolve_course_name.return_value = None
        
        tool = CourseOutlineTool(mock_store)
        result = tool.execute("Nonexistent Course")
        
        assert "No course found matching 'Nonexistent Course'" in result
    
    def test_execute_successful_outline(self):
        """Test execute with successful course outline retrieval"""
        mock_store = Mock()
        mock_store._resolve_course_name.return_value = "Test Course"
        
        # Mock course catalog response
        mock_store.course_catalog.get.return_value = {
            'metadatas': [{
                'title': 'Test Course',
                'instructor': 'Test Instructor',
                'course_link': 'https://example.com/course',
                'lessons_json': '[{"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "https://example.com/lesson1"}]'
            }]
        }
        
        tool = CourseOutlineTool(mock_store)
        result = tool.execute("Test Course")
        
        assert "**Course:** Test Course" in result
        assert "**Instructor:** Test Instructor" in result
        assert "**Course Link:** https://example.com/course" in result
        assert "1. Intro (https://example.com/lesson1)" in result


class TestToolManager:
    """Test ToolManager functionality"""
    
    def test_register_tool(self):
        """Test tool registration"""
        manager = ToolManager()
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"name": "test_tool"}
        
        manager.register_tool(mock_tool)
        
        assert "test_tool" in manager.tools
    
    def test_get_tool_definitions(self):
        """Test getting all tool definitions"""
        manager = ToolManager()
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"name": "test_tool", "description": "Test"}
        
        manager.register_tool(mock_tool)
        definitions = manager.get_tool_definitions()
        
        assert len(definitions) == 1
        assert definitions[0]["name"] == "test_tool"
    
    def test_execute_tool(self):
        """Test tool execution"""
        manager = ToolManager()
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"name": "test_tool"}
        mock_tool.execute.return_value = "Tool executed successfully"
        
        manager.register_tool(mock_tool)
        result = manager.execute_tool("test_tool", query="test")
        
        assert result == "Tool executed successfully"
        mock_tool.execute.assert_called_once_with(query="test")
    
    def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist"""
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool")
        
        assert "Tool 'nonexistent_tool' not found" in result
    
    def test_get_last_sources(self):
        """Test retrieving sources from last search"""
        manager = ToolManager()
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"name": "search_tool"}
        mock_tool.last_sources = ["Source 1", "Source 2"]
        
        manager.register_tool(mock_tool)
        sources = manager.get_last_sources()
        
        assert sources == ["Source 1", "Source 2"]
    
    def test_reset_sources(self):
        """Test resetting sources from all tools"""
        manager = ToolManager()
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"name": "search_tool"}
        mock_tool.last_sources = ["Source 1"]
        
        manager.register_tool(mock_tool)
        manager.reset_sources()
        
        assert mock_tool.last_sources == []


class TestRealVectorStoreIntegration:
    """Test CourseSearchTool with real vector store to expose MAX_RESULTS bug"""
    
    def test_zero_max_results_bug(self, vector_store_with_zero_results, sample_course, sample_chunks):
        """Test that MAX_RESULTS=0 causes search to return no results"""
        store = vector_store_with_zero_results
        
        # Add sample data
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_chunks)
        
        # Create tool and search
        tool = CourseSearchTool(store)
        result = tool.execute("basic concepts")
        
        # Should return error due to MAX_RESULTS=0 bug
        assert "Search error: Number of requested results 0" in result
    
    def test_normal_max_results_works(self, vector_store_with_normal_results, sample_course, sample_chunks):
        """Test that MAX_RESULTS=5 allows search to return results"""
        store = vector_store_with_normal_results
        
        # Add sample data
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_chunks)
        
        # Create tool and search
        tool = CourseSearchTool(store)
        result = tool.execute("basic concepts")
        
        # Should return results when MAX_RESULTS > 0
        assert "No relevant content found" not in result
        assert "Test Course" in result
        assert "basic concepts" in result or "lesson 1" in result.lower()