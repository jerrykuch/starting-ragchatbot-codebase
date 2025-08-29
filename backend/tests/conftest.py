import pytest
import tempfile
import os
import sys
import shutil
from unittest.mock import Mock, MagicMock

# Add backend to path so we can import modules
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem


@pytest.fixture
def temp_chroma_path():
    """Create temporary directory for ChromaDB testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Test Course",
        course_link="https://example.com/test-course",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=1, title="Introduction", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Advanced Topics", lesson_link="https://example.com/lesson2")
        ]
    )


@pytest.fixture
def sample_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is lesson 1 content about basic concepts",
            course_title="Test Course",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="This is lesson 1 continued content about fundamentals",
            course_title="Test Course", 
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="This is lesson 2 content about advanced topics and techniques",
            course_title="Test Course",
            lesson_number=2,
            chunk_index=2
        )
    ]


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock_store = Mock(spec=VectorStore)
    
    # Setup default search behavior
    mock_store.search.return_value = SearchResults(
        documents=["Sample search result content"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.1]
    )
    
    mock_store._resolve_course_name.return_value = "Test Course"
    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"
    
    return mock_store


@pytest.fixture 
def vector_store_with_zero_results(temp_chroma_path):
    """Create real vector store configured with MAX_RESULTS=0 (broken config)"""
    return VectorStore(
        chroma_path=temp_chroma_path,
        embedding_model="all-MiniLM-L6-v2",
        max_results=0  # This is the bug!
    )


@pytest.fixture
def vector_store_with_normal_results(temp_chroma_path):
    """Create real vector store configured with MAX_RESULTS=5 (fixed config)"""
    return VectorStore(
        chroma_path=temp_chroma_path,
        embedding_model="all-MiniLM-L6-v2", 
        max_results=5  # This is the fix!
    )


@pytest.fixture
def populated_vector_store(vector_store_with_normal_results, sample_course, sample_chunks):
    """Vector store with sample data loaded"""
    store = vector_store_with_normal_results
    store.add_course_metadata(sample_course)
    store.add_course_content(sample_chunks)
    return store


@pytest.fixture
def course_search_tool(mock_vector_store):
    """Create CourseSearchTool with mock vector store"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def mock_anthropic_client():
    """Create mock Anthropic client for testing"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text="Test response")]
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_tool_manager():
    """Create mock tool manager for testing"""
    mock_manager = Mock(spec=ToolManager)
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
        }
    ]
    mock_manager.execute_tool.return_value = "Mock tool result"
    mock_manager.get_last_sources.return_value = ["Test Course - Lesson 1"]
    return mock_manager


@pytest.fixture
def test_config():
    """Create test configuration"""
    class TestConfig:
        ANTHROPIC_API_KEY = "test-key"
        ANTHROPIC_MODEL = "claude-sonnet-4-20250514" 
        EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        CHUNK_SIZE = 800
        CHUNK_OVERLAP = 100
        MAX_RESULTS = 0  # Test with broken config initially
        MAX_HISTORY = 2
        MAX_TOOL_ROUNDS = 2
        CHROMA_PATH = "./test_chroma_db"
    
    return TestConfig()