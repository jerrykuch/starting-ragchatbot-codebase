# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- **Quick start**: `./run.sh` (creates necessary directories and starts server)
- **Manual start**: `cd backend && uv run uvicorn app:app --reload --port 8000`
- **Install dependencies**: `uv sync`

### Environment Setup
- Create `.env` file with `ANTHROPIC_API_KEY=your_key_here`
- Application runs on http://localhost:8000 (web UI and API)
- API docs available at http://localhost:8000/docs

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) chatbot** for course materials with a FastAPI backend and vanilla JavaScript frontend.

### Core RAG Flow
1. **Document Processing**: Course files → sentence-based chunks (800 chars, 100 overlap)
2. **Vector Storage**: ChromaDB with dual collections (`course_catalog` + `course_content`)
3. **Query Processing**: Claude API with tool-based semantic search
4. **Response Generation**: AI synthesis with source attribution

### Key Components

**RAGSystem** (`rag_system.py`): Central orchestrator that coordinates all components
- Manages document loading from `docs/` folder
- Orchestrates query processing through AI + tools
- Handles session management and source tracking

**DocumentProcessor** (`document_processor.py`): Transforms course documents into searchable chunks
- Parses course metadata (title, instructor, lessons)
- Implements intelligent sentence-based chunking with overlap
- Adds contextual prefixes ("Course X Lesson Y content: ...")

**VectorStore** (`vector_store.py`): ChromaDB integration with two collections
- `course_catalog`: Course metadata for name resolution
- `course_content`: Text chunks with embeddings for semantic search
- Smart course name matching and lesson filtering

**AIGenerator** (`ai_generator.py`): Claude API integration with tool calling
- Two-phase conversation: initial call → tool execution → final response
- System prompt optimized for educational content
- Tool result processing and response synthesis

**Search Tools** (`search_tools.py`): Tool framework for AI-driven search
- `CourseSearchTool`: Semantic search with course/lesson filtering
- `ToolManager`: Registers and executes tools, tracks sources for UI

### Data Models
- **Course**: Title (unique ID), instructor, lessons, links
- **Lesson**: Number, title, optional link
- **CourseChunk**: Content, course title, lesson number, chunk index

### Session Management
- In-memory conversation history (configurable limit)
- Session-based continuity for multi-turn conversations
- Automatic session creation if not provided

## Document Format Expected

Course files should follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 1: [lesson title]
Lesson Link: [optional url]
[lesson content...]

Lesson 2: [lesson title]
[lesson content...]
```

## Configuration

**Key settings in `config.py`:**
- `CHUNK_SIZE: 800` - Maximum characters per chunk
- `CHUNK_OVERLAP: 100` - Character overlap between chunks  
- `MAX_RESULTS: 5` - Search results returned
- `MAX_HISTORY: 2` - Conversation exchanges remembered

## API Endpoints

- `POST /api/query`: Process user queries with RAG pipeline
- `GET /api/courses`: Get course statistics and titles

## Frontend Integration

Static files served from `/` with no-cache headers for development. Frontend communicates via JSON API and displays markdown-formatted responses with collapsible source sections.

## ChromaDB Storage

Vector database persisted in `backend/chroma_db/` directory. Course documents auto-loaded on startup from `docs/` folder. Supports incremental loading (skips existing courses).
- use uv to run python files or add any dependencies