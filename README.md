# Aadhaar Chat Agent

A terminal-based conversational agent that leverages Aadhaar PDF documents to answer user questions using OpenAI LLM and BGE-M3 embeddings with vector database for context retention.

## Features

- 📄 **PDF Processing**: Automatically extracts text from Aadhaar-related PDFs
- 🔍 **Vector Search**: Uses BGE-M3 embeddings for semantic document search
- 💬 **Conversational AI**: OpenAI GPT integration with context retention
- 🎨 **Rich Terminal UI**: Beautiful terminal interface with Rich library
- 🧠 **Context Memory**: Maintains conversation history during the session

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get OpenAI API Key

- Visit: https://platform.openai.com/api-keys
- Create a new API key
- The agent will prompt for it when needed (no configuration files required)

### 3. Ensure PDF Files

Make sure your `Supporting Documents` folder contains the Aadhaar PDF files:
- Aadhaar_Enrolment__and__Update__-__English.pdf
- AadhaarHandbook2021.pdf
- List_of_Supporting_Document_for_Aadhaar_Enrolment_and_Update.pdf

## Usage

### Interactive Chat Mode

```bash
python main.py chat
```

This starts an interactive chat session where you can ask questions about Aadhaar processes, document requirements, etc.

### Single Question Mode

```bash
python main.py ask "What documents are required for Aadhaar enrollment?"
```

### Setup Instructions

```bash
python main.py setup
```

## Commands During Chat

- `quit`, `exit`, `bye` - End the conversation
- `clear` - Clear conversation history
- `help` - Show help information

## How It Works

1. **PDF Processing**: Extracts text from all PDFs in the Supporting Documents folder
2. **Text Chunking**: Splits documents into manageable chunks with overlap
3. **Vector Database**: Creates embeddings using BGE-M3 model and stores in ChromaDB
4. **Semantic Search**: Searches for relevant document chunks based on user queries
5. **LLM Integration**: Uses OpenAI GPT with retrieved context to generate responses
6. **Context Retention**: Maintains conversation history for better context understanding

## Project Structure

```
aadhar/
├── main.py                 # Main terminal interface
├── aadhaar_agent.py        # Core conversational agent
├── pdf_processor.py        # PDF text extraction
├── vector_db.py           # Vector database operations
├── openai_chat.py         # OpenAI LLM integration
├── requirements.txt       # Python dependencies
├── env_example.txt        # Environment variables example
└── Supporting Documents/  # PDF files directory
```

## Dependencies

- `openai` - OpenAI API client
- `sentence-transformers` - BGE-M3 embedding model
- `chromadb` - Vector database
- `pypdf2` - PDF text extraction
- `python-dotenv` - Environment variables
- `rich` - Terminal UI
- `typer` - CLI framework

## Notes

- **Interactive Setup**: The agent will prompt for your OpenAI API key when needed
- **No Configuration Files**: No need to create .env files - just enter your API key when prompted
- **Automatic Processing**: The vector database is created automatically on first run
- **Fast Subsequent Runs**: Subsequent runs are faster as the database is cached
- **Context Retention**: The agent maintains conversation context throughout the session
- **Document-Based**: All responses are based on the provided Aadhaar documents
