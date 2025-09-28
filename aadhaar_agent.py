from pdf_processor import PDFProcessor
from vector_db import VectorDatabase
from openai_chat import OpenAIChat
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import os
from pathlib import Path

class AadhaarChatAgent:
    """Main conversational agent for Aadhaar-related questions"""
    
    def __init__(self, pdf_directory: str = "Supporting Documents"):
        self.console = Console()
        self.pdf_processor = PDFProcessor(pdf_directory)
        self.vector_db = VectorDatabase()
        self.chat = OpenAIChat()
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the agent by processing PDFs and setting up vector database"""
        if self.is_initialized:
            return
            
        self.console.print(Panel.fit("🚀 Initializing Aadhaar Chat Agent...", style="bold blue"))
        
        # Check if vector database already exists
        if self.vector_db.collection.count() == 0:
            self.console.print("📄 Processing PDF documents...")
            documents = self.pdf_processor.process_all_pdfs()
            
            if not documents:
                self.console.print("[red]No PDF documents found in the specified directory![/red]")
                return
            
            self.console.print(f"✅ Found {len(documents)} PDF documents")
            
            # Add documents to vector database
            self.vector_db.add_documents(documents)
        else:
            self.console.print("✅ Vector database already initialized")
        
        self.is_initialized = True
        self.console.print(Panel.fit("🎉 Agent ready! Ask me anything about Aadhaar.", style="bold green"))
    
    def chat_loop(self):
        """Main chat loop"""
        if not self.is_initialized:
            self.initialize()
        
        if not self.is_initialized:
            return
        
        self.console.print("\n[bold cyan]Aadhaar Chat Agent[/bold cyan]")
        self.console.print("Type 'quit', 'exit', or 'bye' to end the conversation")
        self.console.print("Type 'clear' to clear conversation history")
        self.console.print("Type 'help' for more information\n")
        
        while True:
            try:
                # Get user input
                user_input = input("🤔 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    self.console.print("\n👋 Goodbye! Thanks for using Aadhaar Chat Agent.")
                    break
                
                if user_input.lower() == 'clear':
                    self.chat.clear_history()
                    self.console.print("🧹 Conversation history cleared!")
                    continue
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                if not user_input:
                    continue
                
                # Search for relevant documents
                self.console.print("🔍 Searching relevant documents...")
                relevant_docs = self.vector_db.search(user_input, n_results=3)
                
                # Generate response
                self.console.print("💭 Generating response...")
                response = self.chat.generate_response(user_input, relevant_docs)
                
                # Display response
                self.console.print(Panel(response, title="🤖 Aadhaar Agent", border_style="green"))
                
            except KeyboardInterrupt:
                self.console.print("\n\n👋 Goodbye! Thanks for using Aadhaar Chat Agent.")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
📚 Aadhaar Chat Agent Help

This agent can help you with questions about:
• Aadhaar enrollment process
• Document requirements
• Update procedures
• Supporting documents
• General Aadhaar information

Commands:
• 'quit', 'exit', 'bye' - End conversation
• 'clear' - Clear conversation history
• 'help' - Show this help message

The agent uses official Aadhaar documents to provide accurate information.
        """
        self.console.print(Panel(help_text, title="Help", border_style="blue"))
    
    def ask_question(self, question: str) -> str:
        """Ask a single question and get response (for programmatic use)"""
        if not self.is_initialized:
            self.initialize()
        
        if not self.is_initialized:
            return "Agent initialization failed"
        
        relevant_docs = self.vector_db.search(question, n_results=3)
        response = self.chat.generate_response(question, relevant_docs)
        return response
