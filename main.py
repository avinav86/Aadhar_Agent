#!/usr/bin/env python3
"""
Aadhaar Chat Agent - Terminal-based conversational agent for Aadhaar questions

This is the main entry point for the Aadhaar Chat Agent application.
It provides a command-line interface using Typer for:
1. Interactive chat sessions
2. Single question queries
3. Setup instructions

The application uses BGE embeddings, ChromaDB vector database, and OpenAI GPT
to provide accurate answers based on official Aadhaar documents.

Author: Avinav Mishra
Repository: https://github.com/avinav86/Aadhar_Agent
"""

import typer
from rich.console import Console
from rich.panel import Panel
from aadhaar_agent import AadhaarChatAgent
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from config.env file
# This allows users to store their OpenAI API key securely
load_dotenv('config.env')

# Initialize Typer app for CLI commands
app = typer.Typer()
# Initialize Rich console for beautiful terminal output
console = Console()

@app.command()
def chat():
    """
    Start the interactive chat session with the Aadhaar agent.
    
    This command launches the main conversational interface where users can:
    - Ask questions about Aadhaar processes
    - Get document requirements
    - Learn about enrollment procedures
    - Receive contextual follow-up responses
    """
    # Check for OpenAI API key in environment variables
    # First check if it's loaded from config.env or set manually
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If no API key found or it's still the placeholder value
    if not api_key or api_key == "your_openai_api_key_here":
        # Display helpful instructions for setting up the API key
        console.print(Panel.fit(
            "🔑 OpenAI API Key Required\n\n"
            "Please add your API key to 'config.env' file:\n"
            "1. Open config.env file\n"
            "2. Replace 'your_openai_api_key_here' with your actual API key\n"
            "3. Save the file and run again\n\n"
            "Get your API key from: https://platform.openai.com/api-keys",
            title="Configuration Required",
            border_style="yellow"
        ))
        
        # Provide fallback option for manual entry (not saved)
        console.print("\n[dim]Or enter it manually now (will not be saved):[/dim]")
        api_key = typer.prompt("Please enter your OpenAI API key", hide_input=True)
        
        # Exit if no API key provided
        if not api_key:
            console.print("[red]❌ No API key provided. Exiting.[/red]")
            return
        
        # Set the environment variable for this session only
        os.environ["OPENAI_API_KEY"] = api_key
        console.print("[green]✅ API key set for this session![/green]")
        console.print("[dim]💡 Tip: Add it to config.env for permanent storage[/dim]")
    
    # Verify that the Supporting Documents directory exists
    # This directory contains the Aadhaar PDF files for the knowledge base
    pdf_dir = "Supporting Documents"
    if not Path(pdf_dir).exists():
        console.print(Panel.fit(
            f"❌ PDF directory '{pdf_dir}' not found!\n\n"
            "Please ensure the Supporting Documents folder exists\n"
            "and contains the Aadhaar PDF files.",
            title="Directory Not Found",
            border_style="red"
        ))
        return
    
    # Initialize and start the chat agent
    try:
        # Create the main agent instance with the PDF directory
        agent = AadhaarChatAgent(pdf_dir)
        # Start the interactive chat loop
        agent.chat_loop()
    except Exception as e:
        # Handle any errors during agent initialization or execution
        console.print(f"[red]Error starting agent: {str(e)}[/red]")

@app.command()
def ask(question: str):
    """
    Ask a single question and get an immediate response.
    
    This command is useful for:
    - Quick queries without starting a full chat session
    - Scripting and automation
    - Testing specific questions
    
    Args:
        question (str): The Aadhaar-related question to ask
        
    Example:
        python main.py ask "What documents are required for enrollment?"
    """
    # Check for OpenAI API key (same logic as chat command)
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Handle missing or placeholder API key
    if not api_key or api_key == "your_openai_api_key_here":
        # Display configuration instructions
        console.print(Panel.fit(
            "🔑 OpenAI API Key Required\n\n"
            "Please add your API key to 'config.env' file:\n"
            "1. Open config.env file\n"
            "2. Replace 'your_openai_api_key_here' with your actual API key\n"
            "3. Save the file and run again\n\n"
            "Get your API key from: https://platform.openai.com/api-keys",
            title="Configuration Required",
            border_style="yellow"
        ))
        
        # Provide manual entry option as fallback
        console.print("\n[dim]Or enter it manually now (will not be saved):[/dim]")
        api_key = typer.prompt("Please enter your OpenAI API key", hide_input=True)
        
        # Exit if no API key provided
        if not api_key:
            console.print("[red]❌ No API key provided. Exiting.[/red]")
            return
        
        # Set API key for this session only
        os.environ["OPENAI_API_KEY"] = api_key
        console.print("[green]✅ API key set for this session![/green]")
        console.print("[dim]💡 Tip: Add it to config.env for permanent storage[/dim]")
    
    # Process the question and display response
    try:
        # Initialize the agent with the Supporting Documents directory
        agent = AadhaarChatAgent("Supporting Documents")
        # Get response for the single question
        response = agent.ask_question(question)
        # Display the response in a styled panel
        console.print(Panel(response, title="Response", border_style="green"))
    except Exception as e:
        # Handle any errors during processing
        console.print(f"[red]Error: {str(e)}[/red]")

@app.command()
def setup():
    """
    Display comprehensive setup instructions for the Aadhaar Chat Agent.
    
    This command provides step-by-step guidance for:
    - Installing required dependencies
    - Configuring the OpenAI API key
    - Setting up the document directory
    - Running the application
    """
    # Comprehensive setup instructions with emojis and clear steps
    setup_text = """
🔧 Aadhaar Chat Agent Setup

1. Install dependencies:
   pip install -r requirements.txt

2. Configure OpenAI API key:
   - Visit: https://platform.openai.com/api-keys
   - Create a new API key
   - Open 'config.env' file
   - Replace 'your_openai_api_key_here' with your actual API key
   - Save the file

3. Ensure PDF files are in 'Supporting Documents' folder

4. Run the agent:
   python main.py chat

The agent will:
- Automatically load your API key from config.env
- Process PDFs and create a vector database on first run
- Cache the database for faster subsequent runs
- Maintain conversation context throughout the session

💡 Tip: If you don't set up config.env, the agent will still work
    by prompting for your API key each time (but won't save it)
    """
    # Display the setup instructions in a styled panel
    console.print(Panel(setup_text, title="Setup Instructions", border_style="blue"))

# Entry point for the application when run directly
if __name__ == "__main__":
    # Start the Typer application with all registered commands
    app()
