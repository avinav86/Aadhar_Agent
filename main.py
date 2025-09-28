#!/usr/bin/env python3
"""
Aadhaar Chat Agent - Terminal-based conversational agent for Aadhaar questions
"""

import typer
from rich.console import Console
from rich.panel import Panel
from aadhaar_agent import AadhaarChatAgent
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from config.env file
load_dotenv('config.env')

app = typer.Typer()
console = Console()

@app.command()
def chat():
    """Start the interactive chat session"""
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
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
        
        # Still allow manual entry as fallback
        console.print("\n[dim]Or enter it manually now (will not be saved):[/dim]")
        api_key = typer.prompt("Please enter your OpenAI API key", hide_input=True)
        
        if not api_key:
            console.print("[red]❌ No API key provided. Exiting.[/red]")
            return
        
        # Set the environment variable for this session
        os.environ["OPENAI_API_KEY"] = api_key
        console.print("[green]✅ API key set for this session![/green]")
        console.print("[dim]💡 Tip: Add it to config.env for permanent storage[/dim]")
    
    # Check if PDF directory exists
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
    
    # Start the chat agent
    try:
        agent = AadhaarChatAgent(pdf_dir)
        agent.chat_loop()
    except Exception as e:
        console.print(f"[red]Error starting agent: {str(e)}[/red]")

@app.command()
def ask(question: str):
    """Ask a single question and get response"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
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
        
        # Still allow manual entry as fallback
        console.print("\n[dim]Or enter it manually now (will not be saved):[/dim]")
        api_key = typer.prompt("Please enter your OpenAI API key", hide_input=True)
        
        if not api_key:
            console.print("[red]❌ No API key provided. Exiting.[/red]")
            return
        
        # Set the environment variable for this session
        os.environ["OPENAI_API_KEY"] = api_key
        console.print("[green]✅ API key set for this session![/green]")
        console.print("[dim]💡 Tip: Add it to config.env for permanent storage[/dim]")
    
    try:
        agent = AadhaarChatAgent("Supporting Documents")
        response = agent.ask_question(question)
        console.print(Panel(response, title="Response", border_style="green"))
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@app.command()
def setup():
    """Setup instructions"""
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
    console.print(Panel(setup_text, title="Setup Instructions", border_style="blue"))

if __name__ == "__main__":
    app()
