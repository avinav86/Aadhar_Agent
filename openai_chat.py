"""
OpenAI Chat Integration Module for Aadhaar Chat Agent

This module provides sophisticated chat capabilities using OpenAI's GPT models
with enhanced memory management and context retention. It integrates document
context from the vector database with conversation history to provide coherent,
contextual responses.

Key Features:
- OpenAI GPT-3.5-turbo integration
- Enhanced conversation memory (20 exchanges + summaries)
- Automatic conversation summarization
- Strict document adherence with professional responses
- Context-aware response generation
- Memory management and cleanup

Technical Implementation:
- Conversation history: Last 20 exchanges (40 messages)
- Summary generation: Every 20 exchanges using GPT-3.5-turbo
- Context layers: System message → Summary → History → Current query
- Response filtering: Strict adherence to provided documents

Author: Avinav Mishra
Repository: https://github.com/avinav86/Aadhar_Agent
"""

import openai
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables from config.env file
# This enables automatic API key loading from the configuration file
load_dotenv('config.env')

class OpenAIChat:
    """
    Handles OpenAI LLM interactions with enhanced memory and context management.
    
    This class provides sophisticated chat capabilities that combine:
    - Document context from vector database searches
    - Conversation history with automatic summarization
    - Strict adherence to provided Aadhaar documents
    - Professional response formatting
    
    The system maintains conversation context through multiple layers:
    1. System instructions for behavior and constraints
    2. Conversation summaries for long-term context
    3. Recent conversation history for immediate context
    4. Current query with relevant document context
    
    Attributes:
        client (openai.OpenAI): OpenAI API client instance
        conversation_history (List[Dict]): Recent conversation messages
        conversation_summary (str): Summary of older conversation context
        topic_context (Dict): Additional context tracking for topics
    """
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.conversation_history = []
        self.conversation_summary = ""
        self.topic_context = {}
        
    def generate_response(self, user_query: str, context_documents: List[Dict]) -> str:
        """Generate response using OpenAI with context from vector search"""
        
        # Prepare context from retrieved documents
        context = self._prepare_context(context_documents)
        
        # Create system message
        system_message = """You are a specialized Aadhaar assistant that ONLY answers questions based on the provided official Aadhaar documents.

STRICT RULES:
1. ONLY use information from the provided Aadhaar documents
2. If information is NOT in the provided documents, respond with: "Information unavailable at the moment."
3. Do NOT provide any external knowledge or general information
4. Do NOT answer questions unrelated to Aadhaar processes
5. Always cite the specific document source when providing information

You have access to:
1. Official Aadhaar documents (provided as context)
2. Conversation history (previous questions and answers in this chat)

For questions about the conversation itself (like "when did we discuss X?" or "what did you say about Y?"), refer to the conversation history.

Stay strictly within the bounds of the provided Aadhaar documents."""

        # Prepare messages for chat completion
        messages = [{"role": "system", "content": system_message}]
        
        # Add conversation summary if available
        if self.conversation_summary:
            messages.append({"role": "system", "content": f"CONVERSATION SUMMARY: {self.conversation_summary}"})
        
        # Add conversation history (increased from 10 to 20 exchanges)
        for msg in self.conversation_history[-20:]:  # Keep last 20 exchanges (40 messages)
            messages.append(msg)
        
        # Add current query with context
        user_message = f"""RELEVANT DOCUMENT CONTEXT:
{context}

CURRENT QUESTION: {user_query}

IMPORTANT: Only answer if the information is available in the document context above. If not available, respond with "Information unavailable at the moment." Do not provide any external knowledge."""
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            assistant_response = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_query})
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            # Update conversation summary periodically
            self._update_conversation_summary()
            
            return assistant_response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _prepare_context(self, documents: List[Dict]) -> str:
        """Prepare context string from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc["metadata"].get("filename", "Unknown")
            content = doc["content"]
            context_parts.append(f"Source {i} ({source}):\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _update_conversation_summary(self):
        """Update conversation summary every 10 exchanges"""
        if len(self.conversation_history) % 20 == 0 and len(self.conversation_history) > 0:
            try:
                # Create a summary of the conversation so far
                recent_exchanges = self.conversation_history[-20:]
                conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_exchanges])
                
                summary_prompt = f"""Summarize the key topics and information discussed in this Aadhaar-related conversation. Focus on:
1. Main topics discussed
2. Key information provided
3. User's specific needs or questions
4. Important details that should be remembered

Conversation:
{conversation_text}

Provide a concise summary:"""
                
                summary_response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=200,
                    temperature=0.3
                )
                
                self.conversation_summary = summary_response.choices[0].message.content
                
            except Exception as e:
                print(f"Warning: Could not update conversation summary: {e}")
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.conversation_summary = ""
        self.topic_context = {}
