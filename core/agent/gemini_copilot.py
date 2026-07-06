import os
import json
from google import genai
from google.genai import types

def get_copilot_response(user_query: str, mission_context: dict, chat_history: list) -> str:
    """
    Sends the user query along with mission context to Gemini and returns the response.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY environment variable is not set. Please set it to use the AI Co-Pilot."
    
    # Initialize the official Google GenAI client
    client = genai.Client(api_key=api_key)
    
    # Build the system prompt
    system_instruction = (
        "You are an expert Drone Mission Planning AI Co-Pilot. "
        "You help operators analyze flight paths, understand risk calculations, and interpret physics simulation telemetry.\n\n"
        "Here is the current state of the drone mission:\n"
        f"```json\n{json.dumps(mission_context, indent=2)}\n```\n\n"
        "Use this context to give highly specific, technical, but easy-to-understand advice. "
        "If the user asks about battery drain, refer to the simulation summary (if available). "
        "If they ask about wind, refer to the drone parameters. "
        "Always be concise and professional."
    )
    
    # Format chat history for the new SDK
    contents = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )
    
    # Add the current user query
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_query)]
        )
    )
    
    # Configure generation
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.4,
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=config
        )
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {e}"
