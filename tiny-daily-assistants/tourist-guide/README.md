# Tourist Assistant

An interactive voice-enabled tourist guide that provides information about cities, landmarks, and destinations worldwide. This application uses OpenAI's GPT models for text generation and speech features for a natural conversation experience.

![Tourist Assistant Screenshot](travel.jpg)

## Features

- Text-based chat interface for asking questions about tourist destinations
- Voice input capability through microphone recording
- Audio responses using OpenAI's text-to-speech technology
- Clean, responsive user interface with Gradio

## Requirements

- Python 3.9+
- OpenAI API key

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the Application

Start the application by running:

```bash
python tourist-assistant.py
```

The interface will automatically open in your default web browser. If it doesn't, navigate to the URL shown in the terminal (typically http://127.0.0.1:7860/).

## Usage

1. Type your question about any tourist destination in the text box
2. Or click the microphone button and speak your question
3. The assistant will respond with text and spoken audio
4. Use the "Clear" button to start a new conversation

## Technologies Used

- OpenAI GPT-4o Mini for chat completions
- OpenAI Whisper for speech-to-text
- OpenAI TTS for text-to-speech
- Gradio for the web interface
- pydub for audio processing