# Personal AI Research Assistant

A RAG-based AI assistant that can answer questions about research papers in the knowledge base.

## Features

- Uses Retrieval-Augmented Generation (RAG) to provide accurate, contextual answers
- Conversations maintain history for more coherent interactions
- Easy-to-use Gradio web interface
- Searches across PDF papers in the knowledge base

## Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd personalAssistant
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Place your research papers (PDFs) in the `knowledge-base` folder.

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. Open your browser at the URL displayed in the terminal.

3. Enter your OpenAI API key in the provided field and click "Set API Key".

4. Start asking questions about the papers in your knowledge base!

## How it Works

1. PDF papers are loaded from the knowledge-base directory
2. Documents are split into chunks and converted to vector embeddings
3. When you ask a question, the system:
   - Retrieves relevant document chunks from the vector database
   - Uses OpenAI's language model to generate an answer based on the retrieved context
   - Maintains conversation history for follow-up questions

## Environment Variables

You can also set your OpenAI API key in a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```