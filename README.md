# Chat with PDFs

Chat with PDFs is a Streamlit-based web application that allows users to upload PDF documents and interactively ask questions about the content within those documents. The app utilizes Google Generative AI for advanced question-answering capabilities, enabling users to receive context-aware answers derived from their uploaded PDF files.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [How It Works](#how-it-works)

## Features

- Upload multiple PDF files for processing.
- Ask questions about the content of the uploaded PDFs.
- Context-aware responses powered by Google Generative AI.
- User-friendly interface built with Streamlit.

## Technologies Used

- [Python](https://www.python.org/) - Programming language.
- [Streamlit](https://streamlit.io/) - Framework for building interactive web applications.
- [PyPDF2](https://pypi.org/project/PyPDF2/) - Library for reading PDF files.
- [LangChain](https://github.com/hwchase17/langchain) - Framework for developing applications powered by language models.
- [FAISS](https://faiss.ai/) - Library for efficient similarity search and clustering of dense vectors.
- [Google Generative AI](https://cloud.google.com/generative-ai) - API for generating text and embeddings.
- [dotenv](https://pypi.org/project/python-dotenv/) - Library for loading environment variables from a `.env` file.

## Installation

### Prerequisites

- Python 3.10 or higher
- A Google Cloud account with access to the Generative AI API

### Step-by-Step Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sohamfcb/pdf-qa.git
   cd chat-with-pdfs

2. **Create a virtual environment:**

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt

4. **Set up environment variables:**
   Create a .env file in the root directory of the project and add your Google API key:
   ```bash
   GOOGLE_API_KEY=your_google_api_key_here

### How It Works

- **PDF Text Extraction:** The application uses PyPDF2 to extract text from uploaded PDF documents.
- **Text Chunking:** The extracted text is split into manageable chunks using LangChain's RecursiveCharacterTextSplitter.
- **Vector Store Creation:** Text chunks are transformed into embeddings, which are stored in a FAISS vector store for efficient similarity searches.
- **Conversational AI:** When a user submits a question, the application retrieves relevant text chunks and uses Google Generative AI to generate context-aware responses.
