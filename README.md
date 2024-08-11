# RAG - Retrieval-Augmented Generation - Application with Streamlit UI

## Overview

This project is a Retrieval-Augmented Generation (RAG) application built with Streamlit. It provides functionalities to upload PDFs, process their content into text segments, create embeddings, store and retrieve embeddings from an SQLite database, and query the embeddings using a language model.

## Features

- **PDF Upload and Processing**: Upload PDF files and extract text from them.
- **Text Segmentation**: Split extracted text into manageable segments.
- **Embedding Creation**: Generate embeddings for text segments using pre-trained models.
- **Database Management**: Store embeddings in an SQLite database and manage multiple databases.
- **Query System**: Query the embeddings using a language model and retrieve relevant segments.

## Installation

To run this application, you'll need to set up a Python environment and install the required dependencies. Here’s how you can get started:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/vipulgaur/local-rag-for-pdfs.git
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Start the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

2. **Access the app**: Open your web browser and navigate to `http://localhost:8501`.

## File Structure

- `app.py`: The main script for the Streamlit application.
- `database/`: Directory where SQLite databases are stored.
- `uploaded_pdfs/`: Directory where uploaded PDF files are saved.
- `requirements.txt`: List of Python packages required to run the application.

## Functionality

### 1. **Upload PDF Files**

Upload PDF files to process and generate embeddings. Choose an embedding model from the provided options and specify a database name for storing the embeddings.

### 2. **Create and Store Embeddings**

The application extracts text from the uploaded PDFs, splits it into segments, generates embeddings, and stores these embeddings in the specified SQLite database.

### 3. **Select Existing Embeddings**

Select an existing SQLite database to query. The application allows you to choose an embedding model and loads the embeddings for that model from the selected database.

### 4. **Query Embeddings**

Enter a query and select a language model. The application retrieves relevant text segments from the embeddings and generates responses based on these segments.

## Requirements

The application requires the following Python packages:

- `streamlit`
- `ollama`
- `pdfplumber`
- `sqlite3` (included with Python)
- `sentence-transformers`
- `langchain`
- `numpy`
- `faiss-cpu`

You can install all required packages using the `requirements.txt` file provided.

## Example

1. **Upload PDFs**:
   - Click the "Upload PDF Files" section and choose PDF files to upload.
   - Select an embedding model and specify a database name.
   - Click "Process New PDFs" to generate and store embeddings.

2. **Select and Query**:
   - In the "Select Existing Embeddings Database" section, choose a database and embedding model.
   - Enter a query and select an LLM model.
   - Click "Query" to retrieve and display relevant segments and generated responses.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to open issues or submit pull requests. Your contributions are welcome!

## Contact

For any questions or feedback, please contact [hirevipulgaur@gmail.com].

