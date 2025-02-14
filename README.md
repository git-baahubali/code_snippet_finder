
# Code snippet Finder

. It allows you to search for code snippets based on natural language descriptions and chat with your data—all in one cohesive environment.
This Project is a Retrieval-Augmented Generation (RAG) application built with LangChain, Ollama, and ChromaDB

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:

1. **Python 3.10 or Higher**  
   It is recommended to use Python version **3.10** or above.
2. **Ollama**  
   Code snippet finder leverages Ollama for language model interactions.  
   **Download and Install Ollama**:  
   Visit the [Ollama download page](https://ollama.com/download) and follow the instructions for your operating system.

3. ** Download models **
    ```
    ollama pull nomic-embed-text:latest qwen2.5-coder:1.5b
    ollama pull qwen2.5-coder:1.5b
    ```


3. **Git**  
   To clone the repository.

---

## Installation

Follow these steps to set up the project on your local machine:

1. **Clone the Repository**

   Open your terminal and run:

   ```bash
   git clone https://github.com/git-baahubali/code_snippet_finder.git
   cd code_snippet_finder
   ```

2. **Create a Virtual Environment**

   It’s a good practice to isolate your project dependencies. Create and activate a virtual environment:

   ```bash
   # For Unix or macOS:
   python3 -m venv venv
   source venv/bin/activate

   # For Windows:
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Required Libraries**

   Install the necessary Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file includes all the dependencies needed for Code Snippet finder (including LangChain, chromadb, etc.).

---

## Usage

Once installed, you can start the application as follows:

1. **Start the Application**

   Run the main application file (e.g., `app.py`):

   ```bash
   streamlit run app.py
   ```

2. **Upload Files and Interact**

   - Use the sidebar to upload your documents (PDF, TXT, MD).
   - Process the file to create embeddings.
   - Use the chat interface to search for code snippets and interact with the bot.

---

## Project Structure

```
code_snippet_finder/
├── app.py                # Main Streamlit application file
├── README.md             # This file
├── requirements.txt      # List of dependencies
├── documents/            # Folder for sample documents (if any)
├── models/               # Folder for additional model assets (if needed)
└── utils/                # Helper functions and modules
```

---

## Troubleshooting

- **Ollama Not Installed?**  
  Make sure you have installed Ollama from the [Ollama download page](https://ollama.com/download) before running the application.

- **Dependency Issues:**  
  If you encounter issues with `chromadb` or other dependencies, consider downgrading or upgrading the relevant packages as mentioned in the [Troubleshooting section](#troubleshooting).

- **Pydantic Import Errors:**  
  If you see errors related to Pydantic (e.g., `model_validator` not found), ensure you're using a compatible version of Pydantic (v1.x recommended).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Happy coding! If you have any questions or run into issues, please open an issue in the repository or contact the project maintainers.

