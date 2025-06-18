# Command-Line Chatbot with TinyLlama

This project implements a command-line chatbot in Python using the `TinyLlama/TinyLlama-1.1B-Chat-v1.0` model, run locally via the Hugging Face `transformers` library. It maintains a short-term memory of the conversation using a sliding window mechanism to provide coherent, multi-turn replies.

## Features

-   **Local Model:** Runs the lightweight `TinyLlama-1.1B-Chat` model locally on your CPU.
-   **Conversational Memory:** Remembers the last 3 exchanges to keep track of context.
-   **Standard Libraries:** Uses the standard `transformers` and `torch` libraries.
-   **Robust CLI:** A simple and continuous command-line interface 
-   **Graceful Exit:** Terminate the chat by typing `/exit`.
-   **Modular Code:** Logic is separated into modules for the pipeline loader, memory management, and user interface.

## Requirements

-   Python 3.8+
-   An internet connection (for the initial model download).

## Setup Instructions

1.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install Dependencies:**
    This project requires `transformers`, `torch` for CPU inference, and `accelerate` for efficient model loading.

    ```bash
    pip install transformers torch accelerate
    ```
    *Note: `torch` is a large library. The installation might take some time.*

## How to Run

1.  **Navigate to the Directory:**
    Make sure you are in the root directory of the project where `interface.py` is located.

2.  **Run the Chatbot:**

    Execute the main interface script. The first time you run it, it will download the TinyLlama model (approx. 2 GB), which may take some time depending on your internet connection.

    ```bash
    python interface.py
    ```

3.  **Start Chatting:**
    Once you see the message `Chatbot initialized.` `Type '/exit' to quit.`, you can start typing your messages.

## Sample Interaction
```bash
User: Hi there!
Bot: Hello! How can I help you today?
User: What is the capital of France?
Bot: The capital of France is Paris.
User: And what about Italy?
Bot: The capital of Italy is Rome.
User: Can you tell me something about the Eiffel Tower?
Bot: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
User: /exit
Exiting chatbot. Goodbye!
```