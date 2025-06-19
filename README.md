# Command-Line Chatbot with TinyLlama

A command-line chatbot using the `TinyLlama/TinyLlama-1.1B-Chat-v1.0` model, run locally via the Hugging Face `transformers` library. It maintains a short-term memory of the conversation using a sliding window mechanism to provide coherent, multi-turn replies.

no api keys, no internet after setup, just you and a 1.1b parameter model having a conversation in your terminal.

## Features

-   **Local Model:** Runs the lightweight `TinyLlama-1.1B-Chat` model locally after initial setup.
-   **Conversational Memory:** Remembers the last 3 exchanges to keep track of context.
-   **Standard Libraries:** Uses the standard `transformers` and `torch` libraries.
-   **Robust CLI:** A simple and continuous command-line interface 
-   **Graceful Exit:** Terminate the chat by typing `/exit`.
-   **Modular Code:** Logic is separated into modules for the pipeline loader, memory management, and user interface.

## Requirements

-   Python 3.8+
-   An internet connection (for the initial model download).
-   About 3gb of free disk space (for the model)

## Setup Instructions

1. **Clone this repository**
    ```bash
    git clone https://github.com/aashu-0/cli-chatbot.git
    cd cli-chatbot
    ```
2.  **Setup a Virtual Environment:**
    ```bash
    python -m venv .venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    This project requires `transformers`, `torch` for CPU inference, and `accelerate` for efficient model loading.

    ```bash
    pip install transformers torch accelerate
    ```
    *Note: `torch` is a large library. The installation might take some time.*

4.  **Run the Chatbot**
    ```bash
    python interface.py
    ```

3.  **Start Chatting:**
    Once you see the message `Chatbot initialized.` `Type '/exit' to quit.`, you can start typing your messages.

## Sample Interaction
```text
User: what is the capital of India
Bot: The capital of India is New Delhi, also known as "Delhi."
User: what about China
Bot: China's capital city is Beijing, also known as "Beijing."
User: Name of current ruling party there
Bot: The current ruling party in China is the Chinese Communist Party (CCP).
User: what about India
Bot: India's current ruling party is the Bharatiya Janata Party(BJP), which is currently in power.
User: /exit

Exiting chatbot. Goodbye!
```
## File Structure
```bash
├── interface.py          # main cli loop
├── model_loader.py       # handles model loading
├── chat_memory.py        # manages conversation history
├── README.md            # this file
└── .gitignore           # ignores venv and cache files
```
## Customization
want to tweak the behavior? check these settings in interface.py:

- window_size=3: change how many exchanges to remember
- max_new_tokens=256: adjust response length
- temperature=0.3: make responses more creative (higher) or focused (lower)

## Limitations
- responses can be inconsistent (it's a 1.1b model, not gpt-4)
- cpu-only inference means slower responses
- model knowledge cutoff depends on training data

## Why TinyLlama?
it's small enough to run on most machines, fast enough to be usable, and smart enough to have decent conversations. perfect for learning, experimenting, or when you want a private chatbot that doesn't send your conversations to the cloud.