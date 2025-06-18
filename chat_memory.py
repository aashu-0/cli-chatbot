# chat_memory.py

class ChatMemory:
    """
    Manages the short-term memory of the conversation using a sliding window.
    Stores history as a list of dictionaries, compatible with Hugging Face pipelines.
    """
    def __init__(self, window_size=3):
        """
        Initializes the memory buffer.
        Args:
            window_size (int): The number of user-bot exchanges to remember.
        """
        self.history = []
        # Each exchange is one user message and one assistant message (2 items)
        self.max_history_length = window_size * 2

    def add_exchange(self, user_input, bot_response):
        """
        Adds a new user-bot exchange to the history.
        If the history exceeds the window size, the oldest exchange is removed.
        """
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": bot_response})

        # Enforce the sliding window
        if len(self.history) > self.max_history_length:
            # Remove the oldest two items (one user message, one assistant reply)
            self.history = self.history[2:]

    def get_history(self):
        """
        Returns the stored conversation history.
        """
        return self.history