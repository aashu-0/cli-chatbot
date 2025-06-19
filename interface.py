import model_loader
import chat_memory

def main():
    """
    The main function to run the chatbot CLI.
    """
    # 1. Initialize Pipeline and Memory
    print("Initializing chatbot...")
    pipe = model_loader.load_pipeline()
    if pipe is None:
        return  # Exit if pipeline loading failed

    # window_size = 3  -> number of previous exchanges to remember
    memory = chat_memory.ChatMemory(window_size=3)

    # system prompt with examples for concise responses
    system_prompt = """You are a helpful chatbot that gives concise, direct answers. Keep responses short and to the point. Don't over-explain unless asked for details.

Examples:
User: What is the capital of France?
Assistant: Paris.

User: How are you?
Assistant: I'm doing well, thanks! How can I help you today?

User: Tell me about gravity
Assistant: Gravity is the force that pulls objects toward each other. On Earth, it makes things fall down and keeps us on the ground.

User: What's 2+2?
Assistant: 4.

Remember: Be brief, accurate, and conversational. Only give detailed explanations when specifically asked."""

    print("Chatbot initialized. Type '/exit' to quit.")
    print("-" * 50)

    # 2. Start the CLI loop
    while True:
        user_input = input("User: ")

        # Check for exit command
        if user_input.lower() == "/exit":
            break

        # 3. Prepare the messages for the pipeline
        conversation_history = memory.get_history()
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            *conversation_history,
            {"role": "user", "content": user_input},
        ]

        # 4. Generate a response using the pipeline
        try:
            prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            response = pipe(
                prompt, 
                max_new_tokens=256,
                do_sample=True, 
                temperature=0.3,
                top_k=20,
                top_p=0.85,
                return_full_text=False
            )

            # Extract the generated text
            bot_text = response[0]['generated_text'].strip()
            
            # Clean up any potential formatting issues and truncate if too long
            if bot_text.startswith("assistant\n"):
                bot_text = bot_text[len("assistant\n"):].strip()
            if bot_text.startswith("Assistant:"):
                bot_text = bot_text[len("Assistant:"):].strip()

            print(f"Bot: {bot_text}")

            # 5. Update memory with the new exchange
            memory.add_exchange(user_input, bot_text)

        except Exception as e:
            print(f"\nAn error occurred during text generation: {e}")
            print("Please try again.")

    print("\nExiting chatbot. Goodbye!")

if __name__ == "__main__":
    main()