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

    # system prompt
    system_prompt = "You are a friendly and helpful conversational chatbot."

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
            prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False)
            response = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

            # Extract just the assistant's new reply
            # The output is a list with the full conversation, so we get the last message
            bot_response_full = response[0]['generated_text']
            bot_text = bot_response_full[-1]['content'].strip()

            print(f"Bot: {bot_text}")

            # 5. Update memory with the new exchange
            memory.add_exchange(user_input, bot_text)

        except Exception as e:
            print(f"\nAn error occurred during text generation: {e}")
            print("Please try again.")


    print("\nExiting chatbot. Goodbye!")

if __name__ == "__main__":
    main()