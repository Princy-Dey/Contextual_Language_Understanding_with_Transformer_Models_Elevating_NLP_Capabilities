import torch
from transformers import pipeline
import openai
import os
from dotenv import load_dotenv


# Load Pretrained Models
print("Loading NLP models...")

sentiment_model = pipeline("sentiment-analysis")
summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

print("Models loaded successfully!")

# Function: Sentiment Analysis
def analyze_sentiment(text):
    result = sentiment_model(text)
    return result[0]  # Returns {label: 'POSITIVE', score: 0.99}

# Function: Summarization
def summarize_text(text):
    summary = summarization_model(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Function: Question Answering
def answer_question(context, question):
    result = qa_model(question=question, context=context)
    return result['answer']

# Function: GPT-3.5 Chatbot
def chat_with_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response["choices"][0]["message"]["content"]

# Function: Display NLP Menu
def display_menu():
    print("\n🔹 Choose an NLP Task:")
    print("1️⃣ Sentiment Analysis")
    print("2️⃣ Text Summarization")
    print("3️⃣ Question Answering")
    print("4️⃣ Chat with GPT-3.5")
    print("5️⃣ Exit")

# Main Interactive Loop
def main():
    while True:
        display_menu()
        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == "1":
            text = input("\nEnter the text for Sentiment Analysis: ")
            result = analyze_sentiment(text)
            print("\n📝 Sentiment Analysis Result:", result)

        elif choice == "2":
            text = input("\nEnter the text for Summarization: ")
            result = summarize_text(text)
            print("\n📝 Summarized Text:", result)

        elif choice == "3":
            context = input("\nEnter the context (paragraph): ")
            question = input("\nEnter your question: ")
            result = answer_question(context, question)
            print("\n📝 Answer:", result)

        elif choice == "4":
            user_prompt = input("\nEnter your message for GPT-3.5 Chatbot: ")
            result = chat_with_gpt(user_prompt)
            print("\n🤖 GPT-3.5 Response:", result)

        elif choice == "5":
            print("\n🚀 Exiting NLP System. Have a great day!")
            break  # Exit the loop

        else:
            print("\n❌ Invalid choice! Please select a valid option.")

# Run the program
if __name__ == "__main__":
    main()
