# Contextual Language Understanding with Transformer Models

![NLP](https://img.shields.io/badge/NLP-Transformers-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)
![Torch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)

This project implements multiple Natural Language Processing (NLP) tasks using Transformer models, including Sentiment Analysis, Text Summarization, Question Answering, and a GPT-3.5-powered Chatbot.

---

## 🚀 Features

- **Sentiment Analysis** – Classifies text as positive or negative.
- **Text Summarization** – Generates a summary of the given text.
- **Question Answering** – Extracts an answer based on the given context.
- **GPT-3.5 Chatbot** – Engages in conversation using OpenAI’s GPT-3.5.

---

## 🛠️ Setup Instructions

### Install Dependencies
```bash
pip install torch transformers openai python-dotenv
```

### Set Up OpenAI API Key  
Create a `.env` file in the project directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

---

## 📌 Usage

Run the script:
```bash
python main.py
```

Select an option from the interactive menu to perform different NLP tasks.

---

## 📚 Dependencies

- `torch`
- `transformers`
- `openai`
- `python-dotenv`

---

## 🏆 Example Outputs

### 🔹 Sentiment Analysis:
```
Enter the text for Sentiment Analysis: I love this product!
📝 Sentiment Analysis Result: {'label': 'POSITIVE', 'score': 0.99}
```

### 🔹 Text Summarization:
```
Enter the text for Summarization: (Long paragraph)
📝 Summarized Text: "This is a brief summary of the given text."
```

### 🔹 Question Answering:
```
Enter the context (paragraph): (A passage of text)
Enter your question: What is the main topic?
📝 Answer: "The main topic is..."
```

### 🔹 GPT-3.5 Chatbot:
```
Enter your message for GPT-3.5 Chatbot: Hello, how are you?
🤖 GPT-3.5 Response: "I'm an AI assistant, ready to help!"
```

---

## 🏗️ Future Enhancements
- Improve accuracy with fine-tuned models.
- Extend support for additional NLP tasks.