import tkinter as tk
from tkinter import scrolledtext

from main import initialize_models, answer_query


# Initialize Foundry Local and load the models
embedding_client, chat_client = initialize_models()


def ask_question():
    question = question_entry.get("1.0", tk.END).strip()

    if not question:
        return

    answer = answer_query(
        question,
        embedding_client,
        chat_client
    )

    answer_box.delete("1.0", tk.END)
    answer_box.insert(tk.END, answer)


# Create the main window
window = tk.Tk()
window.title("Local RAG Assistant")
window.geometry("800x600")


# Title
title_label = tk.Label(
    window,
    text="Local RAG Assistant",
    font=("Arial", 20, "bold")
)
title_label.pack(pady=20)


# Question label
question_label = tk.Label(
    window,
    text="Ask a question about your documents:",
    font=("Arial", 12)
)
question_label.pack()


# Question input
question_entry = scrolledtext.ScrolledText(
    window,
    height=4,
    width=80,
    font=("Arial", 11)
)
question_entry.pack(pady=10)


# Ask button
ask_button = tk.Button(
    window,
    text="Ask",
    command=ask_question,
    font=("Arial", 11, "bold")
)
ask_button.pack(pady=10)


# Answer label
answer_label = tk.Label(
    window,
    text="Answer:",
    font=("Arial", 12, "bold")
)
answer_label.pack(pady=(20, 5))


# Answer box
answer_box = scrolledtext.ScrolledText(
    window,
    height=15,
    width=80,
    font=("Arial", 11),
    wrap=tk.WORD
)
answer_box.pack(padx=20, pady=10)


# Start the application
window.mainloop()