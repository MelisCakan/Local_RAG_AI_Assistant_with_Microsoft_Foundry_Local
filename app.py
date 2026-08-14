import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk

from main import answer_query, delete_document, initialize_models, ingest_document_file, list_uploaded_documents


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class RAGApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Cute Local RAG Assistant")
        self.geometry("1100x720")
        self.minsize(900, 620)

        self.embedding_client = None
        self.chat_client = None
        self.models_loading = False

        self.configure(fg_color="#0f172a")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=260, corner_radius=20, fg_color="#111827")
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(18, 10), pady=18)
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_columnconfigure(0, weight=1)

        self.sidebar_title = ctk.CTkLabel(
            self.sidebar,
            text="My Documents",
            font=ctk.CTkFont(family="Trebuchet MS", size=22, weight="bold"),
            text_color="#f8fafc",
        )
        self.sidebar_title.grid(row=0, column=0, padx=18, pady=(18, 12), sticky="w")

        self.upload_button = ctk.CTkButton(
            self.sidebar,
            text="Upload files",
            command=self.upload_documents,
            fg_color="#f9a8d4",
            hover_color="#f472b6",
            text_color="#4c0519",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.upload_button.grid(row=1, column=0, padx=18, pady=(0, 8), sticky="ew")

        self.delete_button = ctk.CTkButton(
            self.sidebar,
            text="Delete selected",
            command=self.delete_selected_document,
            fg_color="#fbcfe8",
            hover_color="#f9a8d4",
            text_color="#4c0519",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.delete_button.grid(row=2, column=0, padx=18, pady=(0, 12), sticky="ew")

        self.doc_listbox = tk.Listbox(
            self.sidebar,
            bg="#0f172a",
            fg="#fdf2f8",
            selectbackground="#f472b6",
            selectforeground="#4c0519",
            font=("Segoe UI", 11),
            activestyle="none",
            height=20,
            borderwidth=0,
            highlightthickness=0,
        )
        self.doc_listbox.grid(row=3, column=0, padx=18, pady=(0, 18), sticky="nsew")

        self.sidebar.grid_rowconfigure(3, weight=1)

        self.main = ctk.CTkFrame(self, corner_radius=22, fg_color="#111827")
        self.main.grid(row=0, column=1, sticky="nsew", padx=(0, 18), pady=18)
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(2, weight=1)

        self.header = ctk.CTkLabel(
            self.main,
            text="Ask about your local documents",
            font=ctk.CTkFont(family="Trebuchet MS", size=30, weight="bold"),
            text_color="#f8fafc",
        )
        self.header.grid(row=0, column=0, padx=22, pady=(22, 8), sticky="w")

        self.status_label = ctk.CTkLabel(
            self.main,
            text="Loading local models...",
            font=ctk.CTkFont(family="Trebuchet MS", size=12),
            text_color="#93c5fd",
        )
        self.status_label.grid(row=1, column=0, padx=22, pady=(0, 8), sticky="w")

        self.chat_frame = ctk.CTkScrollableFrame(
            self.main,
            corner_radius=18,
            fg_color="#0f172a",
            scrollbar_button_color="#334155",
            scrollbar_button_hover_color="#475569",
        )
        self.chat_frame.grid(row=2, column=0, padx=22, pady=(0, 10), sticky="nsew")
        self.chat_frame.grid_columnconfigure(0, weight=1)

        self.message_log = []

        self.prompt_frame = ctk.CTkFrame(self.main, corner_radius=16, fg_color="#0f172a")
        self.prompt_frame.grid(row=3, column=0, padx=22, pady=(0, 10), sticky="ew")
        self.prompt_frame.grid_columnconfigure(0, weight=1)

        self.question_entry = ctk.CTkTextbox(
            self.prompt_frame,
            height=90,
            border_width=1,
            corner_radius=14,
            font=ctk.CTkFont(family="Trebuchet MS", size=14),
            fg_color="#0f172a",
            border_color="#334155",
        )
        self.question_entry.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.ask_button = ctk.CTkButton(
            self.prompt_frame,
            text="Ask",
            command=self.ask_question,
            fg_color="#f9a8d4",
            hover_color="#f472b6",
            text_color="#4c0519",
            font=ctk.CTkFont(size=14, weight="bold"),
            width=100,
        )
        self.ask_button.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="e")

        self.clear_button = ctk.CTkButton(
            self.prompt_frame,
            text="Clear chat",
            command=self.clear_chat,
            fg_color="#fbcfe8",
            hover_color="#f9a8d4",
            text_color="#4c0519",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=110,
        )
        self.clear_button.grid(row=0, column=2, padx=(0, 10), pady=10, sticky="e")

        self.question_entry.bind("<Shift-Return>", lambda event: self.question_entry.insert("end", "\n"))
        self.question_entry.bind("<Return>", self.on_enter_key)

        self.ask_button.configure(state="disabled")
        self.delete_button.configure(state="disabled")

        self.after(200, self.start_model_loading)
        self.refresh_document_list()

    def add_chat_message(self, sender, text):
        bubble = ctk.CTkFrame(
            self.chat_frame,
            corner_radius=18,
            fg_color="#fbcfe8" if sender == "assistant" else "#f9a8d4",
        )
        bubble.grid(
            sticky="e" if sender == "user" else "w",
            padx=12,
            pady=(6, 6),
            ipadx=14,
            ipady=8,
        )
        bubble.grid_columnconfigure(0, weight=1)

        sender_label = ctk.CTkLabel(
            bubble,
            text=sender.title(),
            font=ctk.CTkFont(family="Trebuchet MS", size=12, weight="bold"),
            text_color="#4c0519",
        )
        sender_label.grid(row=0, column=0, sticky="w", padx=10, pady=(8, 0))

        message_label = ctk.CTkLabel(
            bubble,
            text=text,
            font=ctk.CTkFont(family="Trebuchet MS", size=14),
            wraplength=620,
            justify="left",
            text_color="#4c0519",
        )
        message_label.grid(row=1, column=0, sticky="w", padx=10, pady=(4, 10))

        self.message_log.append((sender, text))
        self.chat_frame.update_idletasks()
        if hasattr(self.chat_frame, "_parent_canvas"):
            self.chat_frame._parent_canvas.yview_moveto(1.0)

    def clear_chat(self):
        for widget in self.chat_frame.winfo_children():
            widget.destroy()
        self.message_log.clear()
        self.status_label.configure(text="Chat cleared. Ready for a new conversation.")

    def on_enter_key(self, event=None):
        if self.ask_button.cget("state") == "normal":
            self.ask_question()
        return "break"

    def start_model_loading(self):
        if self.models_loading:
            return

        self.models_loading = True
        self.status_label.configure(text="Preparing the local AI models... this may take a minute")

        thread = threading.Thread(target=self._load_models_worker, daemon=True)
        thread.start()

    def _load_models_worker(self):
        try:
            embedding_client, chat_client = initialize_models()
        except Exception as exc:  # pragma: no cover - UI error path
            self.after(0, lambda: self._handle_model_error(str(exc)))
            return

        self.embedding_client = embedding_client
        self.chat_client = chat_client
        self.after(0, self._on_models_ready)

    def _handle_model_error(self, error_message):
        self.models_loading = False
        self.status_label.configure(text=f"Model loading failed: {error_message}")
        messagebox.showerror("Model loading error", error_message)

    def _on_models_ready(self):
        self.models_loading = False
        self.status_label.configure(text="Models ready. Upload your documents and ask a question.")
        self.ask_button.configure(state="normal")
        self.delete_button.configure(state="normal")
        self.refresh_document_list()

    def refresh_document_list(self):
        self.doc_listbox.delete(0, tk.END)
        docs = list_uploaded_documents()

        if not docs:
            self.doc_listbox.insert(tk.END, "No documents uploaded yet")
            self.doc_listbox.configure(state="disabled")
            return

        self.doc_listbox.configure(state="normal")
        for doc in docs:
            self.doc_listbox.insert(tk.END, doc)

    def upload_documents(self):
        if self.embedding_client is None or self.chat_client is None:
            messagebox.showinfo("Models loading", "Please wait a moment while the local models finish loading.")
            return

        file_paths = filedialog.askopenfilenames(
            title="Select documents to upload",
            filetypes=[
                ("Text files", "*.txt"),
                ("PDF files", "*.pdf"),
                ("Markdown files", "*.md"),
                ("Word files", "*.docx"),
                ("All files", "*.*"),
            ],
        )

        if not file_paths:
            return

        self.status_label.configure(text=f"Uploading {len(file_paths)} document(s)...")

        for file_path in file_paths:
            try:
                ingest_document_file(file_path, self.embedding_client)
            except Exception as exc:  # pragma: no cover - UI error path
                messagebox.showerror("Upload failed", f"Could not process {file_path}: {exc}")
                continue

        self.refresh_document_list()
        self.status_label.configure(text="Documents uploaded successfully.")

    def delete_selected_document(self):
        selected = self.doc_listbox.curselection()

        if not selected:
            messagebox.showinfo("No selection", "Please select a document to delete.")
            return

        source_name = self.doc_listbox.get(selected[0])
        if source_name == "No documents uploaded yet":
            return

        deleted_rows = delete_document(source_name)
        if deleted_rows <= 0:
            messagebox.showwarning("Delete failed", f"No rows were removed for {source_name}.")
            return

        self.refresh_document_list()
        self.status_label.configure(text=f"Deleted {source_name} from the local knowledge base.")

    def ask_question(self):
        question = self.question_entry.get("1.0", "end").strip()
        if not question:
            return

        if self.embedding_client is None or self.chat_client is None:
            self.add_chat_message("assistant", "The local models are still loading. Please wait a moment and try again.")
            return

        self.ask_button.configure(state="disabled")
        self.status_label.configure(text="Thinking...")
        self.add_chat_message("user", question)
        self.question_entry.delete("1.0", "end")

        def run_query():
            answer = answer_query(question, self.embedding_client, self.chat_client)
            self.after(0, lambda: self._display_answer(answer))

        threading.Thread(target=run_query, daemon=True).start()

    def _display_answer(self, answer):
        self.add_chat_message("assistant", answer)
        self.ask_button.configure(state="normal")
        self.status_label.configure(text="Ready")


if __name__ == "__main__":
    app = RAGApp()
    app.mainloop()