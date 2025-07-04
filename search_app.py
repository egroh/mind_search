from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QTextEdit, QDockWidget, QLabel, QListWidget
)
from PySide6.QtCore import Qt, Slot
import sys

class SearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Latent Space Search Application")
        self.setGeometry(100, 100, 800, 600)

        self._setup_ui()
        self._setup_chatbot_dock()

    def _setup_ui(self):
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Search Input and Button
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter your search query...")
        self.search_input.returnPressed.connect(self._perform_search)
        search_button = QPushButton("Search")
        search_button.clicked.connect(self._perform_search)

        search_layout.addWidget(self.search_input)
        search_layout.addWidget(search_button)
        main_layout.addLayout(search_layout)

        # Search Results Display
        self.results_display = QListWidget()
        
        main_layout.addWidget(self.results_display)

        # Chatbot Toggle Button
        self.chat_button = QPushButton("Toggle Chatbot")
        self.chat_button.clicked.connect(self._toggle_chatbot)
        search_layout.addWidget(self.chat_button)

    def _setup_chatbot_dock(self):
        self.chatbot_dock = QDockWidget("Chatbot", self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.chatbot_dock)
        self.chatbot_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable | QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.chatbot_dock.hide() # Start hidden

        chatbot_content_widget = QWidget()
        chatbot_layout = QVBoxLayout(chatbot_content_widget)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setPlaceholderText("Chat history...")
        chatbot_layout.addWidget(self.chat_history)

        chat_input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type your message...")
        self.chat_input.returnPressed.connect(self._send_chat_message)
        send_button = QPushButton("Send")
        send_button.clicked.connect(self._send_chat_message)

        chat_input_layout.addWidget(self.chat_input)
        chat_input_layout.addWidget(send_button)
        chatbot_layout.addLayout(chat_input_layout)

        self.chatbot_dock.setWidget(chatbot_content_widget)

    @Slot()
    def _perform_search(self):
        query = self.search_input.text().strip()
        if not query:
            self.results_display.clear()
            self.results_display.addItem("Please enter a search query.")
            return

        self.results_display.clear()
        self.results_display.addItem(f"Searching for: '{query}' using latent space embeddings...")
        self.results_display.addItem("--- Mock Results ---")

        # Mocking search results based on query
        if "apple" in query.lower():
            self.results_display.addItem("Result 1: Red Apple - Fruit")
            self.results_display.addItem("Result 2: Apple Inc. - Technology Company")
        elif "car" in query.lower():
            self.results_display.addItem("Result A: Sports Car - Vehicle")
            self.results_display.addItem("Result B: Electric Car - Vehicle")
        else:
            self.results_display.addItem("No specific mock results for this query. Try 'apple' or 'car'.")
            self.results_display.addItem("Result X: Generic item related to your query.")
            self.results_display.addItem("Result Y: Another generic item.")

    @Slot()
    def _toggle_chatbot(self):
        if self.chatbot_dock.isVisible():
            self.chatbot_dock.hide()
        else:
            self.chatbot_dock.show()

    @Slot()
    def _send_chat_message(self):
        message = self.chat_input.text().strip()
        if not message:
            return

        self.chat_history.append(f"You: {message}")
        self.chat_input.clear()

        # Mock chatbot response
        if "hello" in message.lower():
            self.chat_history.append("Chatbot: Hello there! How can I assist you with your search today?")
        elif "search" in message.lower():
            self.chat_history.append("Chatbot: I can help you refine your search. What are you looking for?")
        elif "embedding" in message.lower():
            self.chat_history.append("Chatbot: Latent space embeddings help us find semantically similar items. What would you like to know about them?")
        else:
            self.chat_history.append("Chatbot: I'm a prototype chatbot. I can answer basic questions about search or embeddings.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SearchApp()
    window.show()
    sys.exit(app.exec())
