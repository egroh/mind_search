import itertools
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from groq_helpers import chat_groq, transcribe_wav, Recorder
from search_engine import SearchEngine

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

# ----------------------------------------------------------------------------
# 0)  Ensure demo corpus exists + DB initialised
# ----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
DEMO_FOLDER = SCRIPT_DIR / "sample_docs"
DEMO_FOLDER.mkdir(exist_ok=True)
if not any(DEMO_FOLDER.iterdir()):
    (DEMO_FOLDER / "readme.txt").write_text("This is a tiny demo file about apples.")

_engine = SearchEngine()
_engine.ingest_folder(DEMO_FOLDER)


# ----------------------------------------------------------------------------
# 1)  Worker thread (so the UI remains responsive)
# ----------------------------------------------------------------------------
# --- worker thread ----------------------------------------------------------
class SearchWorker(QThread):
    results_ready = Signal(list)

    def __init__(self, query: str, db_path: str = "search.db"):
        super().__init__()
        self._query = query
        self._db_path = db_path

    def run(self):
        # Each thread gets its *own* SearchEngine → its own SQLite connection
        engine = SearchEngine(self._db_path)
        results = engine.query(self._query, k=10)
        self.results_ready.emit(results)


class VoiceWorker(QThread):
    transcript_ready = Signal(str)

    def __init__(self, wav_path: str):
        super().__init__()
        self._wav_path = wav_path

    def run(self):
        text = "Error"
        try:
            text = transcribe_wav(self._wav_path)
        except Exception as e:
            text = f"[STT error] {e}"
            raise e
        finally:
            try:
                os.remove(self._wav_path)
            except FileNotFoundError:
                pass
        self.transcript_ready.emit(text)


# --------------------------------------------------------------------------
# Chat LLM worker  (runs Groq call off-UI thread)
# --------------------------------------------------------------------------
class ChatWorker(QThread):
    reply_ready = Signal(str)

    def __init__(self, messages: list[dict]):
        super().__init__()
        # copy so UI can keep appending while thread runs
        self._messages = messages.copy()

    def run(self):
        try:
            reply = chat_groq(self._messages)
        except Exception as e:
            reply = f"[Groq API error] {e}"
        self.reply_ready.emit(reply)


class SearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Latent Space Search Application")
        self.setGeometry(100, 100, 800, 600)

        self._setup_ui()
        self._setup_chatbot_dock()

        self._recorder = Recorder()
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._toggle_mic_icon)
        self._mic_state_cycle = itertools.cycle(["Rec 🔴", "Rec 🎤"])
        self._recording = False

        self._chat_history_messages = [
            {"role": "system", "content": "You are a helpful desktop-search assistant."}
        ]

    # ------------------- UI helpers ----------------------------------------
    def _setup_ui(self):
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Search Input and Button
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter your search query…")
        self.search_input.returnPressed.connect(self._perform_search)

        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self._perform_search)

        self.mic_btn = QPushButton("Voice 🎤")
        self.mic_btn.setToolTip("Click to start/stop recording")
        self.mic_btn.clicked.connect(self._toggle_recording)

        search_layout.addWidget(self.search_input)
        search_layout.addWidget(search_btn)
        search_layout.addWidget(self.mic_btn)
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
        self.chatbot_dock.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable
        )
        self.chatbot_dock.hide()

        chatbot_content_widget = QWidget()
        chatbot_layout = QVBoxLayout(chatbot_content_widget)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
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

    # ------------------- Slots ---------------------------------------------
    @Slot()
    def _perform_search(self):
        query = self.search_input.text().strip()
        if not query:
            return
        self.results_display.clear()
        self.results_display.addItem("Searching…")

        self.worker = SearchWorker(query)
        self.worker.results_ready.connect(self._display_results)
        self.worker.start()

    @Slot(list)
    def _display_results(self, results: list):
        self.results_display.clear()
        if not results:
            self.results_display.addItem("No results.")
            return
        for score, path, _ in results:
            self.results_display.addItem(f"{score:.3f} – {path}")

    @Slot()
    def _toggle_recording(self):
        if not self._recording:  # start
            self.results_display.clear()
            self.results_display.addItem("🔴 Recording… click again to stop")
            self._recorder.start()
            self._blink_timer.start(500)  # blink twice per second
            self._recording = True
        else:  # stop + transcribe
            wav_path = self._recorder.stop_and_save()
            self._blink_timer.stop()
            self.mic_btn.setText("Voice 🎤")
            self._recording = False

            self.voice_worker = VoiceWorker(wav_path)
            self.voice_worker.transcript_ready.connect(self._voice_to_search)
            self.voice_worker.start()

    def _toggle_mic_icon(self):
        self.mic_btn.setText(next(self._mic_state_cycle))

    @Slot(str)
    def _voice_to_search(self, text: str):
        self.results_display.clear()
        if text.startswith("[STT error]"):
            self.results_display.addItem(text)
            return
        self.search_input.setText(text)
        self._perform_search()

    @Slot()
    def _toggle_chatbot(self):
        self.chatbot_dock.setVisible(not self.chatbot_dock.isVisible())

    @Slot()
    def _send_chat_message(self):
        user_msg = self.chat_input.text().strip()
        if not user_msg:
            return

        # Update GUI immediately
        self.chat_history.append(f"You: {user_msg}")
        self.chat_input.clear()

        # Track convo for the LLM
        self._chat_history_messages.append({"role": "user", "content": user_msg})

        # Kick off background Groq request
        self.chat_worker = ChatWorker(self._chat_history_messages)
        self.chat_worker.reply_ready.connect(self._display_chat_reply)
        self.chat_worker.start()

    @Slot(str)
    def _display_chat_reply(self, reply: str):
        self.chat_history.append(f"Assistant: {reply}")
        # Persist assistant turn for future context
        self._chat_history_messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SearchApp()
    window.show()
    sys.exit(app.exec())
