import itertools
import os
import platform
import sys
from pathlib import Path
from dotenv import load_dotenv
import keyboard
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QGuiApplication, QPalette, QCursor
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from general.dataset_setup import download_dataset_to_subfolder
from general.groq_helpers import chat_groq, transcribe_wav, Recorder
from latent_search.search_engine import SearchEngine
from general.win11_theme import apply_win11_theme

print(f"[DEBUG] Python {platform.python_version()} on {platform.platform()}")
print(f"[DEBUG] FIRST 5 PATH entries → {os.environ.get('PATH', '').split(';')[:5]}")
print(f"[DEBUG] OVERLAY flag → {'--overlay' in sys.argv}")

# ----------------------------------------------------------------------------
# 0)  Ensure demo corpus exists + DB initialised
# ----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
DEMO_FOLDER = SCRIPT_DIR / "sample_files"
DEMO_FOLDER.mkdir(exist_ok=True)
if not any(DEMO_FOLDER.iterdir()):
    (DEMO_FOLDER / "readme.txt").write_text("This is a tiny demo file about apples.")

download_dataset_to_subfolder("manisha717/dataset-of-pdf-files", DEMO_FOLDER)

_engine = SearchEngine(DEMO_FOLDER)

# ----------------------------------------------------------------------------
# 1)  Worker thread (so the UI remains responsive)
# ----------------------------------------------------------------------------
class SearchWorker(QThread):
    results_ready = Signal(str, list)  # query, results

    def __init__(self, query: str):
        super().__init__()
        self._query = query

    def run(self):
        # Each thread gets its *own* SearchEngine → its own SQLite connection
        results = _engine.search(self._query, k=10)
        self.results_ready.emit(self._query, results)


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


class ResultCard(QWidget):
    def __init__(self, score: float, path: str, content: str):
        super().__init__()
        self.path = path
        self.content = content

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Icon
        icon_label = QLabel()
        icon_label.setPixmap(self._get_icon_for_file(path))
        layout.addWidget(icon_label)

        # Text content
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)

        title_label = QLabel(f"<b>{Path(path).name}</b>")
        title_label.setTextFormat(Qt.RichText)
        text_layout.addWidget(title_label)

        snippet_label = QLabel(self._get_snippet(content))
        snippet_label.setWordWrap(True)
        text_layout.addWidget(snippet_label)

        layout.addLayout(text_layout, 1)

        # Score
        score_label = QLabel(f"<i>{score:.3f}</i>")
        score_label.setTextFormat(Qt.RichText)
        layout.addWidget(score_label)

    def _get_icon_for_file(self, path: str):
        # You would have a more robust way of getting icons,
        # for now, we'll use a simple placeholder.
        # You can use QIcon.fromTheme() for system icons
        from PySide6.QtGui import QIcon
        import mimetypes

        mime_type, _ = mimetypes.guess_type(path)
        if mime_type:
            if mime_type.startswith("image"):
                return QIcon.fromTheme("image-x-generic").pixmap(32, 32)
            if mime_type == "application/pdf":
                return QIcon.fromTheme("application-pdf").pixmap(32, 32)
        return QIcon.fromTheme("text-x-generic").pixmap(32, 32)

    def _get_snippet(self, content: str, max_len=150):
        return content[:max_len] + "..." if len(content) > max_len else content


class SearchApp(QMainWindow):
    hotkeyFired = Signal()

    def __init__(self, *, overlay: bool = False):
        super().__init__()
        apply_win11_theme(self, palette="auto", acrylic=True)

        self._overlay_mode = overlay

        if not self._overlay_mode:
            self.resize(1200, 800)

        self.setWindowTitle("Latent Space Search")
        self._setup_ui()
        self._setup_chatbot_dock()

        # NEW — cosmetic + hot-key
        if self._overlay_mode:
            self._configure_overlay()
        if platform.system() == "Windows":
            self.hotkeyFired.connect(self._toggle_overlay)
            # ── GLOBAL HOTKEY via `keyboard` ────────────────────────────────────
            try:
                keyboard.add_hotkey(
                    'ctrl+alt+x',
                    lambda: self.hotkeyFired.emit(),
                    suppress=True,
                )
                print("[DEBUG] keyboard hotkey registered → Ctrl+Alt+X")
            except Exception as e:
                print(f"[DEBUG] keyboard.hotkey failed: {e}")

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
        self.search_input.textChanged.connect(self._debounced_search)

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.timeout.connect(self._perform_search)

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
        self.results_area = QScrollArea()
        self.results_area.setWidgetResizable(True)
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_area.setWidget(self.results_widget)

        main_layout.addWidget(self.results_area)

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

    @Slot()
    def _debounced_search(self):
        self._search_timer.start(100)

    # ------------------- Slots ---------------------------------------------
    @Slot()
    def _perform_search(self):
        query = self.search_input.text().strip()
        if not query:
            self._clear_results()
            return

        # Don't show "Searching..." if results are already displayed
        if self.results_layout.count() == 0:
            self._clear_results()
            self.results_layout.addWidget(QLabel("Searching…"))

        self.worker = SearchWorker(query)
        self.worker.results_ready.connect(self._display_results)
        self.worker.start()

    def _clear_results(self):
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    @Slot(str, list)
    def _display_results(self, query: str, results: list):
        # If the query has changed since the search was started, ignore these results
        if query != self.search_input.text().strip():
            return

        self._clear_results()
        if not results:
            self.results_layout.addWidget(QLabel("No results."))
            return

        for score, path, content in results:
            card = ResultCard(score, path, content)
            self.results_layout.addWidget(card)

        self.results_layout.addStretch()  # Pushes cards to the top

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

    def _configure_overlay(self):
        """Apply frameless flags and centre on the screen where the mouse is."""
        self.setWindowFlags(Qt.FramelessWindowHint |
                            Qt.Tool |
                            Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # screen under the cursor
        cursor_pos = QCursor.pos()
        scr = QGuiApplication.screenAt(cursor_pos) or QGuiApplication.primaryScreen()
        geo = scr.availableGeometry()

        bar_h = 10
        w = int(geo.width() * 0.40)
        h = 500
        x = geo.x() + (geo.width() - w) // 2
        y = geo.y() + geo.height() - bar_h - h - 8
        self.setGeometry(x, y, w, h)

    # ---------- NEW: theme helper (run once) ----------------------
    def _apply_system_theme(self):
        # crude check: Windows dark mode registry value (0 = dark)
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
                v, _ = winreg.QueryValueEx(k, "AppsUseLightTheme")
            if v == 0:
                QApplication.setStyle("Fusion")
                dark = QApplication.palette()
                dark.setColor(QPalette.Window, Qt.black)
                dark.setColor(QPalette.WindowText, Qt.white)
                QApplication.setPalette(dark)
        except OSError:
            pass  # fallback: leave default

    def _animate_visibility(self, show: bool):
        """Instantly show or hide – no fade (debug mode)."""
        if show:
            self.setWindowOpacity(1.0)
            self._configure_overlay()  # still centre on the active monitor
            self.show()
            self.raise_()
            self.activateWindow()
        else:
            self.hide()

    def _toggle_overlay(self):
        if self.isVisible():
            self._animate_visibility(False)
        else:
            self._configure_overlay()  # centre every time
            self._animate_visibility(True)

    def closeEvent(self, ev):
        super().closeEvent(ev)


if __name__ == "__main__":
    overlay_flag = "--overlay" in sys.argv
    if overlay_flag:
        sys.argv.remove("--overlay")

    app = QApplication(sys.argv)
    window = SearchApp(overlay=overlay_flag)
    if overlay_flag:
        window._animate_visibility(True)  # start shown & animated
    else:
        window.show()
    sys.exit(app.exec())
