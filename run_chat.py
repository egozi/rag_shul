import sys
from http.server import HTTPServer
from pathlib import Path
from dotenv import load_dotenv

CHAT_UI = Path(__file__).parent / "chat-ui"

load_dotenv(CHAT_UI / ".env")

sys.path.insert(0, str(CHAT_UI))
from api.chat import handler as ChatHandler


class LocalHandler(ChatHandler):
    def do_GET(self):
        try:
            with open(CHAT_UI / "index.html", "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # silence request logs


if __name__ == "__main__":
    port = 3000
    print(f"Running at http://localhost:{port}")
    HTTPServer(("", port), LocalHandler).serve_forever()
