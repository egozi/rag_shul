"""
server.py — Local dev server for chat-ui
=========================================
Serves index.html and routes POST /api/chat to api/chat.py handler.

Usage:
    cd chat-ui
    OPENAI_API_KEY=sk-... python server.py
    # or
    OPENAI_API_KEY=sk-... python server.py --port 8080
"""

import argparse
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Load .env if present
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if _line.strip() and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from api.chat import handler as ChatHandler
from api.eval import handler as EvalHandler


class LocalHandler(ChatHandler, SimpleHTTPRequestHandler):
    """Serves static files from chat-ui/ and delegates API routes to their handlers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def do_OPTIONS(self):
        if self.path in ("/api/chat", "/api/eval"):
            ChatHandler.do_OPTIONS(self)
        else:
            super().do_OPTIONS()

    def do_GET(self):
        if self.path.split("?")[0] == "/api/eval":
            EvalHandler.do_GET(self)
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/chat":
            ChatHandler.do_POST(self)
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} {fmt % args}")


def main():
    parser = argparse.ArgumentParser(description="Local chat-ui server")
    parser.add_argument("--port", type=int, default=3000, help="Port to listen on (default: 3000)")
    parser.add_argument("--host", default="localhost", help="Host to bind (default: localhost)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        print("Run:  OPENAI_API_KEY=sk-... python server.py")
        sys.exit(1)

    server = HTTPServer((args.host, args.port), LocalHandler)
    print(f"Server running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
