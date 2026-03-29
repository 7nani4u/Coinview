from http.server import BaseHTTPRequestHandler
from api._shared import handle_endpoint


class handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[CoinOracle/validation] {fmt % args}")

    def do_OPTIONS(self):
        from api.index import _send
        _send(self, {})

    def do_GET(self):
        handle_endpoint(self, '/api/validation')
