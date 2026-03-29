from http.server import BaseHTTPRequestHandler
from api._shared import handle_endpoint

class handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[CoinOracle/sentiment] {fmt % args}")

    def do_OPTIONS(self):
        from api.index import _send
        _send(self, {})

    def do_GET(self):
        endpoint = '/api/health' if 'sentiment' == 'health' else '/api/sentiment'
        if endpoint == '/api/health':
            from api.index import _send
            _send(self, {'status': 'ok', 'service': 'CoinOracle'})
        else:
            handle_endpoint(self, endpoint)
