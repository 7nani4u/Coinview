from http.server import BaseHTTPRequestHandler
from api._shared import handle_endpoint

class handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[CoinOracle/screener] {fmt % args}")

    def do_OPTIONS(self):
        from api.index import _send
        _send(self, {})

    def do_GET(self):
        endpoint = '/api/health' if 'screener' == 'health' else '/api/screener'
        if endpoint == '/api/health':
            from api.index import _send
            _send(self, {'status': 'ok', 'service': 'CoinOracle'})
        else:
            handle_endpoint(self, endpoint)
