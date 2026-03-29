from urllib.parse import urlparse, parse_qs
from api.index import route, _send, VALID_INTERVALS
import re

def _params_from_path(path: str):
    parsed = urlparse(path)
    return {k: v[0] for k, v in parse_qs(parsed.query).items()}, parsed.path.rstrip('/') or '/'

def _validate_common(handler_self, endpoint_path: str, params: dict):
    if endpoint_path in ('/api/coin', '/api/stock'):
        ticker = params.get('ticker', '')
        if len(ticker) > 30 or (ticker and not re.match(r'^[a-zA-Z0-9가-힣.\-\s]+$', ticker)):
            _send(handler_self, {'error': 'Invalid ticker format'}, 400)
            return False
        interval = params.get('interval', '1d')
        if interval not in VALID_INTERVALS:
            _send(handler_self, {'error': f"Invalid interval. Allowed: {', '.join(sorted(VALID_INTERVALS))}"}, 400)
            return False
    return True

def handle_endpoint(handler_self, endpoint_path: str):
    params, _ = _params_from_path(handler_self.path)
    if not _validate_common(handler_self, endpoint_path, params):
        return
    result = route(endpoint_path, params)
    if result is None:
        _send(handler_self, {'error': 'Not found'}, 404)
    else:
        _send(handler_self, result)
