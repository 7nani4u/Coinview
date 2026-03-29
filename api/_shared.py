from urllib.parse import parse_qs, urlparse

from api.config import VALID_INTERVALS
from api.index import _send, route
from api.validators import validate_interval, validate_ticker


def _params_from_path(path: str):
    parsed = urlparse(path)
    return {k: v[0] for k, v in parse_qs(parsed.query).items()}, parsed.path.rstrip('/') or '/'


def _validate_common(handler_self, endpoint_path: str, params: dict):
    if endpoint_path in ('/api/coin', '/api/stock', '/api/validation'):
        ok, ticker_or_err = validate_ticker(params.get('ticker', params.get('symbol', '')))
        if not ok:
            _send(handler_self, {'error': ticker_or_err}, 400)
            return False
    if endpoint_path in ('/api/coin', '/api/stock', '/api/validation'):
        ok, interval_or_err = validate_interval(params.get('interval', '1d'))
        if not ok:
            _send(handler_self, {'error': interval_or_err, 'valid_intervals': sorted(VALID_INTERVALS)}, 400)
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
