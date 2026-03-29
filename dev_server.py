from http.server import HTTPServer

from api.index import handler

if __name__ == "__main__":
    port = 3000
    server = HTTPServer(("", port), handler)
    print(f"Starting CoinOracle dev server on port {port}...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
