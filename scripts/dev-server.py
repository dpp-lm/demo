"""Local development server with a mirror of the GPU server.

Serves this folder, and forwards /dpp_api/... to the GPU server, so a page opened from localhost
talks to its own origin and the GPU server's CORS rule (it allows only timvieira.github.io) never
applies. The page uses the mirror only when opened with ?server=local, so tests that mock the
GPU server are unaffected.

    python3 scripts/dev-server.py [port]        # then open http://localhost:8770/?server=local
"""
import http.server, os, sys, urllib.error, urllib.request

UPSTREAM = "https://zueri-monstro.rufous-hamlet.ts.net"
PREFIX = "/dpp_api/"


class Handler(http.server.SimpleHTTPRequestHandler):
    def _forward(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or None
        req = urllib.request.Request(UPSTREAM + self.path, data=body, method=self.command,
                                     headers={k: v for k, v in self.headers.items() if k.lower() in ("content-type", "authorization", "accept")})
        try:
            with urllib.request.urlopen(req, timeout=330) as r:
                status, headers, data = r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            status, headers, data = e.code, e.headers, e.read()
        except Exception as e:  # unreachable: report it the way an unreachable server looks to the page
            self.send_error(502, f"mirror: {type(e).__name__}: {e}")
            return
        self.send_response(status)
        self.send_header("Content-Type", headers.get("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith(PREFIX): return self._forward()
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith(PREFIX): return self._forward()
        self.send_error(405)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
    print(f"http://localhost:{port}/?server=local  (mirror of {UPSTREAM}{PREFIX})", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
