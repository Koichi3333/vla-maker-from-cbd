#!/usr/bin/env python3
"""Sign the Colab CLI in without copying a 700-character URL out of a terminal.

This runs the CLI's *own* OAuth flow -- same bundled client, same scopes, same
registered landing page (`colab_cli.auth.REMOTE_REDIRECT_URI`) -- and changes
only how the authorization URL reaches your browser. The URL is longer than
700 characters; copied out of a terminal it wraps, gets truncated, loses
`&scope=...`, and Google answers "Error 400: invalid_request / Missing
required parameter: scope". So instead it is:

  * opened directly: `$BROWSER` in Codespaces / VS Code Remote (opens in the
    browser on *your* machine), the default browser on a local desktop;
  * always written to colab-cli/ws/.auth/ as a text file and a one-link HTML
    page, so you can open it from the editor if nothing opened;
  * optionally served over HTTP (--serve PORT) for port-forwarded setups.

Google then shows an authorization code; paste it at the prompt. The token is
written to ~/.config/colab-cli/token.json (mode 0600) exactly where the CLI
reads it, refresh token included, so this is a one-time step per machine.

Run it through colab-cli/auth.sh, which picks the Python that has colab_cli.

Two-step mode (for agents, or when stdin is not a terminal):

    auth.sh --url-only        # writes the link files and exits
    auth.sh --code 4/0A...    # exchanges the code from the landing page
"""

import argparse
import html
import http.server
import json
import os
import platform
import shutil
import socketserver
import subprocess
import sys
import threading
import webbrowser
from importlib import resources

os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")  # partial grants are checked below

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from colab_cli import auth as colab_auth
    PUBLIC_SCOPES = colab_auth.PUBLIC_SCOPES
    TOKEN_PATH = colab_auth.TOKEN_CONFIG_PATH
    REDIRECT = colab_auth.REMOTE_REDIRECT_URI
except (ImportError, AttributeError) as exc:  # a future colab-cli may rename these
    sys.exit(
        f"[auth] this colab-cli does not expose what auth.py needs ({exc}).\n"
        "[auth] run colab-cli/setup.sh, or sign in with the CLI directly:\n"
        "         colab whoami 2> /tmp/colab_auth_url.txt   # URL goes to the file, prompt stays here"
    )

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK_DIR = os.path.join(HERE, "ws", ".auth")
LINK_TXT = os.path.join(LINK_DIR, "auth_url.txt")
LINK_HTML = os.path.join(LINK_DIR, "index.html")
STASH = os.path.join(os.path.dirname(TOKEN_PATH), ".auth_flow.json")
REQUIRED_SCOPE = "https://www.googleapis.com/auth/colaboratory"


def client_config():
    return json.loads(resources.files("colab_cli").joinpath("oauth_config.json").read_text())


def write_private(path, text):
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)


def existing_token_ok():
    """True if the cached token still loads and refreshes, with the Colab scope."""
    if not os.path.exists(TOKEN_PATH):
        return False
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, PUBLIC_SCOPES)
        if not creds.valid:
            creds.refresh(Request())
        return REQUIRED_SCOPE in (creds.granted_scopes or creds.scopes or [])
    except Exception:
        return False


def start_flow():
    flow = InstalledAppFlow.from_client_config(client_config(), PUBLIC_SCOPES)
    flow.redirect_uri = REDIRECT
    # Identical to colab_cli.auth._run_remote_flow.
    url, state = flow.authorization_url(prompt="consent", token_usage="remote")
    write_private(STASH, json.dumps({"code_verifier": flow.code_verifier, "state": state}))
    write_private(LINK_TXT, url + "\n")
    write_private(LINK_HTML, (
        "<!doctype html><meta charset=utf-8><title>Colab CLI sign-in</title>"
        "<body style='font:16px system-ui;margin:3em'>"
        f"<p><a href=\"{html.escape(url, quote=True)}\" style='font-size:1.4em'>"
        "Sign in to the Colab CLI</a></p>"
        "<p>Approve, then copy the code Google shows and paste it into the terminal.</p>"
    ))
    return url


def open_browser(url):
    """Open the URL where the human is. Returns a short description or None."""
    browser = os.environ.get("BROWSER")
    if browser and shutil.which(browser.split()[0]):
        # Codespaces / VS Code Remote set $BROWSER to a helper that opens the
        # URL on the client machine, not in the container.
        if subprocess.call([browser, url], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) == 0:
            return "$BROWSER (your local browser)"
    local = platform.system() == "Darwin" or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    if local and not os.environ.get("SSH_CONNECTION"):
        try:
            if webbrowser.open(url):
                return "your default browser"
        except Exception:
            pass
    return None


def serve(port):
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=LINK_DIR, **kw)

        def log_message(self, *a):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.TCPServer(("127.0.0.1", port), Quiet)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    name, domain = os.environ.get("CODESPACE_NAME"), os.environ.get("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN")
    if name and domain:
        print(f"[auth] serving the link page: https://{name}-{port}.{domain}/")
        print("[auth]   (or PORTS panel -> port %d -> Open in Browser)" % port)
    else:
        print(f"[auth] serving the link page on 127.0.0.1:{port}")
        print(f"[auth]   from your machine: ssh -L {port}:127.0.0.1:{port} <host>, then open http://127.0.0.1:{port}/")
    return srv


def exchange(code):
    if not os.path.exists(STASH):
        sys.exit("[auth] no sign-in in progress. Start one with: colab-cli/auth.sh --url-only")
    with open(STASH, encoding="utf-8") as fh:
        stash = json.load(fh)
    flow = InstalledAppFlow.from_client_config(
        client_config(), PUBLIC_SCOPES,
        code_verifier=stash["code_verifier"], state=stash["state"])
    flow.redirect_uri = REDIRECT
    try:
        flow.fetch_token(code=code.strip())
    except Exception as exc:
        sys.exit(f"[auth] code exchange failed: {exc}\n"
                 "[auth] codes are single-use and expire within minutes; start again.")
    creds = flow.credentials
    write_private(TOKEN_PATH, creds.to_json())
    for path in (STASH, LINK_TXT, LINK_HTML):
        if os.path.exists(path):
            os.remove(path)
    granted = creds.granted_scopes or creds.scopes or []
    print(f"[auth] token written to {TOKEN_PATH}")
    if REQUIRED_SCOPE not in granted:
        print("[auth] WARNING: the 'colaboratory' scope was not granted (a checkbox was left "
              "unticked on the consent screen). Sessions will fail with 403. "
              "Run colab-cli/auth.sh --force and tick every box.", file=sys.stderr)
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="sign in again even if a working token exists")
    ap.add_argument("--no-open", action="store_true", help="do not try to open a browser")
    ap.add_argument("--serve", type=int, metavar="PORT", help="also serve the link page on 127.0.0.1:PORT")
    ap.add_argument("--url-only", action="store_true", help="write the link files and exit (step 1 of 2)")
    ap.add_argument("--code", help="exchange this authorization code (step 2 of 2)")
    args = ap.parse_args()

    if args.code:
        return exchange(args.code)

    if not args.force and existing_token_ok():
        print(f"[auth] already signed in ({TOKEN_PATH}). Use --force to sign in again.")
        return 0

    url = start_flow()
    print(f"[auth] sign-in link written to {os.path.relpath(LINK_TXT)} and {os.path.relpath(LINK_HTML)}")
    if args.url_only:
        print("[auth] open that link, approve, then run: colab-cli/auth.sh --code <CODE>")
        return 0

    opened = None if args.no_open else open_browser(url)
    srv = serve(args.serve) if args.serve else None
    if opened:
        print(f"[auth] opened the sign-in page in {opened}.")
    elif not srv:
        print("[auth] could not open a browser here. Open the link file above in your editor "
              "(Ctrl/Cmd+click the URL), or re-run with --serve 8765.")
    print("[auth] approve in the browser; Google then shows an authorization code.")
    try:
        code = input("Paste the authorization code: ")
    except (EOFError, KeyboardInterrupt):
        print("\n[auth] no code entered. Finish later with: colab-cli/auth.sh --code <CODE>")
        return 1
    finally:
        if srv:
            srv.shutdown()
    return exchange(code)


if __name__ == "__main__":
    sys.exit(main())
