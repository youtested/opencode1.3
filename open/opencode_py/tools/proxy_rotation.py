"""Optional Tor-backed IP rotation for the web tools.

WHY THIS EXISTS
    Bot walls come in two flavours, and the cheap one only gets you so far:
      * client fingerprint  (Brave, Cloudflare) — rejected by httpx, accepted
        by curl/curl-cffi. The existing bypass cascade in cloudflare_bypass
        already handles this, no IP change needed.
      * IP reputation       (DuckDuckGo, Brave rate limits) — the *address*
        itself is the problem. No client trickery fixes this, and free public
        proxies do not either: they are the most-abused IPs on the internet
        and arrive pre-flagged.
    Tor is the one free source of addresses that are not pre-flagged, so it is
    the only thing that actually lifts the ceiling.

DESIGN — zero when unused
    * Nothing is started, imported or installed. Tor is the user's choice.
    * Detection is LAZY: the port is probed only when a caller asks for the
      escape hatch, so a healthy session never touches Tor at all.
    * The result is cached hard (10 min up / 90 s down), because a probe is a
      loopback connect costing a couple of ms — frequent probing is waste.
    * A failed request retires Tor immediately, so one broken circuit is not
      re-paid on every subsequent request.
    * ON-DEMAND LIFECYCLE (the point of all this): a phone cannot afford a
      permanently resident relay. Tor is started only when an engine is
      actually blocked, and reaped after AUTO_STOP_AFTER seconds idle. That
      turns ~11 MB of standing overhead into a cost paid only on the rare
      search that needs it. A Tor the user started themselves is NEVER
      touched — only a process this module launched is ever reaped.
    * No Tor installed / not running / OPENCODE_TOR=0 -> available() is False
      and callers carry on exactly as before.

REUSE
    ``TorRotation.shared().curl_args()`` returns the curl flags for the live
    proxy, so any tool with a curl invocation (webfetch, future downloaders)
    can route through the same rotation without importing anything else.
"""

from __future__ import annotations

from ..util.log import swallow
import os
import shutil
import signal
import socket
import subprocess
import time
from typing import Any

# Tor's default SOCKS port, then the Tor Browser one.
DEFAULT_PORTS: tuple[int, ...] = (9050, 9150)
# Once Tor is up we trust it for 10 minutes; once we find it down we back off
# for 90 seconds. Both are far longer than a single request needs, and the
# probe itself is a loopback connect.
_TTL_UP = 600.0
_TTL_DOWN = 90.0
_PROBE_TIMEOUT = 0.25
_FETCH_TIMEOUT = 20.0

# Reap a Tor we started once it has been unused this long. Long enough to
# cover a burst of research searches, short enough that a phone is not paying
# ~11 MB for the rest of the session.
AUTO_STOP_AFTER = 90.0
# Tor needs to bootstrap a circuit before it can serve; do not start it unless
# the caller still has enough budget to be worth the wait.
MIN_START_BUDGET = 12.0
# How long to wait for the SOCKS port to come up after launching.
_BOOT_WAIT = 25.0


def _env_off() -> bool:
    """OPENCODE_TOR=0/empty-ish force-disables without touching anything."""
    return os.environ.get("OPENCODE_TOR", "").strip().lower() in ("0", "false", "no", "off")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _port_open(port: int, timeout: float = _PROBE_TIMEOUT) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


class TorRotation:
    """Lazily-detected local Tor SOCKS proxy. Never required, never blocking.

    Thread-safe enough for the single-threaded tool loop; the worst case under
    a race is one extra ~0.1 ms probe.
    """

    def __init__(self, ports: tuple[int, ...] = DEFAULT_PORTS,
                 ttl_up: float = _TTL_UP, ttl_down: float = _TTL_DOWN):
        self._ports = tuple(ports)
        self._ttl_up = ttl_up
        self._ttl_down = ttl_down
        self._port: int | None = None
        self._checked_at = 0.0
        self._was_up = False
        # lifecycle: only a process WE launched is ever reaped
        self._owned_pid: int | None = None
        self._last_used = 0.0

    # -- construction -----------------------------------------------------
    @classmethod
    def shared(cls) -> "TorRotation":
        """Process-wide instance, rebuilt if the env-disable changes."""
        inst = getattr(cls, "_shared", None)
        off = _env_off()
        if inst is None or inst._disabled != off:
            inst = cls()
            inst._disabled = off
            cls._shared = inst
        return inst

    _disabled = False

    # -- lifecycle --------------------------------------------------------
    @staticmethod
    def binary() -> str | None:
        """Path to the tor binary, or None when it is not installed."""
        return shutil.which("tor")

    @staticmethod
    def data_dir() -> str:
        """A writable DataDirectory. Termux has no /var/lib/tor by default."""
        base = os.environ.get("PREFIX") or os.path.expanduser("~/.local")
        path = os.path.join(base, "var", "lib", "tor")
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            path = os.path.expanduser("~/.cache/opencode/tor")
            os.makedirs(path, exist_ok=True)
        return path

    def _alive(self, pid: int | None) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return False
        return True

    def start(self, budget: float = _BOOT_WAIT) -> bool:
        """Launch tor and wait for its SOCKS port. No-op if already up.

        Only ever started when it is genuinely needed, and only when the
        caller has budget to wait for the bootstrap.
        """
        if self._disabled or _env_off():
            return False
        if self._owned_pid and self._alive(self._owned_pid):
            return True
        exe = self.binary()
        if not exe:
            return False
        port = self._ports[0]
        cmd = [exe, "--DataDirectory", self.data_dir(), "--SocksPort", str(port)]
        try:
            # start_new_session so it outlives this process group; DEVNULL
            # because a chatty relay log on a phone is just wasted storage.
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError:
            return False
        self._owned_pid = self._proc.pid
        deadline = time.monotonic() + max(1.0, min(budget, _BOOT_WAIT))
        while time.monotonic() < deadline:
            if _port_open(port, 0.3):
                self._port = port
                self._was_up = True
                self._checked_at = time.monotonic()
                self._last_used = time.monotonic()
                return True
            if self._proc.poll() is not None:   # died during bootstrap
                self._owned_pid = None
                return False
            time.sleep(0.4)
        return False

    def stop(self) -> bool:
        """Stop a tor THIS MODULE started. A user's own tor is never touched."""
        pid, self._owned_pid = self._owned_pid, None
        if not pid or not self._alive(pid):
            return False
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError as _oc_exc:
                swallow(_oc_exc, "opencode_py/tools/proxy_rotation.py:197")
                pass
        # escalate if it ignores SIGTERM
        for _ in range(10):
            if not self._alive(pid):
                break
            time.sleep(0.2)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError as _oc_exc:
                swallow(_oc_exc, "opencode_py/tools/proxy_rotation.py:207")
                pass
        self._port = None
        self._was_up = False
        self._checked_at = 0.0
        return True

    def reap(self, now: float | None = None) -> bool:
        """Stop our tor once it has been idle long enough. Cheap when idle-free."""
        if not self._owned_pid:
            return False
        if not _env_flag("OPENCODE_TOR_AUTOSTOP", True):
            return False
        now = time.monotonic() if now is None else now
        if self._last_used and now - self._last_used < AUTO_STOP_AFTER:
            return False
        return self.stop()

    def ensure(self, budget: float) -> bool:
        """Make sure a proxy is reachable, starting tor only if it is needed."""
        if self._disabled or _env_off():
            return False
        self.reap()
        if self.available():
            self._last_used = time.monotonic()
            return True
        if not _env_flag("OPENCODE_TOR_AUTOSTART", True):
            return False
        if budget < MIN_START_BUDGET:
            return False
        if not self.start(budget=min(budget, _BOOT_WAIT)):
            return False
        self._last_used = time.monotonic()
        return True

    # -- state ------------------------------------------------------------
    def reset(self) -> None:
        """Forget probe state (tests / after a settings change)."""
        self._port = None
        self._checked_at = 0.0
        self._was_up = False
        self._owned_pid = None
        self._last_used = 0.0

    def note_failure(self) -> None:
        """Retire Tor until the next backoff window (one bad circuit is enough)."""
        self._port = None
        self._was_up = False
        self._checked_at = time.monotonic()

    def available(self) -> bool:
        """True when a local Tor SOCKS port is listening. Cached; cheap."""
        if self._disabled or _env_off():
            return False
        now = time.monotonic()
        if now - self._checked_at < (self._ttl_up if self._was_up else self._ttl_down):
            return self._was_up
        found = None
        for port in self._ports:
            if _port_open(port):
                found = port
                break
        self._port = found
        self._was_up = found is not None
        self._checked_at = now
        return self._was_up

    @property
    def port(self) -> int | None:
        return self._port if self.available() else None

    def proxy_url(self) -> str | None:
        p = self.port
        return f"socks5://127.0.0.1:{p}" if p else None

    def curl_args(self) -> list[str]:
        """curl flags that route the request through the live Tor proxy.

        ``--socks5-hostname`` (not ``--socks5``) sends the hostname to the
        proxy so DNS resolution also happens inside Tor — using ``--socks5``
        would leak every DNS lookup to the local resolver.
        """
        p = self.port
        if not p:
            return []
        return ["--socks5-hostname", f"127.0.0.1:{p}"]

    # -- fetching ---------------------------------------------------------
    def fetch(self, url: str, timeout: float = _FETCH_TIMEOUT,
              extra_args: list[str] | None = None) -> tuple[str | None, str | None]:
        """GET ``url`` through Tor via the curl binary.

        The curl binary is the only transport on the box that speaks SOCKS
        with no extra wheels (httpx needs ``socksio``, requests needs
        ``PySocks``), and it is already a dependency of the bypass cascade.

        Returns (body, None) on success or (None, reason) on failure; a
        failure retires Tor so the caller is not charged again immediately.
        """
        socks = self.curl_args()
        if not socks or not shutil.which("curl"):
            return None, "tor unavailable"
        self._last_used = time.monotonic()
        cmd = ["curl", "-s", "-L", "--compressed", "--max-time", str(int(timeout)),
               *socks, *(extra_args or []), url]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        except (subprocess.TimeoutExpired, OSError) as e:
            self.note_failure()
            return None, f"tor curl: {type(e).__name__}"
        body = proc.stdout or ""
        # curl exits non-zero on network failure; a block page still exits 0,
        # so callers should validate the body themselves.
        if proc.returncode != 0 or not body.strip():
            err = (proc.stderr or "").strip()[:80]
            self.note_failure()
            return None, err or f"curl exit {proc.returncode}"
        return body, None

    # -- diagnostics ------------------------------------------------------
    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "enabled": not (self._disabled or _env_off()),
            "installed": bool(self.binary()),
            "autostart": _env_flag("OPENCODE_TOR_AUTOSTART", True),
            "autostop": _env_flag("OPENCODE_TOR_AUTOSTOP", True),
            "available": self.available(),
            "port": self.port,
            "proxy": self.proxy_url(),
            "owned": bool(self._owned_pid),
            "idle_s": round(max(0.0, now - self._last_used), 1) if self._last_used else None,
            "last_checked_age_s": round(max(0.0, now - self._checked_at), 1)
            if self._checked_at else None,
        }


__all__ = ["TorRotation", "DEFAULT_PORTS"]
