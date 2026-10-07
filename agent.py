#!/usr/bin/env python3
"""ojee-loq agent — hardware telemetry and control over the tailnet.

Runs on the LOQ laptop. The web module talks to this; the desktop app talks to
the same `hardware` and `control` modules directly, so the two can never
disagree about what the machine is doing.

    GET  /module.json      module manifest (lets an ojee-console mount it)
    GET  /api/health       liveness + which controls this machine actually has
    GET  /api/state        one full snapshot
    GET  /api/events       SSE — a snapshot every second
    POST /api/control      {"key": "profile", "value": "performance"}
    POST /api/revert       confirm/cancel a pending advanced change

Stdlib only apart from what hardware.py already needs. No pip install, no
virtualenv: a service you forget about is a service that still works in a year.

Security: binds to the tailnet address and requires a bearer token. The tailnet
is the real boundary; the token stops another peer on it — a phone, a CI
runner — from reclocking your GPU.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import control  # noqa: E402
import hardware as hw  # noqa: E402

def _tailnet_ip(timeout_s: float = 90.0) -> str | None:
    """Wait for this machine to have a Tailscale address.

    `tailscale ip -4` answers nothing for the first few seconds after boot,
    while tailscaled starts and authenticates. systemd's After= only orders
    the START of the unit, not readiness — so asking once, at startup, is a
    race. Losing it meant binding loopback and staying there for the life of
    the process: the agent looked healthy, the port was open, and nothing off
    the machine could reach it. That is the "LOQ doesn't always connect".
    """
    deadline = time.monotonic() + timeout_s
    delay = 0.5
    while True:
        try:
            r = subprocess.run(['tailscale', 'ip', '-4'],
                               capture_output=True, text=True, timeout=5)
            addr = (r.stdout or '').strip().splitlines()
            if r.returncode == 0 and addr and addr[0].strip():
                return addr[0].strip()
        except (OSError, subprocess.SubprocessError):
            pass
        if time.monotonic() >= deadline:
            return None
        time.sleep(delay)
        delay = min(delay * 1.6, 5.0)


def _default_bind() -> str:
    """This machine's Tailscale address, or loopback — never 0.0.0.0.

    0.0.0.0 puts a hardware-control API on every interface the box has. This
    laptop sits on home wifi, so the practical effect of that default is that
    anyone on the LAN can reach CPU power limits and a process-kill endpoint
    with nothing but a guessable token in front of them. Defaulting to the
    tailnet address means the port simply does not exist off the tailnet.

    Falling back to 127.0.0.1 rather than 0.0.0.0 keeps the failure SAFE: if
    Tailscale is not up yet, the agent starts unreachable instead of starting
    wide open. LOQ_BIND overrides this for anyone fronting it differently.
    """
    ip = _tailnet_ip()
    if ip:
        return ip
    # Exit rather than bind loopback. systemd restarts us, and a service that
    # keeps retrying is far better than one that is up, healthy-looking and
    # permanently unreachable. Set LOQ_BIND explicitly to override.
    sys.exit('no Tailscale address after 90s — refusing to bind loopback '
             'silently. Set LOQ_BIND to override.')


HOST = os.environ.get('LOQ_BIND') or _default_bind()
PORT = int(os.environ.get('LOQ_PORT', '8300'))
TOKEN = os.environ.get('LOQ_TOKEN', '')
NAME = os.environ.get('LOQ_NAME', socket.gethostname())
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui')
PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'public')

if not TOKEN:
    sys.exit('LOQ_TOKEN is required. Generate one with: openssl rand -hex 32')

# How long an advanced change survives without confirmation before reverting.
# Chosen so a dropped phone connection cannot leave the machine mis-clocked:
# long enough to tap "keep", short enough that you are still in the room.
REVERT_SECONDS = int(os.environ.get('LOQ_REVERT_SECONDS', '15'))


# ── sampling ───────────────────────────────────────────────────────────

class Sampler:
    """Holds the deltas that per-second rates need.

    CPU, network and disk counters are cumulative, so a rate needs the previous
    sample. Keeping that here rather than recomputing per request means two
    clients watching at once see the same numbers instead of each other's
    half-intervals.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.prev_stat = hw.read_cpu_stat()
        self.prev_cores = hw.read_per_core_stats()
        self.prev_net = hw.read_net_stats()
        self.prev_disk = hw.read_disk_stats()
        self.prev_energy = (hw.read_cpu_package_energy_uj(), time.monotonic())
        self.prev_time = time.monotonic()
        self.cpu_zone = hw.find_cpu_zone()
        self.bat = hw.find_battery()
        hw.fix_rapl_perms()

    def snapshot(self) -> dict:
        with self.lock:
            now = time.monotonic()
            dt = max(0.001, now - self.prev_time)

            stat = hw.read_cpu_stat()
            cores = hw.read_per_core_stats()
            net = hw.read_net_stats()
            disk = hw.read_disk_stats()

            cpu_pct = _delta_pct(self.prev_stat, stat)
            core_pct = [_delta_pct(a, b) for a, b in zip(self.prev_cores, cores)]

            # read_net_stats/read_disk_stats return {device: (in, out)}, not a
            # pair — sum across devices so a machine with several NICs or two
            # NVMe drives reports its real total rather than one arbitrary one.
            rx, tx = _sum_delta(self.prev_net, net, dt)
            rd, wr = _sum_delta(self.prev_disk, disk, dt)

            energy, e_at = hw.read_cpu_package_energy_uj(), now
            cpu_w = 0.0
            if energy is not None and self.prev_energy[0] is not None:
                de = energy - self.prev_energy[0]
                if de < 0:                       # counter wrapped
                    de += hw.read_cpu_max_energy_uj() or 0
                cpu_w = max(0.0, de / 1e6 / max(0.001, e_at - self.prev_energy[1]))
            self.prev_energy = (energy, e_at)

            self.prev_stat, self.prev_cores = stat, cores
            self.prev_net, self.prev_disk = net, disk
            self.prev_time = now

        # nvidia_query returns a positional LIST in the order asked for, not a
        # dict — zip it back to names rather than indexing, so adding a field
        # later cannot silently shift every value by one.
        GPU_FIELDS = ('utilization.gpu', 'temperature.gpu', 'clocks.sm',
                      'clocks.mem', 'memory.used', 'memory.total', 'power.draw')
        gpu = dict(zip(GPU_FIELDS, hw.nvidia_query(list(GPU_FIELDS)) or []))
        fan_cpu, fan_gpu = hw.read_fan_rpm()
        throttled = _throttle_count()

        return {
            'at': time.time(),
            'name': NAME,
            'cpu': {
                'model': hw.cpu_model_short(),
                'usage': round(cpu_pct, 1),
                'cores': [round(c, 1) for c in core_pct],
                'tempC': _temp(self.cpu_zone),
                'watts': round(cpu_w, 1),
                'tdp': _current_tdp(),
                'throttled': throttled,
            },
            'gpu': {
                'model': hw.gpu_model_short(),
                'usage': _num(gpu.get('utilization.gpu')),
                'tempC': _num(gpu.get('temperature.gpu')),
                'clockMhz': _num(gpu.get('clocks.sm')),
                'memClockMhz': _num(gpu.get('clocks.mem')),
                'vramUsedMb': _num(gpu.get('memory.used')),
                'vramTotalMb': _num(gpu.get('memory.total')),
                'watts': _num(gpu.get('power.draw')),
            },
            'fans': {'cpuRpm': fan_cpu, 'gpuRpm': fan_gpu},
            'battery': _battery(self.bat),
            'io': {
                'netRxPerS': round(rx), 'netTxPerS': round(tx),
                'diskReadPerS': round(rd), 'diskWritePerS': round(wr),
            },
            'state': _control_state(),
            'drives': hw.list_drives(),
            # Mounted filesystems with usage, and what the machine says it
            # runs. Both are what a fleet view needs and the drives list
            # (hardware only, no usage) cannot give it.
            'mounts': hw.list_mounts(),
            'os': hw.os_name(),
            'nics': hw.list_active_nics(),
        }


def _sum_delta(prev: dict, cur: dict, dt: float) -> tuple[float, float]:
    """Per-second rates summed over every device present in BOTH samples.

    Intersecting the keys matters: a NIC that appears mid-interval (a VPN going
    up, a drive being plugged in) has no previous counter, and treating its
    absolute total as a delta produces an absurd one-off spike.
    """
    a = b = 0.0
    for dev, (ci, co) in cur.items():
        if dev not in prev:
            continue
        pi, po = prev[dev]
        a += max(0, ci - pi)
        b += max(0, co - po)
    return a / dt, b / dt


def _delta_pct(prev, cur) -> float:
    """CPU busy percentage between two /proc/stat samples."""
    try:
        pt, pi = prev
        ct, ci = cur
        dt, di = ct - pt, ci - pi
        return 0.0 if dt <= 0 else max(0.0, min(100.0, (dt - di) / dt * 100.0))
    except Exception:                                # noqa: BLE001
        return 0.0


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _temp(zone):
    raw = hw.read_sys(f'{zone}/temp')
    try:
        return round(int(raw) / 1000, 1)
    except (TypeError, ValueError):
        return None


def _current_tdp():
    raw = hw.read_sys(f'{hw.RAPL}/constraint_0_power_limit_uw')
    try:
        return round(int(raw) / 1_000_000)
    except (TypeError, ValueError):
        return None


def _throttle_count():
    raw = hw.read_sys('/sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count')
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _battery(path):
    if not path:
        return None
    def n(f):
        try:
            return int(hw.read_sys(f'{path}/{f}') or 0)
        except ValueError:
            return 0
    full, design = n('energy_full'), n('energy_full_design')
    return {
        'percent': n('capacity'),
        'status': hw.read_sys(f'{path}/status'),
        'cycles': n('cycle_count'),
        # Health is the number people actually want; deriving it here means the
        # web and desktop views cannot disagree about it.
        'healthPct': round(full / design * 100) if full and design else None,
        'powerW': round(n('power_now') / 1e6, 1) if n('power_now') else None,
        'history': hw.battery_history_summary(),
    }


_SLOW_STATE = {'at': 0.0, 'value': {}}
_SLOW_TTL = 10.0


def _slow_control_state():
    """The three controls that cost a subprocess to read, cached.

    profile/conservation/fnLock/kbdBacklight are sysfs reads — free, and read
    every tick. gpuMode, micMuted and touchpad each fork a process, and doing
    that three times a second for a value that changes maybe twice a day is
    how a monitoring agent becomes the thing worth monitoring.
    """
    now = time.monotonic()
    if now - _SLOW_STATE['at'] < _SLOW_TTL:
        return _SLOW_STATE['value']

    out = {}
    try:
        r = subprocess.run(['prime-select', 'query'], capture_output=True,
                           text=True, timeout=5)
        if r.returncode == 0:
            out['gpuMode'] = r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        r = subprocess.run(['pactl', 'get-source-mute', '@DEFAULT_SOURCE@'],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            out['micMuted'] = 'yes' in r.stdout.lower()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        r = subprocess.run(['gsettings', 'get',
                            'org.gnome.desktop.peripherals.touchpad', 'send-events'],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            out['touchpad'] = 'disabled' not in r.stdout
    except (OSError, subprocess.SubprocessError):
        pass

    _SLOW_STATE.update(at=now, value=out)
    return out


def _control_state():
    ideapad = control.IDEAPAD
    return {
        'profile': hw.read_sys(control.PLATFORM_PROFILE),
        'conservation': hw.read_sys(f'{ideapad}/conservation_mode') == '1',
        'fnLock': hw.read_sys(f'{ideapad}/fn_lock') == '1',
        'kbdBacklight': int(hw.read_sys(control.KBD_BACKLIGHT) or 0),
        **_slow_control_state(),
    }


SAMPLER = Sampler()


# ── pending advanced changes ───────────────────────────────────────────

class Pending:
    """An advanced change that reverts unless confirmed.

    This is the safety net for the case that actually worries me: someone drags
    a GPU clock slider on a phone, the connection drops, and nobody is sitting
    at the laptop. Without this, the machine stays mis-clocked until a human
    walks over to it.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.item = None        # {key, previous, deadline, timer}

    def arm(self, key, previous):
        with self.lock:
            self._cancel_locked()
            timer = threading.Timer(REVERT_SECONDS, self._fire, args=(key, previous))
            timer.daemon = True
            timer.start()
            self.item = {'key': key, 'previous': previous,
                         'deadline': time.time() + REVERT_SECONDS, 'timer': timer}

    def _fire(self, key, previous):
        try:
            control.apply(key, previous)
            print(f'[revert] {key} not confirmed — restored to {previous}', flush=True)
        except Exception as e:                        # noqa: BLE001
            print(f'[revert] {key} FAILED to restore: {e}', flush=True)
        with self.lock:
            self.item = None

    def confirm(self):
        with self.lock:
            if not self.item:
                return None
            key = self.item['key']
            self._cancel_locked()
            return key

    def revert_now(self):
        with self.lock:
            if not self.item:
                return None
            key, prev = self.item['key'], self.item['previous']
            self._cancel_locked()
        control.apply(key, prev)
        return key

    def _cancel_locked(self):
        if self.item and self.item.get('timer'):
            self.item['timer'].cancel()
        self.item = None

    def describe(self):
        with self.lock:
            if not self.item:
                return None
            return {'key': self.item['key'],
                    'revertsInMs': max(0, round((self.item['deadline'] - time.time()) * 1000))}


PENDING = Pending()

class ProcSampler:
    """Top processes by CPU, sampled at most once a second.

    Separate from Sampler because walking /proc costs ~700 open() pairs and the
    state stream ticks every second whether or not anyone is looking at the
    process table. This only runs when the System view asks.
    """

    MIN_INTERVAL = 0.7

    def __init__(self):
        self.lock = threading.Lock()
        self.prev = {}
        self.prev_total = 0
        self.at = 0.0
        self.cached = []

    def top(self, limit=40, query=''):
        with self.lock:
            now = time.monotonic()
            if now - self.at >= self.MIN_INTERVAL:
                table = hw.read_proc_table()
                total = hw.read_cpu_total_jiffies()
                delta = max(total - self.prev_total, 1)
                mem_total = hw.mem_total_kb()

                rows = []
                for pid, cur in table.items():
                    was = self.prev.get(pid)
                    # A process first seen this tick has no baseline; reporting
                    # its lifetime average as instantaneous CPU would put every
                    # freshly-spawned process at the top of the table.
                    pct = 0.0 if was is None else max(
                        0.0, (cur['jiffies'] - was['jiffies']) / delta * 100)
                    rows.append({
                        'pid': pid,
                        'name': cur['name'],
                        'user': cur['user'],
                        'own': cur['uid'] == os.getuid(),
                        'cpuPct': round(pct, 1),
                        'memPct': round(cur['memKb'] / max(mem_total, 1) * 100, 1),
                        'memKb': cur['memKb'],
                        'state': cur['state'],
                        'cmdline': cur['cmdline'][:200],
                    })

                self.prev, self.prev_total, self.at = table, total, now
                self.cached = sorted(rows, key=lambda r: (-r['cpuPct'], -r['memKb']))

            rows = self.cached
            if query:
                q = query.lower()
                rows = [r for r in rows
                        if q in r['name'].lower() or q in r['cmdline'].lower()
                        or q == str(r['pid'])]
            return {'total': len(self.cached), 'matched': len(rows),
                    'processes': rows[:max(1, min(200, limit))]}


PROCS = ProcSampler()

# Killing is bounded by the AGENT'S OWN UID, deliberately, and never routed
# through sudo. The web module can end exactly what the agent user could end
# from a shell on that machine — no more. Signalling PID 1 or a root daemon
# over a tailnet token is not a feature anybody asked for, and the scoped
# sudoers file exists to keep the token from becoming a root shell.
KILL_SIGNALS = {'TERM': signal.SIGTERM, 'KILL': signal.SIGKILL,
                'INT': signal.SIGINT, 'HUP': signal.SIGHUP}


def kill_process(pid: int, sig: str = 'TERM') -> dict:
    if pid <= 1:
        raise control.ControlError('refusing to signal PID %d' % pid,
                                   status=400, code='pid_protected')
    signum = KILL_SIGNALS.get(str(sig).upper())
    if signum is None:
        raise control.ControlError(
            'unknown signal %r; have %s' % (sig, ', '.join(KILL_SIGNALS)),
            status=400, code='unknown_signal')
    try:
        st = os.stat('/proc/%d' % pid)
    except FileNotFoundError:
        raise control.ControlError('no such process', status=404,
                                   code='no_such_process') from None
    if st.st_uid != os.getuid() and os.getuid() != 0:
        raise control.ControlError(
            'process belongs to uid %d; the agent runs as %d and does not '
            'escalate to signal it' % (st.st_uid, os.getuid()),
            status=403, code='not_owned')
    try:
        os.kill(pid, signum)
    except ProcessLookupError:
        raise control.ControlError('no such process', status=404,
                                   code='no_such_process') from None
    except PermissionError:
        raise control.ControlError('permission denied', status=403,
                                   code='not_owned') from None
    return {'pid': pid, 'signal': sig.upper()}



# ── standalone session ─────────────────────────────────────────────────
# Mounted behind ojee-console the gateway asserts identity and this never
# runs. Standalone, a browser needs SOME way in, and EventSource cannot set
# an Authorization header — which is why the agent also accepts ?token=.
# Putting a cookie in front of both means the generic ojee-ui shell works
# unmodified: same-origin fetch and EventSource send cookies on their own.

SESSION_COOKIE = 'loq_session'
SESSION_MAX_AGE = 30 * 24 * 3600


def _session_value() -> str:
    """Derived from the token, so rotating the token invalidates every cookie.

    Not a random session id: there is exactly one principal here and no server
    state worth keeping. It is never the token itself, so a leaked cookie from
    a browser profile cannot be replayed as a bearer token against the API.
    """
    return hmac.new(TOKEN.encode(), b'loq-standalone-session', hashlib.sha256).hexdigest()


def _has_session(headers) -> bool:
    raw = headers.get('cookie') or ''
    for part in raw.split(';'):
        name, _, value = part.strip().partition('=')
        if name == SESSION_COOKIE:
            return hmac.compare_digest(value, _session_value())
    return False


MANIFEST = {
    'id': 'loq',
    'name': 'LOQ',
    'version': '1.0.0',
    # Monitor and Battery only. The browser module READS this machine; it does
    # not drive it.
    #
    # Three surfaces — the Qt app, the browser on a desktop, the browser on a
    # phone — had drifted into three different subsets of the same controls,
    # which is worse than one surface having them and one not. Controls live
    # in the desktop app, where a human is sitting at the machine. That also
    # means the auto-revert, the confirm dialogs and the privileged write
    # paths no longer need to be reachable over the tailnet at all.
    #
    # /api/control, /api/revert and /api/kill still EXIST — the desktop app
    # and any future client use them, and they stay gated by the token and by
    # LOQ_ALLOW_ADVANCED. The web UI simply stops calling them.
    'views': [
        {'id': 'monitor', 'label': 'Monitor', 'icon': 'i-gauge'},
        {'id': 'battery', 'label': 'Battery', 'icon': 'i-temp'},
    ],
    'ui': '/ui/index.js',
    'health': '/api/health',
    'capabilities': ['sse', 'commands', 'processes'],
}


class Handler(BaseHTTPRequestHandler):
    server_version = 'ojee-loq-agent/1.0'

    def log_message(self, fmt, *args):
        pass                # the gateway health-polls; do not fill the journal

    # ── plumbing ───────────────────────────────────────────────────────
    def _send(self, code, payload, ctype='application/json'):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('content-type', ctype)
        self.send_header('content-length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _token(self):
        auth = self.headers.get('authorization', '')
        if auth.startswith('Bearer '):
            return auth[7:]
        if '?' in self.path:
            from urllib.parse import parse_qs, urlparse
            return (parse_qs(urlparse(self.path).query).get('token') or [''])[0]
        return ''

    def _guard(self):
        if hmac.compare_digest(self._token(), TOKEN) or _has_session(self.headers):
            return True
        self._send(401, {'error': 'unauthorized'})
        return False

    def _wants_html(self):
        # Sec-Fetch-Mode is set by every browser navigation and by nothing
        # else, so a fetch() for JSON never gets an HTML login page back.
        if self.headers.get('sec-fetch-mode') == 'navigate':
            return True
        return 'text/html' in (self.headers.get('accept') or '')

    def _path(self):
        return self.path.split('?')[0]

    def _static(self, root, rel):
        # Normalised and confined: a module's own assets, nothing above them.
        full = os.path.normpath(os.path.join(root, rel.lstrip('/')))
        if not full.startswith(os.path.realpath(root)):
            self._send(403, {'error': 'forbidden'})
            return True
        if not os.path.isfile(full):
            return False
        ctype = ('text/javascript; charset=utf-8' if full.endswith('.js')
                 else 'text/css; charset=utf-8' if full.endswith('.css')
                 else 'image/svg+xml' if full.endswith('.svg')
                 else 'text/html; charset=utf-8' if full.endswith('.html')
                 else 'application/octet-stream')
        with open(full, 'rb') as fh:
            body = fh.read()
        # no-cache means REVALIDATE, not "never cache" — the browser still
        # keeps the copy, it just asks first. Everything here is a few KB over
        # a tailnet, and a stale /ui/index.js after an agent upgrade is a bug
        # that looks exactly like the new code not working.
        self.send_response(200)
        self.send_header('content-type', ctype)
        self.send_header('content-length', str(len(body)))
        self.send_header('cache-control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)
        return True

    # ── routes ─────────────────────────────────────────────────────────
    def do_GET(self):
        path = self._path()

        if path == '/module.json':
            return self._send(200, MANIFEST)

        if path.startswith('/ui/'):
            if self._static(UI_DIR, path[4:]):
                return
            return self._send(404, {'error': 'not_found'})

        if path == '/api/health':
            if not self._guard():
                return
            caps = control.capabilities()
            return self._send(200, {
                'ok': True,
                'name': NAME,
                'controls': caps,
                'advanced': caps['advancedEnabled'],
                'pending': PENDING.describe(),
            })

        if path == '/api/state':
            if not self._guard():
                return
            return self._send(200, {**SAMPLER.snapshot(),
                                    'controls': control.capabilities(),
                                    'pending': PENDING.describe()})

        if path == '/api/processes':
            if not self._guard():
                return
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            return self._send(200, PROCS.top(
                limit=int((q.get('limit') or ['40'])[0] or 40),
                query=(q.get('q') or [''])[0]))

        if path == '/api/events':
            if not self._guard():
                return
            return self._sse()

        # Standalone shell. A browser navigating here without a session gets
        # the login page rather than the app skeleton — the app would only
        # render a wall of 401s.
        if self._wants_html() and not _has_session(self.headers) and not self._token():
            if self._static(PUBLIC_DIR, 'login.html'):
                return

        rel = 'index.html' if path == '/' else path
        if self._static(PUBLIC_DIR, rel):
            return
        self._send(404, {'error': 'not_found', 'path': path})

    def _sse(self):
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        self.send_header('cache-control', 'no-cache, no-transform')
        self.send_header('x-accel-buffering', 'no')
        self.end_headers()
        try:
            while True:
                payload = {**SAMPLER.snapshot(), 'pending': PENDING.describe()}
                self.wfile.write(b'data: ' + json.dumps(payload).encode() + b'\n\n')
                self.wfile.flush()
                time.sleep(1)
        except (BrokenPipeError, ConnectionResetError):
            pass            # the client went away; entirely normal

    def do_POST(self):
        path = self._path()

        if path == '/api/session':
            length = int(self.headers.get('content-length') or 0)
            try:
                body = json.loads(self.rfile.read(min(length, 4096)) or b'{}')
            except json.JSONDecodeError:
                return self._send(400, {'error': 'bad_json'})
            # Constant-time, and rate-limited by the tailnet being the only
            # thing that can reach this port at all.
            if not hmac.compare_digest(str(body.get('token') or ''), TOKEN):
                time.sleep(0.5)
                return self._send(401, {'error': 'bad_token'})
            body_b = json.dumps({'ok': True}).encode()
            self.send_response(200)
            self.send_header('content-type', 'application/json')
            self.send_header('content-length', str(len(body_b)))
            self.send_header(
                'set-cookie',
                f'{SESSION_COOKIE}={_session_value()}; Max-Age={SESSION_MAX_AGE}; '
                f'Path=/; HttpOnly; SameSite=Strict')
            self.end_headers()
            return self.wfile.write(body_b)

        if not self._guard():
            return

        length = int(self.headers.get('content-length') or 0)
        if length > 8192:
            return self._send(413, {'error': 'too_large'})
        try:
            body = json.loads(self.rfile.read(length) or b'{}')
        except json.JSONDecodeError:
            return self._send(400, {'error': 'bad_json'})

        if path == '/api/control':
            key, value = body.get('key'), body.get('value')
            if not key:
                return self._send(400, {'error': 'key_required'})
            try:
                previous = _current_value(key)
                result = control.apply(key, value)
                _SLOW_STATE['at'] = 0.0
                # Advanced changes arm the auto-revert unless the caller opts
                # out (the desktop app, sitting at the machine, does).
                if key in control.ADVANCED and body.get('confirm') is not True:
                    PENDING.arm(key, previous)
                return self._send(200, {**result, 'pending': PENDING.describe()})
            except control.ControlError as e:
                return self._send(e.status, {'error': e.code, 'detail': str(e)})
            except Exception as e:                    # noqa: BLE001
                return self._send(500, {'error': 'control_failed',
                                        'detail': f'{type(e).__name__}: {e}'})

        if path == '/api/kill':
            try:
                return self._send(200, kill_process(int(body.get('pid') or 0),
                                                    body.get('signal') or 'TERM'))
            except control.ControlError as e:
                return self._send(e.status, {'error': e.code, 'detail': str(e)})
            except (TypeError, ValueError):
                return self._send(400, {'error': 'bad_pid'})

        if path == '/api/revert':
            if body.get('confirm'):
                return self._send(200, {'confirmed': PENDING.confirm()})
            return self._send(200, {'reverted': PENDING.revert_now()})

        self._send(404, {'error': 'not_found', 'path': path})


def _current_value(key):
    """What a control is set to now — the value auto-revert restores."""
    state = _control_state()
    if key in state:
        return state[key]
    snap = SAMPLER.snapshot()
    return {
        'cpuTdp': snap['cpu']['tdp'],
        'gpuClock': None,          # released rather than restored to a guess
        'gpuMemClock': None,
        'gpuTgp': 45,
    }.get(key)


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    caps = control.capabilities()
    print(f'ojee-loq-agent  {NAME}  http://{HOST}:{PORT}', flush=True)
    print(f'  cpu       {hw.cpu_model_short()}', flush=True)
    print(f'  gpu       {hw.gpu_model_short()}', flush=True)
    print(f'  controls  {sum(1 for k, v in caps.items() if v and k != "advancedEnabled")} available', flush=True)
    print(f'  advanced  {"ENABLED" if caps["advancedEnabled"] else "disabled (LOQ_ALLOW_ADVANCED=1 to permit)"}',
          flush=True)
    print(f'  revert    {REVERT_SECONDS}s auto-revert on unconfirmed advanced changes', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == '__main__':
    main()
