"""Hardware access for Lenovo LOQ laptops — sysfs, RAPL, nvidia-smi, EC.

Extracted from loq-control.py so the desktop app and the web agent read the
SAME code. They previously could not: the agent did not exist, and adding one
that reimplemented these paths would have produced two subtly different views
of the same machine, which is the sort of divergence nobody notices until a
number is wrong in one place only.

Everything here is read-only or a narrowly-scoped write, and none of it imports
Qt — the agent runs headless on a machine that may have no session at all.

Paths are as verified on a Lenovo LOQ 16IRX9 (83JE), i7-14700HX, RTX 5060,
kernel 6.8+:

  platform_profile   /sys/firmware/acpi/platform_profile        (legion_laptop)
  fan RPM            /sys/kernel/debug/legion/fancurve          (WMI3 method —
                     the default EC method returns 0 on this model)
  conservation mode  ideapad_acpi VPC2004:00/conservation_mode
  CPU TDP            intel-rapl:0/constraint_0_power_limit_uw
  battery            BAT1, not BAT0, on LOQ models
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

# ── paths and limits ───────────────────────────────────────────────────
CONFIG_DIR = os.path.expanduser('~/.config/loq-control')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.json')
HISTORY_FILE = os.path.join(CONFIG_DIR, 'battery_history.csv')
DEFAULT_CONFIG = {
    'theme': 'dark',
    'oc_power_limit': 0,
    'oc_gpu_clock': 0,
    'oc_mem_clock_idx': 0,
    'auto_apply_oc': False,
}
RAPL = '/sys/class/powercap/intel-rapl:0'
GPU_CLOCK_MIN = 180
GPU_CLOCK_MAX = 3090
GPU_CLOCK_STEP = 5


def load_config():
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)
        return cfg
    except Exception:
        return dict(DEFAULT_CONFIG)

def save_config(cfg):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)

def _gpu_clock_snap(v):
    if v < GPU_CLOCK_MIN // 2:
        return 0
    if v < GPU_CLOCK_MIN:
        return GPU_CLOCK_MIN
    rem = (v - GPU_CLOCK_MIN) % GPU_CLOCK_STEP
    if rem == 0:
        return min(GPU_CLOCK_MAX, v)
    up = GPU_CLOCK_STEP - rem
    snapped = v - rem if rem <= up else v + up
    return max(GPU_CLOCK_MIN, min(GPU_CLOCK_MAX, snapped))

def read_sys(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return ''

def run_cmd(cmd):
    """DESKTOP APP ONLY. Never call this from the agent.

    This is `sudo sh -c <string>` — a root shell. It is acceptable in the Qt
    app, where sudo prompts a human who is sitting at the machine, and it is
    NOT acceptable in the agent, where the caller is whoever holds a bearer
    token on the tailnet. sudoers.d/ojee-loq deliberately does not grant it,
    so an agent code path that reached here would fail rather than escalate.

    Agent-side privileged work goes through control._priv() -> loq-privhelper.
    """
    subprocess.run(['sudo', 'sh', '-c', cmd],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def set_gpu_tgp(watts):
    # nvidia-smi -pl is locked on mobile Blackwell; route through Lenovo
    # WMAE method which writes EC GTGP and notifies NVIDIA NPCF.
    w = max(45, min(100, int(watts)))
    run_cmd(
        "modprobe acpi_call 2>/dev/null; "
        "echo '\\_SB.GZFD.WMAE 0 0x12 "
        f"{{0x00, 0x00, 0x02, 0x02, 0x{w:02X}, 0x00, 0x00, 0x00}}' "
        "> /proc/acpi/call")

def run_output(cmd):
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ''

HELPER = os.environ.get('LOQ_HELPER', '/usr/local/lib/ojee-loq/loq-privhelper')


def _priv_read(verb: str, fallback: list[str]) -> str:
    """Run a read-only privileged step, preferring the helper.

    This module is shared by BOTH surfaces, and they have different privilege
    stories:

      agent    installs loq-privhelper and is granted NOPASSWD on it alone.
               `fallback` will fail there, which is correct and harmless.
      desktop  installs no helper and no sudoers at all — sudo prompts a human
               who is sitting at the machine. Without `fallback`, running only
               ./install.sh would silently lose fan RPM and CPU watts.

    So: helper if it exists, otherwise the direct command. Never the reverse,
    or the agent would use the broad grant whenever it happened to be present.
    """
    if os.path.exists(HELPER):
        cmd = ['sudo', '-n', HELPER, verb]
    else:
        cmd = fallback
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=5)
        return r.stdout or ''
    except (OSError, subprocess.SubprocessError):
        return ''


def read_fan_rpm():
    """CPU and GPU fan RPM from legion_laptop's debugfs node.

    Via the privileged helper, NOT `sudo cat`: granting the agent user
    passwordless `cat` is a read primitive for every root-owned file on the
    machine. The helper greps the two lines this needs and nothing else.
    """
    try:
        out = _priv_read('fanrpm',
                         ['sudo', 'cat', '/sys/kernel/debug/legion/fancurve'])
        cpu, gpu = 0, 0
        for line in out.split('\n'):
            s = line.strip()
            if s.startswith('1 fanspeed WMI3:'):
                cpu = int(s.split(':')[1])
            elif s.startswith('2 fanspeed WMI3:'):
                gpu = int(s.split(':')[1])
        return cpu, gpu
    except Exception:
        return 0, 0

def nvidia_query(fields):
    if isinstance(fields, str):
        fields = [fields]
    r = run_output(['nvidia-smi', f'--query-gpu={",".join(fields)}',
                    '--format=csv,noheader,nounits'])
    return [v.strip() for v in r.split(',')] if r else [''] * len(fields)

def get_gpu_power():
    output = run_output(['nvidia-smi', '-q', '-d', 'POWER'])
    info = {'current': 0, 'default': 45, 'min': 5, 'max': 100}
    found, in_mod = set(), False
    for line in output.split('\n'):
        s = line.strip()
        if 'Module Power' in s or 'GPU Memory Power' in s:
            in_mod = True; continue
        if 'GPU Power Readings' in s:
            in_mod = False; continue
        if in_mod or ':' not in s:
            continue
        for pat, key in [('Current Power Limit', 'current'),
                         ('Default Power Limit', 'default'),
                         ('Min Power Limit', 'min'),
                         ('Max Power Limit', 'max')]:
            if s.startswith(pat) and key not in found:
                try:
                    info[key] = float(s.split(':')[1].replace('W', '').strip())
                    found.add(key)
                except (ValueError, IndexError):
                    pass
    return info

def find_cpu_zone():
    for p in sorted(glob.glob('/sys/class/thermal/thermal_zone*')):
        if read_sys(f'{p}/type') in ('x86_pkg_temp', 'TCPU', 'cpu-thermal'):
            return p
    return '/sys/class/thermal/thermal_zone0'

def find_battery():
    for n in ['BAT0', 'BAT1']:
        p = f'/sys/class/power_supply/{n}'
        if read_sys(f'{p}/type') == 'Battery':
            return p
    return None

def fix_rapl_perms():
    """RAPL energy_uj is root-only on modern kernels (PLATYPUS
    mitigation). Make the package energy counters readable so the
    monitor can sample CPU watts without spawning sudo each tick."""
    paths = glob.glob('/sys/class/powercap/intel-rapl:[0-9]*/energy_uj')
    if not paths:
        return
    # Via the helper, which globs the paths itself. `sudo chmod` granted to the
    # agent user would let it make any root-owned file world-readable — so that
    # form survives only as the desktop app's fallback.
    _priv_read('raplperms', ['sudo', '-n', 'chmod', 'a+r', *paths])

def read_cpu_package_energy_uj():
    """Cumulative CPU package energy counter in microjoules (RAPL).

    Take deltas between samples to derive average power over the
    interval. Returns None if unavailable.
    """
    raw = read_sys(f'{RAPL}/energy_uj')
    try:
        return int(raw) if raw else None
    except ValueError:
        return None

def read_cpu_max_energy_uj():
    raw = read_sys(f'{RAPL}/max_energy_range_uj')
    try:
        return int(raw) if raw else None
    except ValueError:
        return None

def read_gpu_power_draw():
    """Instantaneous GPU power draw in W via nvidia-smi power.draw."""
    out = run_output([
        'nvidia-smi', '--query-gpu=power.draw',
        '--format=csv,noheader,nounits'])
    try:
        return float((out or '').strip())
    except (ValueError, AttributeError):
        return None

def cpu_model_short():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    full = line.split(':', 1)[1].strip()
                    s = (full.replace('(R)', '').replace('(TM)', '')
                         .replace('Intel ', '').replace('  ', ' ').strip())
                    return s
    except Exception:
        pass
    return ''

def gpu_model_short():
    name = nvidia_query('name')[0] or ''
    vram = nvidia_query('memory.total')[0] or ''
    n = (name.replace('NVIDIA ', '').replace('GeForce ', '')
         .replace(' GPU', '').replace(' Laptop', ' Laptop'))
    if vram:
        try:
            gb = round(int(vram) / 1024)
            return f'{n.strip()} · {gb} GB'
        except ValueError:
            pass
    return n.strip()

def fmt_bytes(n):
    if n >= 1099511627776:
        return f'{n / 1099511627776:.1f} TB'
    if n >= 1073741824:
        return f'{n / 1073741824:.1f} GB'
    if n >= 1048576:
        return f'{n / 1048576:.0f} MB'
    if n >= 1024:
        return f'{n / 1024:.0f} KB'
    return f'{n} B'

def list_drives():
    """Whole-disk block devices we care about, with model and size."""
    drives = []
    try:
        paths = (glob.glob('/sys/block/nvme*n*')
                 + glob.glob('/sys/block/sd*')
                 + glob.glob('/sys/block/mmcblk*'))
        for path in sorted(paths):
            dev = os.path.basename(path)
            if dev.startswith('nvme') and 'p' in dev.split('n', 1)[-1]:
                continue
            size_sec = int(read_sys(f'{path}/size') or '0')
            size = size_sec * 512
            if size <= 0:
                continue
            model = (read_sys(f'{path}/device/model')
                     or read_sys(f'{path}/device/name') or '').strip()
            drives.append({'dev': dev, 'model': model, 'size': size})
    except Exception:
        pass
    return drives

def drive_temp(dev):
    """NVMe / SATA drive temperature in °C, or None if unavailable."""
    try:
        for hwmon in glob.glob(f'/sys/block/{dev}/device/hwmon/hwmon*'):
            for tf in sorted(glob.glob(f'{hwmon}/temp*_input')):
                raw = read_sys(tf)
                if raw:
                    return int(raw) // 1000
        for hwmon in glob.glob('/sys/class/nvme/*/hwmon*'):
            ctrl = os.path.basename(os.path.dirname(hwmon))
            if dev.startswith(ctrl):
                raw = read_sys(f'{hwmon}/temp1_input')
                if raw:
                    return int(raw) // 1000
    except Exception:
        pass
    return None

def _unescape_mount_path(s):
    """/proc/mounts encodes spaces (and a few other chars) as backslash-
    octal sequences (e.g. \\040 for ' '). os.statvfs needs the real path."""
    out, i = [], 0
    while i < len(s):
        if s[i] == '\\' and i + 3 < len(s) and s[i+1:i+4].isdigit():
            try:
                out.append(chr(int(s[i+1:i+4], 8)))
                i += 4
                continue
            except ValueError:
                pass
        out.append(s[i]); i += 1
    return ''.join(out)

def drive_usage(dev):
    """Best-effort (used, total) bytes for the largest partition on this drive."""
    try:
        with open('/proc/mounts') as f:
            mounts = []
            for line in f:
                p = line.split()
                if len(p) >= 2 and p[0].startswith(f'/dev/{dev}'):
                    mounts.append(_unescape_mount_path(p[1]))
        best_total = 0; best_used = 0
        for mp in mounts:
            try:
                s = os.statvfs(mp)
                total = s.f_blocks * s.f_frsize
                used = (s.f_blocks - s.f_bavail) * s.f_frsize
                if total > best_total:
                    best_total = total; best_used = used
            except OSError:
                pass
        return best_used, best_total
    except Exception:
        return 0, 0

# Filesystems that are not on a disk in this machine. statvfs on a network
# mount asks the far end, and an sshfs whose server went away blocks the
# caller indefinitely; the sampler answers once a second and must not.
_NETWORK_FS = ('nfs', 'nfs4', 'cifs', 'smb3', 'smbfs', 'fuse.sshfs',
               'fuse.rclone', 'fuse.gvfsd-fuse', '9p', 'ceph', 'glusterfs')
_SKIP_MOUNT_PREFIXES = ('/snap', '/var/snap', '/var/lib/docker',
                        '/var/lib/containerd', '/run', '/sys', '/proc', '/dev')

def _parent_disk(dev):
    """sda1 -> sda, nvme0n1p2 -> nvme0n1, dm-0 -> its first slave's disk."""
    name = os.path.basename(os.path.realpath(dev))
    base = f'/sys/class/block/{name}'
    if not os.path.exists(base):
        return name
    slaves = glob.glob(f'{base}/slaves/*')
    if slaves:
        return _parent_disk('/dev/' + os.path.basename(slaves[0]))
    if os.path.exists(f'{base}/partition'):
        return os.path.basename(os.path.dirname(os.path.realpath(base)))
    return name

def list_mounts():
    """Every real filesystem mounted from a local block device, one row per
    device (the shortest mount point wins, so bind mounts do not repeat it),
    with size, used, free and the physical disk it lives on."""
    seen = {}
    try:
        with open('/proc/self/mounts') as f:
            lines = f.readlines()
    except OSError:
        return []
    for line in lines:
        p = line.split()
        if len(p) < 3:
            continue
        dev, mp, fstype = p[0], _unescape_mount_path(p[1]), p[2]
        if not dev.startswith('/dev/') or fstype in _NETWORK_FS or dev.startswith('/dev/loop'):
            continue
        if fstype in ('squashfs', 'overlay', 'tmpfs', 'devtmpfs'):
            continue
        if any(mp == x or mp.startswith(x + '/') for x in _SKIP_MOUNT_PREFIXES):
            continue
        if dev not in seen or len(mp) < len(seen[dev][0]):
            seen[dev] = (mp, fstype)
    out = []
    for dev, (mp, fstype) in seen.items():
        try:
            s = os.statvfs(mp)
        except OSError:
            continue
        total = s.f_blocks * s.f_frsize
        if total <= 0:
            continue
        free = s.f_bavail * s.f_frsize
        out.append({'mount': mp, 'device': dev, 'disk': _parent_disk(dev),
                    'fstype': fstype, 'total': total,
                    'used': total - s.f_bfree * s.f_frsize, 'free': free})
    out.sort(key=lambda m: -m['total'])
    return out

_OS_NAME = None

def os_name():
    """PRETTY_NAME from os-release ("Zorin OS 18.1"), else NAME + VERSION.
    Read once: it does not change while the agent is running."""
    global _OS_NAME
    if _OS_NAME is not None:
        return _OS_NAME or None
    kv = {}
    for path in ('/etc/os-release', '/usr/lib/os-release'):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if '=' not in line or line.startswith('#'):
                        continue
                    k, v = line.split('=', 1)
                    v = v.strip()
                    if len(v) >= 2 and v[0] == v[-1] and v[0] in '"\'':
                        v = v[1:-1]
                    kv[k] = v.replace('\\"', '"')
            break
        except OSError:
            continue
    name = kv.get('PRETTY_NAME') or ' '.join(
        x for x in (kv.get('NAME'), kv.get('VERSION') or kv.get('VERSION_ID')) if x)
    _OS_NAME = name or ''
    return name or None

def list_active_nics():
    """Active non-loopback interfaces with type, SSID/IP info."""
    nics = []
    try:
        for path in sorted(glob.glob('/sys/class/net/*')):
            iface = os.path.basename(path)
            if iface == 'lo':
                continue
            operstate = (read_sys(f'{path}/operstate') or 'unknown').strip()
            carrier = (read_sys(f'{path}/carrier') or '0').strip()
            if operstate != 'up' and carrier != '1':
                continue
            if os.path.isdir(f'{path}/wireless'):
                kind = 'wifi'
            elif iface.startswith(('en', 'eth')):
                kind = 'eth'
            elif iface.startswith(('tailscale', 'wg', 'tun', 'tap')):
                kind = 'vpn'
            else:
                kind = 'other'
            ssid = None
            if kind == 'wifi':
                try:
                    r = subprocess.run(['iwgetid', '-r', iface],
                                       capture_output=True, text=True,
                                       timeout=2)
                    ssid = r.stdout.strip() or None
                except Exception:
                    pass
            ip = None
            try:
                r = subprocess.run(
                    ['ip', '-4', '-br', 'addr', 'show', iface],
                    capture_output=True, text=True, timeout=2)
                parts = r.stdout.strip().split()
                if len(parts) >= 3 and '/' in parts[2]:
                    ip = parts[2].split('/')[0]
            except Exception:
                pass
            nics.append({'iface': iface, 'kind': kind,
                         'ssid': ssid, 'ip': ip})
    except Exception:
        pass
    return nics

def read_cpu_stat():
    try:
        with open('/proc/stat') as f:
            parts = f.readline().split()
        v = [int(x) for x in parts[1:]]
        return sum(v), v[3] + v[4]
    except Exception:
        return 0, 0

def read_per_core_stats():
    cores = []
    try:
        with open('/proc/stat') as f:
            for line in f:
                if line.startswith('cpu') and not line.startswith('cpu '):
                    v = [int(x) for x in line.split()[1:]]
                    cores.append((sum(v), v[3] + v[4]))
    except Exception:
        pass
    return cores

def read_net_stats():
    stats = {}
    try:
        with open('/proc/net/dev') as f:
            for line in f:
                if ':' not in line:
                    continue
                iface, data = line.split(':', 1)
                iface = iface.strip()
                if iface == 'lo':
                    continue
                p = data.split()
                stats[iface] = (int(p[0]), int(p[8]))
    except Exception:
        pass
    return stats

def read_disk_stats():
    stats = {}
    try:
        with open('/proc/diskstats') as f:
            for line in f:
                p = line.split()
                if len(p) < 14:
                    continue
                name = p[2]
                # whole-disk devices only, no partitions
                if name.startswith('nvme'):
                    tail = name.split('n', 1)[-1]
                    if 'p' in tail:
                        continue
                elif name.startswith('sd'):
                    if any(c.isdigit() for c in name[2:]):
                        continue
                elif name.startswith('mmcblk'):
                    if 'p' in name:
                        continue
                else:
                    continue
                stats[name] = (int(p[5]), int(p[9]))
    except Exception:
        pass
    return stats

def fmt_rate(bps):
    if bps >= 1048576:
        return f'{bps / 1048576:.1f} MB/s'
    if bps >= 1024:
        return f'{bps / 1024:.1f} KB/s'
    return f'{bps} B/s'

def get_rapl_info():
    try:
        cur = int(read_sys(f'{RAPL}/constraint_0_power_limit_uw') or '0')
        tdp = int(read_sys(f'{RAPL}/constraint_0_max_power_uw') or '0')
        if cur > 0 and tdp > 0 and read_sys(f'{RAPL}/name') == 'package-0':
            cur_w = cur // 1000000
            tdp_w = tdp // 1000000
            # max_power_uw is TDP; actual limit can be higher (firmware)
            slider_max = max(cur_w, tdp_w * 3, 100)
            slider_max = min(slider_max, 200)
            return min(cur_w, slider_max), slider_max
    except Exception:
        pass
    return None

def log_battery_history(bp):
    if not bp:
        return
    try:
        full = int(read_sys(f'{bp}/energy_full') or
                   read_sys(f'{bp}/charge_full') or '0')
        design = int(read_sys(f'{bp}/energy_full_design') or
                     read_sys(f'{bp}/charge_full_design') or '1')
        health = round(full / design * 100)
        cycles = read_sys(f'{bp}/cycle_count') or '0'
        today = time.strftime('%Y-%m-%d')
        os.makedirs(CONFIG_DIR, exist_ok=True)
        last = ''
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE) as f:
                for line in f:
                    if line.strip():
                        last = line.split(',')[0]
        if last != today:
            with open(HISTORY_FILE, 'a') as f:
                f.write(f'{today},{health},{cycles}\n')
    except Exception:
        pass

def battery_history_summary():
    try:
        if not os.path.exists(HISTORY_FILE):
            return None
        with open(HISTORY_FILE) as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return None
        first, last = lines[0].split(','), lines[-1].split(',')
        return dict(since=first[0], first_h=first[1],
                    cur_h=last[1], n=len(lines))
    except Exception:
        return None


# ── processes ──────────────────────────────────────────────────────────
# The Qt app grew its own /proc walker inside a widget class. Lifting it here
# means the desktop app and the web module enumerate processes identically —
# the same reason every sysfs path in this file was extracted in the first
# place.

_UID_CACHE: dict[int, str] = {}

PROC_STATES = {
    'R': 'running', 'S': 'sleeping', 'D': 'disk-wait', 'Z': 'zombie',
    'T': 'stopped', 't': 'traced', 'X': 'dead', 'I': 'idle',
}


def resolve_uid(uid: int) -> str:
    if uid not in _UID_CACHE:
        try:
            import pwd
            _UID_CACHE[uid] = pwd.getpwuid(uid).pw_name
        except (KeyError, ImportError):
            _UID_CACHE[uid] = str(uid)
    return _UID_CACHE[uid]


def mem_total_kb() -> int:
    try:
        with open('/proc/meminfo') as fh:
            for line in fh:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1])
    except OSError:
        pass
    return 1


def read_cpu_total_jiffies() -> int:
    try:
        with open('/proc/stat') as fh:
            return sum(int(v) for v in fh.readline().split()[1:])
    except (OSError, ValueError):
        return 0


def read_proc_table() -> dict:
    """Every readable process, with raw cumulative CPU jiffies.

    Returns raw counters rather than percentages: a percentage needs two
    samples, and only the caller knows how far apart its samples are.
    """
    out = {}
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f'/proc/{pid}/stat') as fh:
                stat = fh.read()
            name = stat[stat.index('(') + 1:stat.rindex(')')]
            fields = stat[stat.rindex(')') + 2:].split()

            rss_kb, uid = 0, 0
            with open(f'/proc/{pid}/status') as fh:
                for line in fh:
                    if line.startswith('VmRSS:'):
                        rss_kb = int(line.split()[1])
                    elif line.startswith('Uid:'):
                        uid = int(line.split()[1])

            try:
                with open(f'/proc/{pid}/cmdline') as fh:
                    cmdline = fh.read().replace('\0', ' ').strip()
            except OSError:
                cmdline = ''

            out[pid] = {
                'pid': pid,
                'name': name,
                'user': resolve_uid(uid),
                'uid': uid,
                'jiffies': int(fields[11]) + int(fields[12]),
                'memKb': rss_kb,
                'state': PROC_STATES.get(fields[0], fields[0]),
                'cmdline': cmdline or name,
            }
        except (FileNotFoundError, ProcessLookupError, PermissionError,
                ValueError, IndexError):
            continue          # the process exited while we walked /proc
    return out
