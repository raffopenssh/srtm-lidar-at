"""Host CPU / scheduling telemetry for peer profiling.

exe.dev VMs are stamped "2 vCPU / 8 GB" but they live on shared
hypervisor pools with different oversubscription ratios. The cheapest
signal for that is **CPU steal %** from /proc/stat — the fraction of
wall time the vCPU was runnable but the hypervisor preferred another
guest. Sustained steal >5% means the peer is sharing a hot pool; >15%
means it's effectively half-speed. iowait is the same idea for storage.

This module is import-light (stdlib only) so it can be used from
``app.py`` (peer status-push loop), ``austria_processor.py``
(per-KG metrics snapshot), and ``peer_director.py`` (fleet aggregate).

All state is kept in module-level dicts; each caller gets its own
"channel" so e.g. the push loop (30 s cadence) and the processor's
update_system_metrics (per-KG) don't reset each other's deltas.

Public API:
    cpu_snapshot(channel='default') -> dict | None
        Returns percentages over the delta since the previous call on
        the same channel. First call returns None (no delta yet).
        Keys: user, system, iowait, steal, idle, total_pct (1.0 = one
        full vCPU saturated), n_cpu, ts.

    perf_summary(channel='default', window=20) -> dict
        Rolling-window aggregate of recent cpu_snapshot() calls.
        Returns medians + EWMA of steal/iowait/cpu so a quick read
        from process.txt gives a stable view that ignores single-tick
        spikes. Empty dict if no samples yet.

    bw_throughput_mbps() -> float | None
        Wall-clock RX+TX throughput over the last cpu_snapshot()
        window; useful to correlate CPU steal vs. network during
        zenodo uploads.

The rolling-history deques are bounded (CPU_HISTORY_MAX=120). At a
30-s push cadence that's 1 h of trend per peer — enough to spot a
resource-pool change without leaking memory.
"""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Optional

# Per-channel state: {channel: (prev_total, prev_idle, prev_fields_dict, ts)}
_CPU_PREV: dict[str, tuple] = {}
# Per-channel rolling history of (ts, snapshot_dict)
CPU_HISTORY_MAX = 120
_CPU_HIST: dict[str, deque] = {}
# Network counters previous snapshot (rx_bytes, tx_bytes, ts) for throughput.
_NET_PREV: dict[str, tuple] = {}

_CPU_FIELDS = (
    "user", "nice", "system", "idle", "iowait",
    "irq", "softirq", "steal", "guest", "guest_nice",
)


def _read_proc_stat() -> Optional[dict]:
    try:
        with open("/proc/stat") as f:
            line = f.readline()
    except OSError:
        return None
    if not line.startswith("cpu "):
        return None
    parts = line.split()[1:]
    # Tolerate older kernels with fewer fields.
    vals = [int(x) for x in parts[: len(_CPU_FIELDS)]]
    while len(vals) < len(_CPU_FIELDS):
        vals.append(0)
    return dict(zip(_CPU_FIELDS, vals))


def _read_net_bytes() -> Optional[tuple[int, int]]:
    """Sum rx_bytes / tx_bytes across all non-loopback interfaces."""
    rx = tx = 0
    try:
        with open("/proc/net/dev") as f:
            next(f); next(f)  # headers
            for line in f:
                if ":" not in line:
                    continue
                name, rest = line.split(":", 1)
                name = name.strip()
                if name == "lo" or name.startswith(("docker", "veth", "br-")):
                    continue
                fields = rest.split()
                if len(fields) >= 9:
                    rx += int(fields[0])
                    tx += int(fields[8])
    except (OSError, ValueError, StopIteration):
        return None
    return rx, tx


def cpu_snapshot(channel: str = "default") -> Optional[dict]:
    """Sample /proc/stat and return percentages since previous call.

    Percentages are normalised so user + system + ... + idle == 100,
    matching the convention of vmstat / top (i.e. averaged across all
    vCPUs). ``total_pct`` is the non-idle fraction; on a 2-vCPU host
    a fully busy single thread shows ~50, a fully busy fork ~100.
    """
    cur = _read_proc_stat()
    if cur is None:
        return None
    cur_total = sum(cur.values()) - cur["guest"] - cur["guest_nice"]
    cur_idle = cur["idle"] + cur["iowait"]
    ts = time.time()
    prev = _CPU_PREV.get(channel)
    _CPU_PREV[channel] = (cur_total, cur_idle, cur, ts)
    if prev is None:
        return None
    pt, pi, pf, pts = prev
    dt = cur_total - pt
    if dt <= 0:
        return None
    n_cpu = max(os.cpu_count() or 1, 1)
    snap = {
        "ts": ts,
        "window_s": round(ts - pts, 2),
        "n_cpu": n_cpu,
        # Percentages of *total* CPU time (normalised across vCPUs).
        "user": round(100 * (cur["user"] + cur["nice"] - pf["user"] - pf["nice"]) / dt, 1),
        "system": round(100 * (cur["system"] + cur["irq"] + cur["softirq"]
                                - pf["system"] - pf["irq"] - pf["softirq"]) / dt, 1),
        "iowait": round(100 * (cur["iowait"] - pf["iowait"]) / dt, 1),
        "steal": round(100 * (cur["steal"] - pf["steal"]) / dt, 1),
        "idle": round(100 * (cur["idle"] - pf["idle"]) / dt, 1),
    }
    snap["total_pct"] = round(100 - snap["idle"] - snap["iowait"], 1)
    # Rolling history
    hist = _CPU_HIST.setdefault(channel, deque(maxlen=CPU_HISTORY_MAX))
    hist.append(snap)
    return snap


def perf_summary(channel: str = "default", window: int = 20) -> dict:
    """Aggregate the last ``window`` cpu_snapshot() samples.

    Returns medians + a simple EWMA (alpha=0.3) so chronic throttling
    is distinguishable from a single noisy tick. Empty dict if no
    samples have been taken yet on this channel.
    """
    hist = _CPU_HIST.get(channel)
    if not hist:
        return {}
    samples = list(hist)[-window:]
    if not samples:
        return {}

    def _med(key: str) -> float:
        vals = sorted(s[key] for s in samples)
        n = len(vals)
        if n == 0:
            return 0.0
        if n % 2:
            return vals[n // 2]
        return (vals[n // 2 - 1] + vals[n // 2]) / 2.0

    def _ewma(key: str, alpha: float = 0.3) -> float:
        v = samples[0][key]
        for s in samples[1:]:
            v = alpha * s[key] + (1 - alpha) * v
        return round(v, 1)

    return {
        "n": len(samples),
        "window_s": round(sum(s.get("window_s", 0) for s in samples), 1),
        "n_cpu": samples[-1]["n_cpu"],
        "cpu_user_med":   round(_med("user"), 1),
        "cpu_system_med": round(_med("system"), 1),
        "cpu_iowait_med": round(_med("iowait"), 1),
        "cpu_steal_med":  round(_med("steal"), 1),
        "cpu_total_med":  round(_med("total_pct"), 1),
        # EWMA — biased toward recent ticks; better for "is this peer
        # being throttled right now" than the flat median.
        "cpu_steal_ewma":  _ewma("steal"),
        "cpu_iowait_ewma": _ewma("iowait"),
        "cpu_total_ewma":  _ewma("total_pct"),
        # 95th-percentile steal so we can spot peers with bursty
        # noisy-neighbour episodes that don't show in median.
        "cpu_steal_p95": round(sorted(s["steal"] for s in samples)[
            min(len(samples) - 1, int(len(samples) * 0.95))], 1),
        "last_ts": samples[-1]["ts"],
    }


def bw_throughput_mbps(channel: str = "default") -> Optional[float]:
    """RX+TX MB/s since previous call on this channel. None on first call."""
    nb = _read_net_bytes()
    if nb is None:
        return None
    ts = time.time()
    prev = _NET_PREV.get(channel)
    _NET_PREV[channel] = (nb[0], nb[1], ts)
    if prev is None:
        return None
    prx, ptx, pts = prev
    dt = ts - pts
    if dt <= 0:
        return None
    return round(((nb[0] - prx) + (nb[1] - ptx)) / 1e6 / dt, 2)


def host_profile() -> dict:
    """One-shot host fingerprint for the peer-profile section.

    Pairs cpu_model + n_cpu + ram_total with the rolling perf summary
    so the director can group peers into resource-pool buckets without
    needing to maintain a static peers.json field. Cheap; called from
    the status-push loop every ~5 min via a callsite-side throttle.
    """
    info: dict = {}
    try:
        info["n_cpu"] = os.cpu_count() or 0
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
                if line.startswith("Hardware"):  # ARM fallback
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["ram_total_mb"] = int(line.split()[1]) // 1024
                    break
    except OSError:
        pass
    try:
        # Hypervisor hint (KVM / VMware / Xen / cloud-vendor) helps
        # identify which physical pool the peer landed on.
        if os.path.exists("/sys/class/dmi/id/sys_vendor"):
            with open("/sys/class/dmi/id/sys_vendor") as f:
                v = f.read().strip()
                if v:
                    info["sys_vendor"] = v
        if os.path.exists("/sys/class/dmi/id/product_name"):
            with open("/sys/class/dmi/id/product_name") as f:
                v = f.read().strip()
                if v:
                    info["product_name"] = v
    except OSError:
        pass
    try:
        info["boot_ts"] = int(time.time() - float(open("/proc/uptime").read().split()[0]))
    except (OSError, ValueError):
        pass
    return info
