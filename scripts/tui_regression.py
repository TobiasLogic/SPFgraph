#!/usr/bin/env python3
"""Dashboard regression test: drives the blessed TUI in a pty.

Covers the historical failure modes:
  - Esc mid-generation, then a second prompt (blessed focus-stack corruption
    used to leave the input dead after exactly this sequence)
  - Ctrl+E (used to spawn $EDITOR over the dashboard)
  - /clear, then a third generation
  - Ctrl+C must exit with code 0 in under a few seconds

Usage:
    uv run --with pyte python3 scripts/tui_regression.py <model-path> [extra CLI args...]
    # or: pip install pyte && python3 scripts/tui_regression.py <model>

Use a small/fast model; ZEROSHOT_TEST_GGUF works as a default:
    uv run --with pyte python3 scripts/tui_regression.py "$ZEROSHOT_TEST_GGUF"

Exits 0 on pass, 1 on failure.
"""
import fcntl
import os
import pty
import select
import struct
import subprocess
import sys
import termios
import time

import pyte

COLS, ROWS = 100, 30
LOAD_TIMEOUT_S = 180
GEN_TIMEOUT_S = 60

model = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ZEROSHOT_TEST_GGUF", "")
if not model:
    print("usage: tui_regression.py <model-path> [extra CLI args...]", file=sys.stderr)
    sys.exit(2)

bin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin", "zeroshot-run.js")
cmd = ["node", bin_path, "load", model, "--max-tokens", "400", *sys.argv[2:]]

screen = pyte.Screen(COLS, ROWS)
stream = pyte.ByteStream(screen)
master, slave = pty.openpty()
fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
env = dict(os.environ, TERM="xterm-256color")
proc = subprocess.Popen(cmd, stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
os.close(slave)

failures = []

def pump(seconds):
    end = time.time() + seconds
    while time.time() < end:
        r, _, _ = select.select([master], [], [], 0.1)
        if master in r:
            try:
                data = os.read(master, 65536)
            except OSError:
                return False
            if not data:
                return False
            stream.feed(data)
    return True

def screen_text():
    return "\n".join(line.rstrip() for line in screen.display)

def wait_for(needle, timeout, count=1):
    end = time.time() + timeout
    while time.time() < end:
        pump(1)
        if screen_text().count(needle) >= count:
            return True
    return False

def done_lines():
    return {line.strip() for line in screen.display if "[done" in line}

def wait_new_done(before, timeout):
    end = time.time() + timeout
    while time.time() < end:
        pump(1)
        if done_lines() - before:
            return True
    return False

def check(ok, label):
    print(("PASS" if ok else "FAIL") + f"  {label}")
    if not ok:
        failures.append(label)

def send(data):
    os.write(master, data.encode() if isinstance(data, str) else data)

check(wait_for("chat", LOAD_TIMEOUT_S), "dashboard loads")

send("Once upon a time there was a\r")
pump(0.6)
send("\x1b")
check(wait_for("[done", GEN_TIMEOUT_S, count=1), "Esc mid-generation still reaches done")

send("\x05")
pump(1)
check("GNU nano" not in screen_text() and "chat" in screen_text(), "Ctrl+E does not open an editor")

before = done_lines()
send("The little dog ran\r")
check(wait_new_done(before, GEN_TIMEOUT_S), "second generation after abort completes")

send("/clear\r")
pump(1)
check("[done" not in screen_text(), "/clear empties the chat")

send("The sun was\r")
check(wait_for("[done", GEN_TIMEOUT_S, count=1), "generation after /clear completes")

send("\x03")
try:
    proc.wait(timeout=10)
    check(proc.returncode == 0, f"Ctrl+C exits 0 (got {proc.returncode})")
except subprocess.TimeoutExpired:
    proc.kill()
    check(False, "Ctrl+C exits within 10s")

print()
if failures:
    print(f"{len(failures)} FAILURE(S): " + "; ".join(failures))
    sys.exit(1)
print("ALL PASS")
sys.exit(0)
