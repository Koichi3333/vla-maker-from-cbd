"""Run a command on a pty, writing its output to LOG and feeding whatever is written to INFILE into stdin.

A wrapper that lets non-interactive environments (agents such as Claude Code) complete
`colab drivemount` / `colab auth`. These print an authorization URL and then read Enter
from /dev/tty, so without a pty they cannot proceed.

    python3 tools/ptyrun.py /tmp/dm.log /tmp/dm.in \
        colab drivemount -s NAME /content/drive &
    # the authorization URL appears in /tmp/dm.log -> consent in a browser -> send Enter:
    printf '\n' > /tmp/dm.in

INFILE is read and deleted each time it is written, so it works for several inputs.
"""
import fcntl
import os
import pty
import select
import struct
import sys
import termios

LOG, INFILE = sys.argv[1], sys.argv[2]
cmd = sys.argv[3:]
log = open(LOG, "wb", 0)

pid, fd = pty.fork()
if pid == 0:
    # Use a wide terminal so long authorization URLs are not wrapped and become unreadable.
    os.environ["COLUMNS"] = "400"
    os.execvp(cmd[0], cmd)

fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 400, 0, 0))

while True:
    r, _, _ = select.select([fd], [], [], 0.5)
    if fd in r:
        try:
            data = os.read(fd, 65536)
        except OSError:  # The pty closes when the child process exits
            break
        if not data:
            break
        log.write(data)
    if os.path.exists(INFILE):
        payload = open(INFILE, "rb").read()
        os.remove(INFILE)
        if payload:
            os.write(fd, payload)

_, status = os.waitpid(pid, 0)
log.write(f"\n[ptyrun] exit={os.waitstatus_to_exitcode(status)}\n".encode())
