#!/usr/bin/env python3
"""Run a command on a pty: output goes to LOG, whatever you write to INFILE goes to its stdin.

`colab drivemount` and `colab auth` print an authorization URL and then read
Enter from /dev/tty, so without a terminal (an agent, a script, CI) they hang.
Wrapping them in a pty lets them finish:

    python3 colab-cli/tools/ptyrun.py /tmp/dm.log /tmp/dm.in \\
        colab drivemount -s vla /content/drive &
    # the URL appears in /tmp/dm.log -> approve in a browser -> send Enter:
    printf '\\n' > /tmp/dm.in

colab/start.sh and go.sh do exactly this for the Drive mount (colab/common.sh).

INFILE is read and deleted every time it appears, so it can be used repeatedly.
"""
import fcntl
import os
import pty
import select
import struct
import sys
import termios

if len(sys.argv) < 4:
    sys.exit(__doc__)

LOG, INFILE = sys.argv[1], sys.argv[2]
cmd = sys.argv[3:]
log = open(LOG, "wb", 0)

pid, fd = pty.fork()
if pid == 0:
    # Wide enough that the long authorization URL is not wrapped.
    os.environ["COLUMNS"] = "400"
    os.execvp(cmd[0], cmd)

fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 400, 0, 0))

while True:
    r, _, _ = select.select([fd], [], [], 0.5)
    if fd in r:
        try:
            data = os.read(fd, 65536)
        except OSError:  # the pty closes when the child exits
            break
        if not data:
            break
        log.write(data)
    if os.path.exists(INFILE):
        with open(INFILE, "rb") as fh:
            payload = fh.read()
        os.remove(INFILE)
        if payload:
            os.write(fd, payload)

_, status = os.waitpid(pid, 0)
log.write(f"\n[ptyrun] exit={os.waitstatus_to_exitcode(status)}\n".encode())
