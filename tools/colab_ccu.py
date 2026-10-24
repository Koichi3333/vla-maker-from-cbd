#!/usr/bin/env python3
"""Show Colab's CU balance, consumption rate and the GPUs that can be allocated (reuses the colab CLI's credentials).

    ~/.local/share/uv/tools/google-colab-cli/bin/python tools/colab_ccu.py

Use it when `colab new --gpu A100` fails with "Backend rejected accelerator", to find out
**which plan the CLI's account is on**.
On the free tier you get currentBalance 0 / eligibleGpus [T4] (measured 2026-09-12).
While a session is running, consumptionRateHourly shows the actual rate (5.3 for an A100).

The endpoint is /tun/m/ccu-info, the same one Colab's VS Code extension uses.
"""
import json

from colab_cli.auth import get_credentials
from colab_cli.client import XSSI_PREFIX, Prod

r = get_credentials().get(
    Prod().domain + "/tun/m/ccu-info",
    params={"authuser": "0"},
    headers={"Accept": "application/json"},
)
r.raise_for_status()
body = r.text[len(XSSI_PREFIX):] if r.text.startswith(XSSI_PREFIX) else r.text
info = json.loads(body)
print(f"currentBalance        {info.get('currentBalance', 0):.2f} CU")
print(f"consumptionRateHourly {info.get('consumptionRateHourly', 0)} CU/h")
print(f"assignmentsCount      {info.get('assignmentsCount', 0)}")
print(f"eligibleGpus          {info.get('eligibleGpus', [])}")
print(f"ineligibleGpus        {info.get('ineligibleGpus', [])}")
