"""
Simulate an inbound WhatsApp text message hitting our webhook — locally, with a
correct X-Hub-Signature-256 — so you can exercise the full pipeline
(webhook → Celery worker → guardrails → RAG agent → outbound reply) WITHOUT a
public tunnel.

The outbound reply is sent for real via the WhatsApp Cloud API, so `--to` must
be a number that has messaged your test number in the last 24h (open session).

Usage (inside the api container):
    docker compose exec api python scripts/simulate_whatsapp.py \
        --to 9230012345678 --text "Admission ki last date kya hai?"
"""

import argparse
import hashlib
import hmac
import json
import os
import urllib.request

API = os.getenv("SIM_API_URL", "http://127.0.0.1:8058")
APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--to", required=True, help="Sender wa_id (E.164, no '+'). The reply goes here."
    )
    ap.add_argument("--text", default="Admission ke liye kya requirements hain?")
    args = ap.parse_args()

    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "contacts": [{"wa_id": args.to}],
                            "messages": [
                                {
                                    "id": "wamid.SIMULATED",
                                    "from": args.to,
                                    "type": "text",
                                    "text": {"body": args.text},
                                }
                            ],
                        }
                    }
                ]
            }
        ],
    }
    body = json.dumps(payload).encode()
    sig = hmac.new(APP_SECRET.encode(), body, hashlib.sha256).hexdigest()

    req = urllib.request.Request(
        f"{API}/api/v1/whatsapp/webhook",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": f"sha256={sig}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        print("Webhook HTTP", resp.status, resp.read().decode())
    print(
        "→ Worker is now processing; the bot's reply will arrive on WhatsApp at",
        args.to,
    )


if __name__ == "__main__":
    main()
