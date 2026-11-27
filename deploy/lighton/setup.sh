#!/usr/bin/env bash
# Deploy text-extract-be + LightOnOCR on a fresh GPU EC2 instance (Ubuntu 24.04 Deep Learning AMI).
# Run on the instance:
#   git clone https://github.com/vrajmakwana-create/text-extract-be.git ~/workspace/text-extract-be
#   bash ~/workspace/text-extract-be/deploy/lighton/setup.sh
# Re-running is safe: every step skips or overwrites to the same end state.
set -euo pipefail

KIT="$(cd "$(dirname "$0")" && pwd)"
WS=/home/ubuntu/workspace
BE="$WS/text-extract-be"
REPO="${REPO:-https://github.com/vrajmakwana-create/text-extract-be.git}"
CORS_ORIGINS="${CORS_ORIGINS:-https://text-extraction-worker.netlify.app,http://localhost:5173}"

# Public IP -> sslip.io hostname, so Caddy can get a Let's Encrypt certificate with no DNS setup.
TOKEN=$(curl -s -m 5 -X PUT -H "X-aws-ec2-metadata-token-ttl-seconds: 60" http://169.254.169.254/latest/api/token)
PUBLIC_IP=$(curl -s -m 5 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4)
HOST="${HOST:-ocr.${PUBLIC_IP//./-}.sslip.io}"

echo "== 1/6 GPU check"
if ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi fails. On a new g6f instance the AMI swaps in its GRID driver on first boot;"
  echo "reboot once (sudo reboot) and run this script again. See README, 'GPU driver'."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "== 2/6 uv"
[ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
UV="$HOME/.local/bin/uv"

echo "== 3/6 backend code (${BRANCH:-main})"
mkdir -p "$WS"
if [ "$KIT" != "$BE/deploy/lighton" ]; then
  # Run from a standalone copy of this folder: clone the repo, then use its copy.
  [ -d "$BE/.git" ] || git clone -q -b "${BRANCH:-main}" "$REPO" "$BE"
  git -C "$BE" pull -q --ff-only || echo "git pull failed; using the checked-out code"
fi
cp "$BE/deploy/lighton/lighton-serve.sh" "$WS/lighton-serve.sh"
chmod +x "$WS/lighton-serve.sh"

echo "== 4/6 Python environments (vLLM takes a few minutes)"
[ -x "$WS/.venv-vllm/bin/vllm" ] || { "$UV" venv -q -p 3.12 "$WS/.venv-vllm" && "$UV" pip install -q -p "$WS/.venv-vllm" vllm --torch-backend=auto; }
[ -d "$BE/.venv" ] || "$UV" venv -q -p 3.12 "$BE/.venv"
# torch/transformers are only for the TrOCR engine; LightOnOCR runs inside vLLM.
grep -vE "^(torch|transformers|sentencepiece)" "$BE/requirements.txt" > /tmp/req-lighton.txt
"$UV" pip install -q -p "$BE/.venv" -r /tmp/req-lighton.txt

echo "== 5/6 systemd services"
sudo tee /etc/systemd/system/lighton-vllm.service >/dev/null <<EOF
[Unit]
Description=LightOnOCR-3-0.8B on vLLM (127.0.0.1:8000)
After=network-online.target nvidia-kmod-load.service nvidia-persistenced.service
Wants=network-online.target

[Service]
User=ubuntu
WorkingDirectory=${WS}
ExecStart=${WS}/lighton-serve.sh
Restart=always
RestartSec=10
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/text-extract.service >/dev/null <<EOF
[Unit]
Description=text-extract-be (FastAPI, 127.0.0.1:8002) using LightOnOCR
After=network-online.target lighton-vllm.service
Wants=lighton-vllm.service

[Service]
User=ubuntu
WorkingDirectory=${BE}
Environment=OCR_ENGINE=lighton
Environment=OCR_DEVICE=cpu
Environment=OCR_WORKERS=8
Environment=LIGHTON_PARALLEL=8
Environment=CORS_ORIGINS=${CORS_ORIGINS}
ExecStart=${BE}/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8002
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "== 6/6 Caddy (HTTPS on ${HOST})"
if ! command -v caddy >/dev/null; then
  sudo apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl >/dev/null
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y -qq caddy >/dev/null
fi
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
${HOST} {
	request_body {
		max_size 26MB
	}
	reverse_proxy 127.0.0.1:8002 {
		transport http {
			read_timeout 600s
		}
	}
}
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now lighton-vllm.service text-extract.service >/dev/null 2>&1
sudo systemctl restart lighton-vllm.service text-extract.service
sudo systemctl enable caddy >/dev/null 2>&1
sudo systemctl restart caddy

echo "Waiting for LightOnOCR to load (first run downloads the model)..."
for _ in $(seq 1 60); do
  curl -s -m 3 127.0.0.1:8000/v1/models >/dev/null && break
  sleep 10
done
curl -s -m 3 127.0.0.1:8000/v1/models >/dev/null && echo "vLLM: up" || echo "vLLM: not up yet, check: journalctl -u lighton-vllm -n 50"
echo
echo "Backend URL: https://${HOST}"
echo "Needs inbound TCP 80 and 443 open in the security group for the certificate."
