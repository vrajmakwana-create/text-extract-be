# LightOnOCR deployment kit for text-extract-be

This kit deploys the `text-extract-be` backend with LightOnOCR-3-0.8B on a GPU EC2 instance and gives it a public HTTPS URL for the frontend. It reproduces the setup tested on 27 November 2026 on a g6f.2xlarge instance.

## Instructions for the agent

If a user gives you this README:

1. Ask for the instance's public IP and SSH key if they haven't given them. The SSH user is `ubuntu`.
2. Follow [Deploy](#deploy). If `setup.sh` stops at the GPU check, follow [GPU driver](#gpu-driver).
3. Open inbound TCP 80 and 443 on the instance's security group, with the AWS CLI if it has credentials, or ask the user to do it. Ask the user before you change the security group.
4. Run the checks in [Verify](#verify) through the public URL, then give the user the URL.

## What you get

```
frontend ──HTTPS──▶ Caddy :443 (https://ocr.<ip-with-dashes>.sslip.io)
                     └─▶ text-extract-be :8002  (systemd: text-extract)
                          └─▶ vLLM :8000, LightOnOCR-3-0.8B  (systemd: lighton-vllm)
```

- **Endpoint:** the API is unchanged: `POST /extract` with a multipart `file` field. It accepts PDF, DOCX and images.
- **CORS:** allows `https://text-extraction-worker.netlify.app` and `http://localhost:5173`. Set `CORS_ORIGINS` before you run `setup.sh` to change it.
- **Restarts:** both services restart on failure and start again after a reboot.

## Instance

| Choice | Instance | GPU | On-demand (us-east-1) |
|---|---|---|---|
| Tested | g6f.2xlarge | 1/4 L4, 6 GB | about $0.48/hr |
| Faster per request (untested) | g6.xlarge | full L4, 24 GB | about $0.80/hr |

- **AMI:** AWS Deep Learning AMI, Ubuntu 24.04. It includes the NVIDIA driver, Docker and the GRID driver that g6f needs.
- **Disk:** 100 GB root volume. About 25 GB is used after setup.
- **Security group:** inbound 22, 80 and 443.

Don't use g4dn (T4). LightOnOCR ran at about 2 tokens/s there, because the T4 has no bfloat16 support and current vLLM kernels don't run on it.

## Deploy

From your machine:

```bash
IP=1.2.3.4            # the new instance's public IP
KEY=~/path/to/key.pem
ssh -i $KEY ubuntu@$IP 'git clone -q https://github.com/vrajmakwana-create/text-extract-be.git ~/workspace/text-extract-be && bash ~/workspace/text-extract-be/deploy/lighton/setup.sh'
```

Set `BRANCH=<name>` before `setup.sh` to deploy a branch other than `main`.

`setup.sh` takes about 10 minutes and does the following:

1. Checks that `nvidia-smi` works.
2. Installs `uv`.
3. Clones `text-extract-be` if it isn't there yet and pulls the latest code.
4. Creates the vLLM environment and the backend environment.
5. Writes the systemd units.
6. Installs Caddy and prints the URL.

You can run it again safely.

## GPU driver

On a new g6f instance, `nvidia-smi` can fail on the first boot with "couldn't communicate with the NVIDIA driver". The AMI's boot script (`/opt/aws/dlami/bin/nvidia-kmod-load.sh`) swaps the standard driver for its GRID driver, and the swap only takes effect after a reboot.

- **Fix:** run `sudo reboot`, wait 2 minutes, and check that `nvidia-smi` shows `NVIDIA L4-6Q` and that `nvidia-smi -q | grep "License Status"` shows `Licensed`.
- **Don't install NVIDIA drivers by hand.** GRID 620.x is rejected by the host (Xid 78, "Guest driver is incompatible"). A runfile install also conflicts with the AMI's boot script, and uninstalling it deletes the `nvidia-gridd`, `nvidia-topologyd` and `nvidia-persistenced` service units. If that happens, extract `NVIDIA-Linux-x86_64-580.159.03-grid-aws.run` from `s3://ec2-linux-nvidia-drivers/grid-19.5/` with `-x` and copy the three units from `init-scripts/systemd/` to `/usr/lib/systemd/system/`.

## Verify

```bash
URL=https://ocr.<ip-with-dashes>.sslip.io
curl -s -o /dev/null -w "%{http_code}\n" $URL/docs                      # 200
curl -s -D - -o /dev/null -X OPTIONS $URL/extract \
  -H "Origin: https://text-extraction-worker.netlify.app" \
  -H "Access-Control-Request-Method: POST" | grep -i access-control      # allow-origin header
curl -s -w "\n%{time_total}s\n" -F file=@page.png $URL/extract | tail -c 300
```

The first OCR request after a restart takes 15 to 30 s while vLLM warms up.

Expected times on g6f.2xlarge, measured through the public URL:

| Upload | Time |
|---|---|
| Handwritten line | about 2.6 s |
| Digital PDF, 2 pages (text layer, no OCR) | about 3.5 s |
| Clean scanned page | 10 to 11 s |
| Scanned page with boxes drawn over the text | about 27 s per page |

## How the LightOnOCR engine works

- `app/ocr.py` adds `OCR_ENGINE=lighton`. It sends each region to vLLM in LightOnOCR's transcription mode, which is an empty prompt.
- **Why it splits pages:** one request generates only about 30 tokens/s on the g6f slice, so a full page in one request takes about 27 s. vLLM handles parallel requests well: 8 parallel requests gave 178 tokens/s in total.
- **How it splits pages:** `_lighton_region` splits a page into columns at the gutter, cuts each column at blank rows about 320 px apart, reads the strips in parallel and joins them in order. Clean pages come back with every word in the right order in about 7 s.
- `app/extract.py` adds `OCR_WORKERS`, which reads all of a page's layout blocks in parallel. It first does a dry run to collect the blocks, so the output is the same as reading them one at a time.

## Settings

| Variable | Where | Default | Meaning |
|---|---|---|---|
| `OCR_ENGINE` | text-extract.service | `lighton` | `lighton`, `trocr`, `ocria` or `rapidocr` |
| `OCR_WORKERS` | text-extract.service | `8` | Layout blocks read in parallel per page |
| `LIGHTON_PARALLEL` | text-extract.service | `8` | Strips read in parallel per region |
| `LIGHTON_STRIP_PX` | text-extract.service | `320` | Target strip height in pixels |
| `LIGHTON_URL` | text-extract.service | `http://127.0.0.1:8000/v1/chat/completions` | vLLM endpoint |
| `VLLM_GPU_MEM` | lighton-serve.sh | `0.75` | vLLM GPU memory share; 0.85 fails on the 6 GB slice |
| `VLLM_MAX_SEQS` | lighton-serve.sh | `8` | Concurrent sequences in vLLM |

On a full L4 (g6.xlarge), you can set `VLLM_GPU_MEM=0.85` and `VLLM_MAX_SEQS=16`.

## Operations

```bash
systemctl status lighton-vllm text-extract caddy
journalctl -u lighton-vllm -n 50         # vLLM startup errors (memory, CUDA graphs)
journalctl -u text-extract -n 50
sudo systemctl restart lighton-vllm text-extract
```

## Surya OCR 2.5 (tested, not deployed)

`datalab-to/surya-ocr-2.5` was benchmarked on the same g6f.2xlarge with vLLM 0.34.0:

| | Surya 2.5 | LightOnOCR with strips |
|---|---|---|
| Clean page | 31 s | about 7 s |
| 4 pages at once | 13.8 s per page | |
| Clean page accuracy | all 867 words, right order | all 867 words, right order |
| Model license | modified OpenRAIL-M: free only for research, personal use, or startups under $5M funding or revenue | Apache 2.0 |

LightOnOCR was kept. To test Surya again, install `surya-ocr` in its own environment, because it pins opencv versions that conflict with vLLM's. Serve the model with the existing vLLM install:

```bash
vllm serve datalab-to/surya-ocr-2.5 --served-model-name datalab-to/surya-ocr-2.5 \
  --port 8010 --max-model-len 16384 --max-num-seqs 8 --gpu-memory-utilization 0.75 \
  --enable-prefix-caching --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}'
```

Then point the client at it with `SURYA_INFERENCE_BACKEND=vllm SURYA_INFERENCE_URL=http://127.0.0.1:8010/v1`. The 6 GB slice can't hold LightOnOCR and Surya at the same time, so stop `lighton-vllm` first.

## Versions tested

- vLLM 0.34.0, torch 2.14.0+cu132, transformers 5.22.2
- surya-ocr 0.24.0
- NVIDIA GRID driver 580.159.03, from the AMI
- Kernel 7.0.0-1018-aws, Ubuntu 24.04.5
