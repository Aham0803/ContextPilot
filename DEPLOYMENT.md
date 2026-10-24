# 🚀 ContextPilot Deployment Guide

This guide provides step-by-step instructions for deploying **ContextPilot** across different environments, from 100% free cloud hosting to containerized production servers.

---

## 📌 Deployment Options Overview

| Deployment Method | Cost | Best For | Prerequisites |
| :--- | :---: | :--- | :--- |
| **Option 1: Streamlit Community Cloud** | **Free** | Public portfolio, demo links, resumes | GitHub account |
| **Option 2: Docker Compose (VPS)** | ~$5–$15/mo | Production, fully private self-hosted | Any VPS (DigitalOcean, AWS, GCP, Hetzner) |
| **Option 3: Hugging Face Spaces** | **Free** | Open-source community showcase | Hugging Face account |
| **Option 4: Cloudflare Tunnel / Ngrok** | **Free** | Share your local running instance instantly | Local machine |

---

## Option 1: Streamlit Community Cloud (Free & 1-Click from GitHub)

Streamlit Community Cloud is the fastest and easiest way to host ContextPilot for free.

### Step 1: Sign Up / Log In
1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Click **"Continue with GitHub"** and authorize Streamlit.

### Step 2: Deploy App
1. Click **"Create app"** (or **"New app"**).
2. Select your repository: `Aham0803/ContextPilot`.
3. Set the configuration:
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom domain (e.g., `contextpilot.streamlit.app`).

### Step 3: Configure Ollama / LLM Endpoint in Secrets
Because Streamlit Community Cloud runs on a serverless container, it needs to know where your LLM lives.
In the Streamlit app settings under **"Secrets"**, add:
```toml
OLLAMA_BASE_URL = "https://your-tunnel-url-or-cloud-ollama.com"
LLM_MODEL = "mistral:latest"
```
*(If you want to use your local Ollama instance with Streamlit Cloud, see [Option 4: Expose Local Ollama via Cloudflare Tunnel](#option-4-expose-local-ollama-via-cloudflare-tunnel-or-ngrok)).*

4. Click **Deploy!** Your app will build and go live in ~2 minutes.

---

## Option 2: 1-Command Docker Deployment (VPS: DigitalOcean / AWS / GCP)

This option hosts both **Streamlit** and **Ollama** together on the same server, with 100% data privacy and zero external API dependencies.

### Step 1: Provision a Linux VPS
- Recommended specs: 4 vCPUs, 8 GB+ RAM (e.g. DigitalOcean Droplet, AWS EC2 `t3.xlarge`, Hetzner CX31).
- OS: Ubuntu 22.04 LTS or 24.04 LTS.

### Step 2: Install Docker & Docker Compose
```bash
sudo apt update && sudo apt install -y docker.io docker-compose
sudo systemctl enable --now docker
```

### Step 3: Clone the Repository
```bash
git clone https://github.com/Aham0803/ContextPilot.git
cd ContextPilot
```

### Step 4: Launch ContextPilot & Ollama
```bash
docker-compose up -d --build
```

### Step 5: Pull Mistral into the Ollama Container
```bash
docker exec -it contextpilot-ollama ollama pull mistral
```

### Step 6: Access Your App
Open your browser to:
```
http://<YOUR_SERVER_IP>:8501
```

---

## Option 3: Deploy to Hugging Face Spaces (Free)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Name your space: `context-pilot`.
3. Select **Streamlit** as the Space SDK.
4. Set hardware to the free **CPU basic (2 vCPU, 16 GB RAM)**.
5. In your local terminal, link the Hugging Face git remote and push:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-hf-username>/context-pilot
   git push hf main
   ```

---

## Option 4: Expose Local Ollama via Cloudflare Tunnel or Ngrok

If you want to host the frontend on Streamlit Cloud while keeping Mistral running on your local GPU/CPU:

### Using Cloudflare Quick Tunnel (Free, no account needed):
1. Download Cloudflare Tunnel (`cloudflared`):
   ```powershell
   winget install Cloudflare.cloudflared
   ```
2. Start tunnel forwarding your local Ollama port (11434):
   ```powershell
   cloudflared tunnel --url http://localhost:11434
   ```
3. Cloudflare will output a public HTTPS URL (e.g. `https://random-subdomain.trycloudflare.com`).
4. Set this URL as your `OLLAMA_BASE_URL` in Streamlit Cloud Secrets or `.env`:
   ```env
   OLLAMA_BASE_URL=https://random-subdomain.trycloudflare.com
   ```

---

## ⚙️ Environment Variables Reference

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base HTTP URL for the Ollama inference server |
| `LLM_MODEL` | `mistral:latest` | Target LLM model name loaded in Ollama |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers embedding model |
| `DEFAULT_CHUNK_SIZE` | `400` | Default character chunk size for splitting |
| `DEFAULT_CHUNK_OVERLAP` | `50` | Default chunk overlap size |
| `STREAMLIT_SERVER_PORT` | `8501` | Port on which the web server listens |
