# 🤖 AI-Powered CI/CD Auto-Fixer

**100% Automatic DevOps Failure Detection & Fixing** — No manual log dropping required.

The AI automatically:
- Polls multiple CI/CD platforms for failures
- Analyzes root causes
- Generates fixes
- Commits directly to git

## 🎯 How It Works (No Manual Logs!)

```
GitHub Actions ──┐
GitLab CI ────────┼──→ CI Platform Poller ──→ AI Analyzer ──→ Auto-Fixer ──→ Auto-Commit
Docker Registry ─┤                              (Ollama)
Kubernetes ──────┤
System Logs ─────┘
```

## 🚀 Supported Platforms

| Platform | Auto-Detection | Notes |
|----------|:--:|---------|
| **GitHub Actions** | ✅ | Polls API for failed workflows |
| **GitLab CI** | ✅ | Monitors pipeline failures |
| **Docker Registry** | ✅ | Detects build failures |
| **Kubernetes** | ✅ | Monitors pod errors |
| **System Logs** | ✅ | Scans syslog for errors |
| **Jenkins** | 🔄 | Coming soon |

## 📦 Available Tools

| Tool | Purpose |
|------|---------|
| **ci-platform-poller.py** | **[MAIN]** Polls all CI/CD platforms continuously, auto-fixes failures |
| **ci-local-deploy.yml** | GitHub Actions workflow to build and deploy a local app on a self-hosted runner |
| **auto-fixer.py** | Core: detects failure type, generates fix, commits |
| **analyzer.py** | Manual analysis mode (for testing) |
| **workflow-watcher.py** | Alternative: watches directory for manual log drops |
| **github-monitor.py** | Alternative: GitHub-only monitoring |

---

## 🏠 Local GitHub Actions Deployment Pipeline

A sample mini application is included in `sample_app/`. The GitHub Actions workflow `./github/workflows/ci-local-deploy.yml`:

- runs on a self-hosted Linux runner located on your VM
- installs Python dependencies
- runs unit tests
- builds a Docker image
- deploys the container locally
- verifies the app is reachable on `http://localhost:5000`

### When to use this

Use this pipeline when you want a **real GitHub Actions job** to deploy to your own VM, not just simulate log analysis.

### Requirements for local deployment

- A VM with a self-hosted GitHub Actions runner installed
- Docker installed on that VM
- The runner tagged for `self-hosted` and `linux`
- `sample_app/` code present in repository

### How it works

```text
GitHub Actions push → self-hosted runner on VM → build/test → docker deploy → local app available
```

---

## 🔧 Quick Start (3 Steps)

### Step 1: Install & Setup

```powershell
cd AI-BASED-debugger

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Start Ollama (Terminal 1)

```powershell
ollama pull llama3
ollama run llama3
# Keep this running!
```

### Step 3: Start Auto-Fixer Poller (Terminal 2)

#### GitHub Actions (Most Common)

```powershell
# Set your GitHub credentials
$env:GITHUB_TOKEN = "ghp_YOUR_TOKEN_HERE"

# Start polling (runs forever, checks every 5 minutes)
python ci-platform-poller.py `
    --github-owner YOUR_USERNAME `
    --github-repo YOUR_REPO
```

#### Multiple Platforms at Once

```powershell
python ci-platform-poller.py `
    --github-owner YOUR_USERNAME `
    --github-repo YOUR_REPO `
    --gitlab-token YOUR_GITLAB_TOKEN `
    --gitlab-project-id 12345 `
    --enable-kubernetes `
    --enable-syslog
```

#### Test Once (No Loop)

```powershell
python ci-platform-poller.py `
    --github-owner YOUR_USERNAME `
    --github-repo YOUR_REPO `
    --once
```

---

## 🧩 Self-Hosted Runner Setup for Local Deployment

To run `ci-local-deploy.yml` on your VM, install a GitHub Actions self-hosted runner there.

1. Go to your repository on GitHub.
2. Settings → Actions → Runners → Add runner.
3. Choose Linux, then copy the registration commands.
4. Run the commands on your VM.
5. Add the `self-hosted` label to the runner.

Make sure Docker is installed on the VM:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker $USER
```

Then push to `main` and the workflow will execute on your local VM.

---

## ⚡ What Happens Automatically

1. **Poller runs every 5 minutes** (configurable)
2. **Detects failed workflows** in GitHub Actions / GitLab / Docker / K8s
3. **Downloads logs** automatically
4. **Sends to AI** (local Ollama + Llama3)
5. **Generates fix** (JSON with file changes)
6. **Applies fix** (rewrites the broken file)
7. **Commits to git** automatically (`git add → git commit → git push`)

**Result:** You see commits appearing in your repo from the AI fixer!

---

## 📋 Setup by Platform

### GitHub Actions

```powershell
# 1. Create GitHub Personal Access Token
#    https://github.com/settings/tokens
#    Needs: repo + workflow scopes

# 2. Set environment variable
$env:GITHUB_TOKEN = "ghp_xxxxxxxxxxxx"

# 3. Start poller
python ci-platform-poller.py --github-owner YOU --github-repo YOUR_REPO
```

### GitLab CI

```powershell
# 1. Get GitLab personal token from: https://gitlab.com/-/profile/personal_access_tokens
# 2. Get project ID: https://gitlab.com/YOUR_GROUP/YOUR_PROJECT (see URL: /projects/12345)

$env:GITLAB_TOKEN = "glpat-xxxxxxxxxxxx"

python ci-platform-poller.py `
    --gitlab-token $env:GITLAB_TOKEN `
    --gitlab-project-id 12345
```

### Kubernetes

```powershell
# Prerequisites: kubectl configured and access to cluster

python ci-platform-poller.py --enable-kubernetes
```

### System Logs

```powershell
# Monitors /var/log/syslog for ERROR patterns

python ci-platform-poller.py --enable-syslog
```

---

## 🎯 Real Example

### Scenario: GitHub workflow fails

1. Your `.github/workflows/test.yml` job fails
2. Poller detects it (every 5 minutes)
3. Downloads the failure log
4. AI analyzes: "Missing dependency version in requirements.txt"
5. AI generates fix: "Change requests==999.0.0 → requests==2.31.0"
6. Fix is applied automatically
7. Commit appears in your repo: `"Fix: Correct package version mismatch in requirements.txt"`

**You don't do anything!** Just keep the poller running.

---

## 🔐 Security & Configuration

### Environment Variables

```powershell
# GitHub
$env:GITHUB_TOKEN = "ghp_xxxxxxxxxxxx"

# GitLab
$env:GITLAB_TOKEN = "glpat_xxxxxxxxxxxx"

# Docker
$env:DOCKER_USERNAME = "your_username"
$env:DOCKER_TOKEN = "your_api_token"

# Ollama
$env:OLLAMA_API_URL = "http://127.0.0.1:11434/v1/completions"
$env:OLLAMA_MODEL = "llama3"
```

### Configuration File (Alternative)

Create `config.json`:

```json
{
  "github_token": "ghp_xxxxxxxxxxxx",
  "github_owner": "your-org",
  "github_repo": "your-repo",
  "gitlab_token": "glpat_xxxxxxxxxxxx",
  "gitlab_project_id": "12345",
  "poll_interval": 300,
  "enable_github": true,
  "enable_gitlab": true,
  "enable_docker": false,
  "enable_kubernetes": false,
  "enable_syslog": false
}
```

Then run:

```powershell
python ci-platform-poller.py --config config.json
```

---

## 📊 Monitoring & Verification

### Check if auto-fixes are being applied

```powershell
# Watch git commits in real-time
git log --follow --oneline | head -20

# Should see commits like:
# a1b2c3d Fix: Correct package version mismatch in requirements.txt
# e4f5g6h Fix: Update Dockerfile base image to python:3.11
```

### View poller output

```powershell
# Terminal shows live updates
[2026-05-30 14:30:00] Polling all platforms...
[DETECTED] github failure: build-and-test
[SUCCESS] Fixed github failure
[2026-05-30 14:35:00] Polling all platforms...
[2026-05-30 14:35:00] No failures detected
```

---

## 🛠️ Customization

### Change Poll Interval

```powershell
# Check every 60 seconds instead of 300
python ci-platform-poller.py `
    --github-owner YOU `
    --github-repo REPO `
    --poll-interval 60
```

### Use Different AI Model

```powershell
# Use Mistral instead of Llama3 (faster)
$env:OLLAMA_MODEL = "mistral"
python ci-platform-poller.py ...
```

### Disable Git Commits (Dry-Run)

Edit `auto-fixer.py` and change line that calls `commit_fix()` to test.

---

## 🚨 Important Notes

1. **AI commits directly to main** — Review before enabling in production
2. **Requires Ollama running** — Keep Terminal 1 active
3. **Uses your credentials** — Store tokens securely in environment variables
4. **Works locally only** — No external API calls to third parties

---

## 🧪 Demonstration Guide

This section is for live CI/CD demo and verification while showing the system in action.

### 1. Prepare the environment

```powershell
cd c:\Users\mazhar.mehmood\Downloads\AI-BASED-debugger
.\.venv\Scripts\Activate.ps1
```

### 2. Start Ollama in Terminal 1

```powershell
ollama pull llama3
ollama run llama3
```

### 3. Start the CI/CD poller in Terminal 2

```powershell
$env:GITHUB_TOKEN = "ghp_YOUR_TOKEN_HERE"
python ci-platform-poller.py --github-owner YOUR_NAME --github-repo YOUR_REPO
```

### 4. Trigger a failing workflow

For demonstration, push a commit that causes a GitHub Actions failure, for example:

- invalid package version in `requirements.txt`
- broken Dockerfile
- failing test command

Your workflow should fail in GitHub Actions and become visible in the Actions tab.

### 5. Watch the poller output

The poller terminal should display messages like:

```text
[2026-05-30 15:00:00] Polling all platforms...
[DETECTED] github failure: build-and-test
[SUCCESS] Fixed github failure
```

This proves the system detected the real pipeline failure automatically.

### 6. Verify git commit

In Terminal 3, run:

```powershell
git log --oneline -3
```

You should see a new commit created by the AI fixer, for example:

```text
a1b2c3d Fix: Correct package version mismatch in requirements.txt
```

### 7. Review the changed file

Use:

```powershell
git show HEAD
```

Confirm the AI fix is applied to the correct file.

---

## ✅ Verification Checklist for Demonstration

1. **Ollama is running** in Terminal 1
2. **Poller is running** in Terminal 2
3. **Workflow fails** in GitHub Actions or GitLab
4. Poller output shows **detected failure** and **fix success**
5. `git log` shows an **auto-generated commit**
6. `git show HEAD` shows the **actual fixed file contents**

---

## 📚 Learn More

- [ARCHITECTURE.md](ARCHITECTURE.md) — Deep dive into how the system works
- [EXAMPLES.md](EXAMPLES.md) — Copy-paste examples for each platform

---

## 🎓 Next Steps

1. **Create GitHub Personal Access Token:** https://github.com/settings/tokens
2. **Set GITHUB_TOKEN environment variable**
3. **Run `python ci-platform-poller.py --github-owner YOU --github-repo REPO`**
4. **Watch for auto-fixes in your git commits!**


