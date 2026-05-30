# 🎯 100% Automatic Setup (NO Manual Logs)

> **This is the new way** — All failures detected automatically from real CI/CD platforms.

## The Difference

### ❌ Old Way (Manual)
```
You drop a log → Watcher detects → Auto-fix
```
Still requires manual log creation.

### ✅ New Way (Fully Automatic)
```
GitHub fails → Poller detects → AI analyzes → Auto-fix
(No manual work at all!)
```

---

## 5-Minute Setup

### Terminal 1: Start Ollama

```powershell
ollama run llama3
```

Keep this running forever.

---

### Terminal 2: Setup & Run Poller

```powershell
cd c:\Users\mazhar.mehmood\Downloads\AI-BASED-debugger
.\.venv\Scripts\Activate.ps1

# Option A: GitHub Actions Only (Most Common)
$env:GITHUB_TOKEN = "ghp_YOUR_TOKEN"
python ci-platform-poller.py --github-owner YOUR_NAME --github-repo YOUR_REPO
```

**That's it!** The poller is now running.

---

## What Happens Next

### Every 5 Minutes:

1. ✅ Checks GitHub for failed workflows
2. ✅ Downloads failure logs automatically
3. ✅ Sends to local AI (Ollama + Llama3)
4. ✅ Gets back: root cause + fix
5. ✅ Writes fixed file
6. ✅ **Auto-commits to git**

### You See:

```
[2026-05-30 14:30:00] Polling all platforms...
[DETECTED] github failure: build-and-test
[SUCCESS] Fixed github failure
```

And a new commit in your git log!

---

## Get Your GitHub Token

### Step 1: Go to GitHub Settings

```
https://github.com/settings/tokens
```

### Step 2: Click "Generate new token" → "Generate new token (classic)"

### Step 3: Select Scopes

- ✅ `repo` (full control)
- ✅ `workflow` (GitHub Actions)

### Step 4: Copy Token

Keep it safe!

---

## Run Poller

```powershell
$env:GITHUB_TOKEN = "ghp_YOUR_TOKEN_HERE"

python ci-platform-poller.py `
    --github-owner YOUR_GITHUB_USERNAME `
    --github-repo YOUR_REPO_NAME
```

### Real Example:

```powershell
$env:GITHUB_TOKEN = "ghp_abc123def456ghi789jkl012mno345"

python ci-platform-poller.py `
    --github-owner mazhar-mehmood `
    --github-repo my-awesome-project
```

---

## Test It Works

### Option 1: Test Immediately

```powershell
python ci-platform-poller.py `
    --github-owner YOUR_NAME `
    --github-repo YOUR_REPO `
    --once
```

Output:
```
[INFO] Starting CI/CD Platform Poller
[DETECTED] github failure: build-and-test
[SUCCESS] Fixed github failure
```

### Option 2: Trigger a Real Failure

1. Push a broken build to GitHub
2. Let workflow fail
3. Run poller
4. Watch AI fix it automatically!

---

## Monitor the Auto-Fixes

```powershell
# See commits appear in real-time
git log --oneline --follow | head -10

# Should show:
# a1b2c3d Fix: Correct package version mismatch in requirements.txt
# e4f5g6h Fix: Update Dockerfile base image
```

---

## Multi-Platform Setup (Advanced)

### GitHub + GitLab + Kubernetes

```powershell
$env:GITHUB_TOKEN = "ghp_xxxx"
$env:GITLAB_TOKEN = "glpat_xxxx"

python ci-platform-poller.py `
    --github-owner YOU `
    --github-repo REPO `
    --gitlab-project-id 12345 `
    --enable-kubernetes
```

---

## Supported Platforms

### ✅ GitHub Actions (Recommended)

```powershell
python ci-platform-poller.py `
    --github-owner YOUR_NAME `
    --github-repo YOUR_REPO
```

### ✅ GitLab CI

```powershell
$env:GITLAB_TOKEN = "glpat_xxxx"

python ci-platform-poller.py `
    --gitlab-url https://gitlab.com `
    --gitlab-project-id 12345
```

### ✅ Kubernetes

```powershell
# Requires kubectl configured
python ci-platform-poller.py --enable-kubernetes
```

### ✅ System Logs (Linux/Mac)

```powershell
python ci-platform-poller.py --enable-syslog
```

### ✅ Docker Registry

```powershell
$env:DOCKER_USERNAME = "your_username"
$env:DOCKER_TOKEN = "your_api_token"

python ci-platform-poller.py
```

---

## Verification Checklist

- [ ] Ollama running (`ollama run llama3`)
- [ ] Python venv activated
- [ ] GitHub token set (`$env:GITHUB_TOKEN = "..."`)
- [ ] Poller running (`python ci-platform-poller.py ...`)
- [ ] No errors in output
- [ ] Check git log for auto-commits

---

## Real-World Example

### Scenario: Your GitHub workflow fails

**Your workflow file (.github/workflows/test.yml):**

```yaml
name: test
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: pip install -r requirements.txt
      - run: pytest
```

**Your requirements.txt has a bad version:**

```
requests==999.0.0  # ← Invalid!
```

### What Happens:

**1:00 PM** — Workflow fails
```
ERROR: Could not find a version that satisfies the requirement requests==999.0.0
```

**1:05 PM** — Poller detects failure
```
[DETECTED] github failure: test
[ANALYSIS] Invalid version specification in requirements.txt
```

**1:06 PM** — AI generates fix
```
Fixed File: requirements.txt
Fix: Change requests==999.0.0 to requests==2.31.0
```

**1:07 PM** — Auto-commit appears
```
git log --oneline
a1b2c3d Fix: Correct package version mismatch in requirements.txt
```

**You:** Did nothing! AI did everything! 🎉

---

## Troubleshooting

### "Connection refused"
```
Ollama not running?
Terminal 1: ollama run llama3
```

### "401 Unauthorized"
```
Bad GitHub token?
Check: https://github.com/settings/tokens
Regenerate token with correct scopes (repo + workflow)
```

### "No failures detected"
```
That's good! No broken workflows.
Or: Poller hasn't found failures yet (checks every 5 minutes)
```

### "AI fix looks wrong"
```
Edit the prompt in auto-fixer.py for your specific tech stack
Add context: frameworks, languages, specific patterns
```

---

## Next Steps

1. **Create GitHub token** → https://github.com/settings/tokens
2. **Set environment variable** → `$env:GITHUB_TOKEN = "ghp_..."`
3. **Run poller** → `python ci-platform-poller.py --github-owner YOU --github-repo REPO`
4. **Wait for failures** → Or trigger one manually to test
5. **Watch git commits** → `git log --oneline`

---

## That's It!

You now have a **fully automatic AI DevOps system**:

- ✅ No manual logs
- ✅ No manual fixes
- ✅ No manual commits
- ✅ Just let it run!

**The AI handles everything.**

---

**Questions?** See [README.md](README.md) or [ARCHITECTURE.md](ARCHITECTURE.md)
