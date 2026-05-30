# AI-Powered Workflow Auto-Fixer

🤖 **Zero manual intervention DevOps automation** — Automatically detects CI/CD failures, analyzes them with local AI, and fixes them with auto-commit.

## What it does

| Tool | Purpose |
|------|---------|
| **analyzer.py** | Analyzes logs and generates insights (manual mode) |
| **auto-fixer.py** | **[NEW]** Auto-detects failures, generates fixes, and auto-commits |
| **workflow-watcher.py** | **[NEW]** Continuously monitors a directory for new failures and triggers auto-fixer |
| **github-monitor.py** | **[NEW]** Monitors GitHub Actions workflow failures |

### Supported Failure Types

- ✅ GitHub Actions workflow failures
- ✅ Docker build errors
- ✅ Python dependency conflicts (requirements.txt)
- ✅ Node.js dependency issues (package.json)
- ✅ CI/CD build log failures
- ✅ Generic error logs

---

## Quick Start

### 1. Install & Setup

```powershell
# Clone or enter the project
cd AI-BASED-debugger

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Start Ollama (Terminal 1)

```powershell
ollama pull llama3
ollama run llama3
```

### 3. Run Auto-Fixer (Terminal 2)

#### Option A: Single failure analysis + auto-commit

```powershell
# Detect failure, generate fix, and commit
python auto-fixer.py --input sample-log.txt
```

#### Option B: Continuous monitoring (watch a directory)

```powershell
# Create logs directory
mkdir logs

# Start watcher (checks every 5 minutes)
python workflow-watcher.py --watch-dir ./logs --interval 300

# In another terminal, drop new failure logs into ./logs/
# The watcher will automatically fix them and commit
```

#### Option C: GitHub Actions monitoring

```powershell
# Set your GitHub token
$env:GITHUB_TOKEN="ghp_xxxxxxxxxxxx"

# Fetch failed workflows and trigger auto-fixer
python github-monitor.py --owner YOUR_NAME --repo YOUR_REPO --auto-fix
```

---

## Usage Examples

### Single-Shot Fix (Manual Trigger)

```powershell
# Fix a failure log once
python auto-fixer.py --input build-error.log

# Fix without committing (dry run)
python auto-fixer.py --input build-error.log --no-commit
```

### Continuous Auto-Fix Mode

```powershell
# Start watcher in background
Start-Job -ScriptBlock { python workflow-watcher.py --watch-dir ./logs --interval 60 }

# Drop new logs, they're automatically fixed
Copy-Item build-error.log ./logs/
# → Watcher detects it, analyzes, fixes, commits
```

### Analyze Only (No Fix)

```powershell
# Just analyze, don't fix
python analyzer.py --input sample-log.txt --output report.txt
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│ New Failure Detected (log file)                         │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ FailureDetector                                         │
│ • GitHub Actions workflow?                              │
│ • Docker build error?                                   │
│ • Dependency conflict?                                  │
│ • CI/CD build failure?                                  │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ WorkflowAnalyzer (Ollama + Llama3)                      │
│ • Sends failure to local LLM                            │
│ • Receives: root cause, fix, commit message             │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ AutoFixer                                               │
│ • Writes fixed file to disk                             │
│ • Runs: git add, git commit, git push                   │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
        ✅ Fixed & Committed
```

---

## Configuration

### Environment Variables

```powershell
$env:OLLAMA_API_URL = "http://127.0.0.1:11434/v1/completions"
$env:OLLAMA_MODEL = "llama3"
$env:GITHUB_TOKEN = "ghp_xxxxxxxxxxxx"
```

### Auto-Fixer Options

```powershell
python auto-fixer.py --help

# Key options:
#   --input FILE              : Failure log to analyze
#   --no-commit              : Apply fix but don't git push
#   --model MODEL            : Use different Ollama model
#   --api-url URL            : Custom Ollama endpoint
#   --repo PATH              : Path to git repository
```

### Watcher Options

```powershell
python workflow-watcher.py --help

# Key options:
#   --watch-dir DIR          : Directory to monitor for logs
#   --interval SECONDS       : Check frequency (default: 300)
#   --auto-fixer SCRIPT      : Path to auto-fixer.py
#   --once                   : Run once and exit
```

---

## Example: GitHub Actions Auto-Fix Workflow

1. GitHub Actions job fails
2. Logs posted to your repo
3. Run: `python github-monitor.py --owner YOU --repo REPO --auto-fix`
4. AI detects the failure
5. Fix is generated and auto-committed
6. PR created with the fix (optional)

---

## Project Structure

```
AI-BASED-debugger/
├── analyzer.py           # Manual log analysis
├── auto-fixer.py         # Automatic detection & fix + commit
├── workflow-watcher.py   # Continuous directory monitoring
├── github-monitor.py     # GitHub Actions integration
├── requirements.txt      # Dependencies
├── sample-log.txt        # Example failure log
└── README.md            # This file
```

---

## VM Resource Notes

**Your VM Specs:**
- 8 GB RAM (sufficient for llama3)
- 2+ CPU cores (good)

If you encounter memory issues:
1. Use a smaller model: `ollama pull mistral` or `ollama pull phi`
2. Reduce max_tokens in auto-fixer.py
3. Monitor with `top` or Task Manager

---

## Limitations & Future

### Current Limitations
- Fixes are generated once and committed immediately (no review step)
- Works best with structured logs (GitHub Actions, Docker, pytest)
- Limited to local Ollama models

### Future Improvements
- PR creation with fixes for review
- Slack notifications
- Historical failure learning
- Multi-model support
- WebUI dashboard
