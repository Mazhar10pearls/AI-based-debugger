# AI Workflow Auto-Fixer - Architecture & Usage Guide

## 🎯 Your Complete Automated DevOps System

You now have **4 powerful tools** that work together:

### Tool Comparison

| Feature | `analyzer.py` | `auto-fixer.py` | `workflow-watcher.py` | `github-monitor.py` |
|---------|:---:|:---:|:---:|:---:|
| Manual analysis | ✅ | ❌ | ❌ | ❌ |
| Auto-detect failures | ❌ | ✅ | ✅ | ✅ |
| Auto-fix code | ❌ | ✅ | ✅ | ❌ |
| Auto-commit | ❌ | ✅ | ✅ | ❌ |
| Continuous monitoring | ❌ | ❌ | ✅ | ❌ |
| GitHub Actions | ❌ | ❌ | ❌ | ✅ |
| Dry-run mode | ✅ | ✅ | ✅ | ✅ |

---

## 📋 Workflows You Can Enable

### Workflow 1: One-Time Manual Fix
```powershell
# Found a failing log? Fix it once with auto-commit
python auto-fixer.py --input build-error.log
# → AI analyzes, generates fix, commits automatically
```

### Workflow 2: Continuous Auto-Fix (Recommended)
```powershell
# Start watching a directory for new failures
python workflow-watcher.py --watch-dir ./logs --interval 60

# In another terminal, drop logs and they auto-fix
New-Item -Path ./logs/my-failure.log -Value "ERROR: dependency issue"
# → Watcher detects it, analyzes, fixes, commits
```

### Workflow 3: GitHub Actions Auto-Fix
```powershell
# GitHub job fails? Auto-fix it
export GITHUB_TOKEN="ghp_xxxx"
python github-monitor.py --owner myname --repo myrepo --auto-fix
# → Fetches failed workflows, analyzes logs, commits fixes
```

### Workflow 4: Manual Analysis Only
```powershell
# Just analyze, don't fix
python analyzer.py --input error.log --output report.txt
# → Generates insights without modifying code
```

---

## 🔧 Setup Checklist

- [ ] Python 3.10+ installed
- [ ] `ollama pull llama3` completed
- [ ] Virtual environment created & activated
- [ ] `pip install -r requirements.txt`
- [ ] `ollama run llama3` running in Terminal 1
- [ ] Ready to run scripts in Terminal 2+

---

## 🚀 Quick Test

1. **Start Ollama (Terminal 1):**
   ```powershell
   ollama run llama3
   ```

2. **Test Auto-Fixer (Terminal 2):**
   ```powershell
   python quickstart.py
   # Or directly:
   python auto-fixer.py --input sample-log.txt --no-commit
   ```

3. **Try Continuous Mode (Terminal 2):**
   ```powershell
   mkdir logs
   python workflow-watcher.py --watch-dir ./logs --interval 60
   # Then in Terminal 3, drop new logs and watch them auto-fix
   ```

---

## 🎓 What Each Script Does

### `analyzer.py` — Manual Analysis
- **Input:** Failure log file
- **Output:** AI-generated insights & recommendations
- **Commitment:** None (analysis only)
- **Best for:** Understanding failures, dry runs

### `auto-fixer.py` — Automated Fix + Commit
- **Input:** Single failure log
- **Process:**
  1. Detects failure type
  2. Reads file content
  3. Sends to Ollama + Llama3
  4. Gets JSON response with fix
  5. Writes fixed file
  6. Auto-commits to git
- **Best for:** One-off fixes with git integration

### `workflow-watcher.py` — Continuous Monitor
- **Input:** Directory to watch
- **Process:**
  1. Scans directory every N seconds
  2. Finds new failure logs
  3. Calls auto-fixer.py on each
  4. Tracks processed files
- **Best for:** Always-on automation, zero-touch

### `github-monitor.py` — GitHub Integration
- **Input:** GitHub repo credentials
- **Process:**
  1. Queries GitHub API for failed workflows
  2. Downloads logs from each failure
  3. Triggers auto-fixer
  4. Can create PRs with fixes
- **Best for:** GitHub Actions automation

---

## 📊 Failure Types Detected

The auto-fixer automatically recognizes:

1. **Python Dependency Errors**
   - Missing module imports
   - Version conflicts in requirements.txt
   
2. **Docker Build Failures**
   - Missing dependencies
   - Wrong base images
   - Permission issues

3. **GitHub Actions Workflows**
   - YAML syntax errors
   - Job failures
   - Step timeouts

4. **CI/CD Build Failures**
   - Test failures
   - Compilation errors
   - Build step failures

5. **Node.js Dependency Issues**
   - Missing packages
   - Version mismatches in package.json

---

## 🔐 Security Notes

### Git Commits
- ⚠️ Auto-fixer commits directly to `main`
- Use `--no-commit` flag for dry runs
- Review fixes before enabling auto-commit in production

### GitHub Token
- Store token in `GITHUB_TOKEN` env var (never in code)
- Token needs `repo` and `workflow` scopes
- Consider using GitHub personal access tokens with limited scope

### Ollama API
- Runs locally on `127.0.0.1:11434` (not exposed to internet)
- No external API calls for AI (privacy-first)

---

## 🛠️ Customization

### Use Different Ollama Model
```powershell
$env:OLLAMA_MODEL = "mistral"
python auto-fixer.py --input error.log
```

### Change API Endpoint
```powershell
python auto-fixer.py --input error.log --api-url http://custom-ollama:11434/v1/completions
```

### Watch Multiple Directories
```powershell
# Run multiple watchers
Start-Job { python workflow-watcher.py --watch-dir ./logs1 }
Start-Job { python workflow-watcher.py --watch-dir ./logs2 }
```

### Run on Schedule (Windows Task Scheduler)
```powershell
# Run auto-fixer every 15 minutes
$trigger = New-ScheduledTaskTrigger -RepetitionInterval (New-TimeSpan -Minutes 15) -At 00:00:00 -Once
$action = New-ScheduledTaskAction -Execute "python" -Argument "auto-fixer.py --input ./latest-error.log"
Register-ScheduledTask -TaskName "AutoFixerDaily" -Trigger $trigger -Action $action
```

---

## 📈 Performance Tips

### For Your 8GB VM:

1. **Monitor Resource Usage:**
   ```powershell
   Get-Process | Where-Object {$_.Name -eq "ollama"} | Select-Object Name, WorkingSet
   ```

2. **Reduce AI Response Time:**
   - Lower `max_tokens` in auto-fixer.py (default: 2048)
   - Use smaller model: `ollama pull mistral` or `ollama pull neural-chat`

3. **Batch Process Logs:**
   ```powershell
   Get-ChildItem ./logs/*.log | ForEach-Object {
       python auto-fixer.py --input $_.FullName
   }
   ```

---

## 🐛 Troubleshooting

### "Connection refused" error
```powershell
# Ollama not running?
ollama run llama3  # Start in another terminal
```

### "No module named requests"
```powershell
pip install requests
```

### Auto-fixer hangs
```powershell
# Increase timeout or check Ollama logs
# Model might be out of memory
ollama ps  # Check running models
```

### Git commits fail
```powershell
# Git not configured?
git config --global user.name "AI Fixer"
git config --global user.email "ai@fixer.local"
```

---

## 🎯 Next Steps

1. **Test it:** Run `python quickstart.py`
2. **Go continuous:** `python workflow-watcher.py --watch-dir ./logs`
3. **Integrate with GitHub:** Set up `github-monitor.py` with your token
4. **Customize:** Modify prompts in the scripts for your use cases
5. **Scale:** Add more failure detectors for your specific stack

---

## 📚 Learn More

- **Ollama:** https://ollama.com
- **Llama3:** https://llama.meta.com
- **GitHub API:** https://docs.github.com/en/rest

---

**Status:** ✅ Complete automated AI-powered DevOps system ready to go!
