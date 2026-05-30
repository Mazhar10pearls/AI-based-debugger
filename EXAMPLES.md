# Real-World Examples

Try these hands-on examples to see the auto-fixer in action.

## Example 1: Fix a Dependency Error

**Scenario:** Your Python app fails because of a missing dependency version.

### Step 1: Create a failing requirements.txt
```powershell
# Create a bad requirements file
@"
requests==999.0.0
flask==100.0
"@ | Out-File -Encoding UTF8 requirements.txt
```

### Step 2: Create an error log
```powershell
@"
ERROR: Could not find a version that satisfies the requirement requests==999.0.0
FAILED: pip install -r requirements.txt
"@ | Out-File -Encoding UTF8 logs/dep-error.log
```

### Step 3: Run auto-fixer
```powershell
python auto-fixer.py --input logs/dep-error.log --no-commit
```

**Result:** AI detects the invalid version number and suggests correct versions.

---

## Example 2: Fix a Docker Build Error

**Scenario:** Docker build fails because of missing base image tag.

### Step 1: Create a failing Dockerfile
```powershell
@"
FROM python  # ← Wrong! Should specify version
RUN pip install flask
COPY . /app
WORKDIR /app
CMD ["python", "app.py"]
"@ | Out-File -Encoding UTF8 Dockerfile
```

### Step 2: Create Docker build error log
```powershell
@"
Step 1/5 : FROM python
Error response from daemon: manifest not found for python:latest in docker.io
ERROR: Failed to build Docker image
"@ | Out-File -Encoding UTF8 logs/docker-error.log
```

### Step 3: Run auto-fixer
```powershell
python auto-fixer.py --input logs/docker-error.log --no-commit
```

**Result:** AI suggests `FROM python:3.11-slim` or appropriate Python version.

---

## Example 3: Continuous Monitoring Setup

**Scenario:** Want to auto-fix all failures in a CI/CD pipeline without manual work.

### Step 1: Start the watcher
```powershell
# Create logs directory
mkdir logs

# Start continuous watcher (checks every 60 seconds)
python workflow-watcher.py --watch-dir ./logs --interval 60
# This terminal now runs continuously (Ctrl+C to stop)
```

### Step 2: Simulate failures (in another terminal)
```powershell
# Simulate a failing test log
@"
FAILED: test_api.py - ConnectionError: Could not connect to database
ERROR: Database connection string missing from config.py
"@ | Out-File -Encoding UTF8 logs/test-failure-1.log

# Watcher detects it, analyzes, and fixes automatically!
```

### Step 3: Check git log
```powershell
git log --oneline -5
# You'll see auto-commits from the AI fixer
```

---

## Example 4: GitHub Actions Integration

**Scenario:** A GitHub workflow fails and you want AI to auto-fix it.

### Step 1: Set GitHub token
```powershell
$env:GITHUB_TOKEN = "ghp_YOUR_ACTUAL_TOKEN_HERE"
```

### Step 2: Fetch and analyze failed workflows
```powershell
python github-monitor.py --owner YOUR_GITHUB_USERNAME --repo YOUR_REPO --limit 3
```

**Output:**
```
[INFO] Fetching failed workflows for YOUR_USERNAME/YOUR_REPO...
[FOUND] 1 failed run(s)

Run: build-and-test (ID: 123456789)
Conclusion: failure
Head Branch: main
```

### Step 3: Auto-fix the workflow
```powershell
python github-monitor.py --owner YOUR_GITHUB_USERNAME --repo YOUR_REPO --auto-fix
# Downloads logs and triggers auto-fixer
```

---

## Example 5: Batch Fix Multiple Logs

**Scenario:** You have multiple CI/CD logs to fix at once.

```powershell
# Create sample logs
@"
ERROR: npm ERR! 404 Not Found - GET https://registry.npmjs.org/nonexistent-package
"@ | Out-File -Encoding UTF8 logs/npm-error.log

@"
ERROR: ModuleNotFoundError: No module named 'numpy'
"@ | Out-File -Encoding UTF8 logs/python-error.log

# Batch fix all logs in the directory
Get-ChildItem logs/*.log | ForEach-Object {
    Write-Host "Processing: $($_.Name)"
    python auto-fixer.py --input $_.FullName --no-commit
    Start-Sleep -Seconds 5  # Wait between requests
}
```

---

## Example 6: Dry-Run (No Commit)

**Scenario:** You want to see what the AI would fix without committing.

```powershell
# Analyze and generate fix, but don't commit
python auto-fixer.py --input logs/error.log --no-commit

# The fixed file is written locally
# You can review it before committing manually
```

---

## Example 7: Custom Ollama Model

**Scenario:** Mistral model is faster for your use case.

```powershell
# Pull Mistral model
ollama pull mistral

# Use it with auto-fixer
python auto-fixer.py --input logs/error.log --model mistral
```

---

## Example 8: Scheduled Daily Auto-Fixing

**Scenario:** Run auto-fixer every day at 2 AM to catch any overnight failures.

### On Windows (Task Scheduler):
```powershell
# Create a scheduled task
$trigger = New-ScheduledTaskTrigger -Daily -At 2:00AM
$action = New-ScheduledTaskAction -Execute "C:\path\to\python.exe" `
    -Argument "C:\path\to\auto-fixer.py --input latest-error.log"
Register-ScheduledTask -TaskName "DailyAutoFixer" -Trigger $trigger -Action $action -RunLevel Highest
```

### On Linux (cron):
```bash
# Add to crontab
0 2 * * * cd /home/ubuntu/AI-BASED-debugger && python auto-fixer.py --input latest-error.log
```

---

## Example 9: Monitoring Multiple Projects

**Scenario:** Monitor failures from multiple repos/projects.

```powershell
# Create separate watch directories
mkdir logs-project-1
mkdir logs-project-2

# Start multiple watchers
Start-Job { python workflow-watcher.py --watch-dir ./logs-project-1 }
Start-Job { python workflow-watcher.py --watch-dir ./logs-project-2 }

# Each watcher runs independently and auto-fixes
Get-Job  # View all running jobs
```

---

## Example 10: Integration with Slack Alerts

**Scenario:** Get Slack notifications when AI fixes something.

Create a wrapper script `slack-notifier.py`:
```python
import subprocess
import requests

# Run auto-fixer
result = subprocess.run(["python", "auto-fixer.py", "--input", "error.log"])

# Send Slack notification
if result.returncode == 0:
    requests.post(os.environ["SLACK_WEBHOOK"], json={
        "text": "✅ Auto-fixer applied a fix!"
    })
```

Then run:
```powershell
$env:SLACK_WEBHOOK = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
python slack-notifier.py
```

---

## Tips for Success

1. **Start small:** Test with `--no-commit` first
2. **Monitor logs:** Watch the output to understand what's being fixed
3. **Review changes:** Before enabling auto-commit, check `git diff`
4. **Test in non-prod:** Start with a test repo before production
5. **Keep Ollama running:** The AI engine must be available
6. **Handle edge cases:** Some failures may need manual review

---

## Troubleshooting Examples

### AI response is wrong?
- The prompt might be too vague
- Edit the `PROMPT_TEMPLATE` in auto-fixer.py
- Add more context about your specific stack

### Commits too frequent?
- Increase watcher `--interval` (default: 300 seconds)
- Batch process instead of continuous monitoring

### Out of memory?
- Use a smaller model: `ollama pull phi` or `ollama pull neural-chat`
- Reduce `max_tokens` in auto-fixer.py

---

**Ready to try?** Start with Example 1, then move to Example 3 for continuous automation!
