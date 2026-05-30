# AI Workflow Analyzer

A beginner-friendly local AI project that analyzes CI/CD workflow logs using Ollama + Llama3.

## What it does

- Reads CI/CD workflow or failure logs
- Sends logs to a local Ollama LLM
- Detects probable root causes
- Identifies failed components
- Suggests remediation steps
- Summarizes the failure

## Prerequisites

- Python 3.10+
- Ollama installed on your VM
- A local Ollama model such as `llama3` pulled and available

## Setup

1. Install Ollama:

   - Follow the instructions at https://ollama.com/download

2. Pull the Llama3 model:

```powershell
ollama pull llama3
```

3. Create a Python virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> If you are using a Linux VM, replace the activation command with:
> `source .venv/bin/activate`

## Run Ollama locally

Start the Ollama HTTP service before running the analyzer:

```powershell
ollama run llama3
```

This exposes the local API at `http://127.0.0.1:11434`.

## Run the analyzer

```powershell
python analyzer.py --input sample-log.txt
```

Or automatically analyze the newest log file in a directory:

```powershell
python analyzer.py --log-dir .\logs
```

To save the analysis to a file:

```powershell
python analyzer.py --input sample-log.txt --output analysis.txt
```

## GitHub issue update

You can post the AI analysis directly to a GitHub issue comment.

```powershell
$env:GITHUB_TOKEN = "<your-token>"
python analyzer.py --input sample-log.txt --github-repo owner/repo --github-issue 123
```

If you do not want to use the environment variable, pass the token explicitly:

```powershell
python analyzer.py --input sample-log.txt --github-repo owner/repo --github-issue 123 --github-token <your-token>
```

## Custom API settings

If Ollama is listening on a different address, use:

```powershell
python analyzer.py --input sample-log.txt --api-url http://127.0.0.1:11434/v1/completions
```

If you want to use a different model name:

```powershell
python analyzer.py --input sample-log.txt --model llama3
```

## Example log file

A sample log file is included in `sample-log.txt`. Replace it with your own GitHub Actions, Docker build, Kubernetes, or CI/CD error logs.

To automatically analyze logs, save workflow output to a directory and use `--log-dir`.

## Notes for your VM

Your VM has about 8 GB of RAM. That is enough for many local Ollama models, but if `llama3` does not fit, you can switch to a smaller locally-supported model via Ollama.

## Next steps

- Add GitHub Actions integration
- Support multiple log formats
- Add structured JSON output for automation
- Extend the prompt with remediation templates
