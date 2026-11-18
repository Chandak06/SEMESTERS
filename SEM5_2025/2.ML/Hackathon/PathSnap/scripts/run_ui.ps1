param([string]$ApiUrl="http://127.0.0.1:8000",[int]$Port=8501)
$ErrorActionPreference="Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PATHSNAP_API_URL=$ApiUrl
& .\.venv\Scripts\python.exe -m streamlit run ui\app.py --server.port $Port --server.headless true --browser.gatherUsageStats false

