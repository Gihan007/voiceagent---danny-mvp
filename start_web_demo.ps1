# Browser voice demo launcher.
# This starts the small local frontend at http://localhost:8002.

$PROJECT_ROOT = $PSScriptRoot
$ENV_FILE = Join-Path $PROJECT_ROOT ".env"
$PORT = 8002

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Voice Agent Browser Demo" -ForegroundColor Cyan
Write-Host "  Frontend + demo API: http://localhost:$PORT" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

if (-not (Test-Path $ENV_FILE)) {
    Write-Host ""
    Write-Host "[ERROR] .env not found. Copy .env.example to .env and add provider keys." -ForegroundColor Red
    exit 1
}

$envContent = Get-Content $ENV_FILE -Raw
foreach ($key in @("DEEPGRAM_API_KEY","GOOGLE_API_KEY")) {
    if ($envContent -notmatch "$key=\S+" -or $envContent -match "$key=YOUR_") {
        Write-Host "[WARN] $key looks unset in .env" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Starting uvicorn..." -ForegroundColor Cyan
Write-Host "Open http://localhost:$PORT and click Start Demo." -ForegroundColor Green
Write-Host "Press Ctrl+C here to stop." -ForegroundColor DarkGray

Set-Location $PROJECT_ROOT
python -m uvicorn web_voice_demo:app --host 127.0.0.1 --port $PORT --reload
