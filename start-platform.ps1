$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvActivate = Join-Path $root ".venv\Scripts\Activate.ps1"
$backendPath = Join-Path $root "backend"
$frontendFile = Join-Path $root "frontend\index.html"

if (Test-Path $venvActivate) {
    . $venvActivate
}

$backendCommand = @"
Set-Location '$backendPath';
if (Test-Path '..\\.venv\\Scripts\\Activate.ps1') { . '..\\.venv\\Scripts\\Activate.ps1' }
uvicorn main:app --reload
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCommand | Out-Null

$backendReady = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/" -UseBasicParsing -TimeoutSec 1
        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
            $backendReady = $true
            break
        }
    } catch {
        Start-Sleep -Milliseconds 500
    }
}

if (Test-Path $frontendFile) {
    Start-Process $frontendFile | Out-Null
}

if ($backendReady) {
    Write-Host "Platform started. Backend: http://127.0.0.1:8000 | Frontend: frontend/index.html"
} else {
    Write-Warning "Frontend opened, but backend did not become ready in time. If UI shows 'Failed to fetch', check backend terminal for startup errors."
}
