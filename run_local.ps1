param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$venvPython = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating project virtual environment..."
    python -m venv (Join-Path $root ".venv")
}

if (-not $SkipInstall) {
    Write-Host "Installing project dependencies..."
    & $venvPython -m pip install --index-url https://pypi.org/simple --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip." }
    & $venvPython -m pip install --index-url https://pypi.org/simple -r (Join-Path $root "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}

Write-Host "Starting Reddit RabbitHole RAG at http://127.0.0.1:7860/"
& $venvPython (Join-Path $root "app.py")
exit $LASTEXITCODE
