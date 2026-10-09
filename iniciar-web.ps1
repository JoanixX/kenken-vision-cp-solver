param([int]$Port = 8000, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo crear el entorno Python.' }
}
& $taskPython -c 'import fastapi, uvicorn, multipart, torch, cv2, ortools, gradio'
if ($LASTEXITCODE -ne 0) {
    & $taskPython -m pip install -r requirements.txt -r web/requirements.txt 'gradio>=5,<7'
    if ($LASTEXITCODE -ne 0) { throw 'No se pudieron instalar las dependencias.' }
}
& $taskPython web/build.py
if ($LASTEXITCODE -ne 0) { throw 'No se pudo construir la interfaz.' }
Write-Host "KenKen Lab: http://127.0.0.1:$Port"
if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:$Port"
}
& $taskPython -m uvicorn web.server:app --host 127.0.0.1 --port $Port
