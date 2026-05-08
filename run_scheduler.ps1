$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

$BundledPython = "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$PythonExe = $null

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($PythonCommand) {
    $PythonExe = $PythonCommand.Source
} elseif (Test-Path $BundledPython) {
    $PythonExe = $BundledPython
} else {
    throw "No python executable found. Install Python or update run_scheduler.ps1 with your python.exe path."
}

& $PythonExe .\a_share_crawler.py schedule
