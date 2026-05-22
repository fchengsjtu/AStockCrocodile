param(
  [string]$TradeDate = (Get-Date -Format "yyyyMMdd"),
  [string]$Python = "",
  [int]$Workers = 8,
  [double]$Sleep = 0.05,
  [int]$Retries = 3,
  [string]$KType = "D",
  [switch]$UseEnvProxy,
  [switch]$SkipExrights,
  [switch]$SkipKlineCrawl
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

$LogDir = Join-Path $ProjectDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("daily_incremental_with_exrights_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Start-Transcript -Path $LogFile -Append | Out-Null

function Resolve-Python {
  param([string]$Requested)

  if ($Requested -and (Test-Path $Requested)) {
    return (Resolve-Path $Requested).Path
  }

  $VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
  if (Test-Path $VenvPython) {
    return (Resolve-Path $VenvPython).Path
  }

  $Command = Get-Command python -ErrorAction SilentlyContinue
  if ($Command) {
    return $Command.Source
  }

  throw "No Python executable found. Pass -Python or create .venv first."
}

function Invoke-Step {
  param(
    [string]$Name,
    [string]$Exe,
    [string[]]$Arguments
  )

  Write-Host ""
  Write-Host "==== $Name ===="
  Write-Host "$Exe $($Arguments -join ' ')"
  & $Exe @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Step failed: $Name exit=$LASTEXITCODE"
  }
}

$PythonExe = Resolve-Python -Requested $Python
$CommonArgs = @()
if ($UseEnvProxy) {
  $CommonArgs += "--use-env-proxy"
}

Write-Host "ProjectDir=$ProjectDir"
Write-Host "TradeDate=$TradeDate"
Write-Host "Python=$PythonExe"
Write-Host "Workers=$Workers"
Write-Host "Sleep=$Sleep"
Write-Host "Retries=$Retries"
Write-Host "KType=$KType"
Write-Host "LogFile=$LogFile"

try {
  if (-not $SkipExrights) {
    $ExrightsArgs = @(
      ".\a_share_crawler.py", "exrights",
      "--end-date", $TradeDate,
      "--workers", [string]$Workers,
      "--sleep", [string]$Sleep,
      "--retries", [string]$Retries,
      "--ktype", $KType
    ) + $CommonArgs
    Invoke-Step "crawl exrights and refresh changed qfq K-lines" $PythonExe $ExrightsArgs
  } else {
    Write-Host "Skip exrights crawl."
  }

  if (-not $SkipKlineCrawl) {
    $KlineArgs = @(
      ".\a_share_crawler.py", "run",
      "--mode", "incremental",
      "--period", "daily",
      "--end-date", $TradeDate,
      "--workers", [string]$Workers,
      "--sleep", [string]$Sleep,
      "--retries", [string]$Retries,
      "--ktype", $KType
    ) + $CommonArgs
    Invoke-Step "crawl incremental daily K-lines" $PythonExe $KlineArgs
  } else {
    Write-Host "Skip incremental daily K-line crawl."
  }

  Write-Host "Daily incremental workflow completed."
}
finally {
  Stop-Transcript | Out-Null
}
