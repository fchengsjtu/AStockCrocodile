param(
  [string]$TradeDate = (Get-Date -Format "yyyyMMdd"),
  [string]$MainPython = "",
  [int]$CrawlerWorkers = 8
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

$LogDir = Join-Path $ProjectDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("daily_after_close_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Start-Transcript -Path $LogFile -Append | Out-Null

function Resolve-Python {
  param(
    [string]$Requested,
    [string[]]$Candidates
  )
  if ($Requested -and (Test-Path $Requested)) {
    return (Resolve-Path $Requested).Path
  }
  foreach ($candidate in $Candidates) {
    if ($candidate -and (Test-Path $candidate)) {
      return (Resolve-Path $candidate).Path
    }
  }
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) {
    return $cmd.Source
  }
  throw "No Python executable found. Pass -MainPython explicitly."
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

function Get-TradeDateValue {
  return [datetime]::ParseExact($TradeDate, "yyyyMMdd", $null)
}

function Test-WeeklyDue {
  return ((Get-TradeDateValue).DayOfWeek -eq [System.DayOfWeek]::Friday)
}

function Test-MonthlyDue {
  & $MainPythonExe -c "from datetime import datetime; from a_share_crawler import is_last_trade_day; raise SystemExit(0 if is_last_trade_day(datetime.strptime('$TradeDate', '%Y%m%d').date()) else 1)"
  return ($LASTEXITCODE -eq 0)
}

$MainPythonExe = Resolve-Python -Requested $MainPython -Candidates @(".\.venv\Scripts\python.exe")

Write-Host "ProjectDir=$ProjectDir"
Write-Host "TradeDate=$TradeDate"
Write-Host "MainPython=$MainPythonExe"
Write-Host "LogFile=$LogFile"

try {
  Invoke-Step "crawl daily K-lines" $MainPythonExe @(
    ".\a_share_crawler.py", "run",
    "--mode", "incremental",
    "--period", "daily",
    "--end-date", $TradeDate,
    "--workers", [string]$CrawlerWorkers
  )

  if (Test-WeeklyDue) {
    Invoke-Step "generate weekly K-lines" $MainPythonExe @(".\a_share_crawler.py", "generate", "--period", "weekly")
  } else {
    Write-Host "Skip weekly K-line generation: trade date is not Friday."
  }

  if (Test-MonthlyDue) {
    Invoke-Step "generate monthly K-lines" $MainPythonExe @(".\a_share_crawler.py", "generate", "--period", "monthly")
  } else {
    Write-Host "Skip monthly K-line generation: trade date is not the last trading day of the month."
  }

  Write-Host "Daily after-close K-line workflow completed."
}
finally {
  Stop-Transcript | Out-Null
}
