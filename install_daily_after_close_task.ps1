param(
  [string]$TaskName = "AStockCrocodile Daily After Close",
  [string]$At = "16:00",
  [string]$ScriptPath = ""
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ScriptPath) {
  $ScriptPath = Join-Path $ProjectDir "run_daily_after_close.ps1"
}
if (-not (Test-Path $ScriptPath)) {
  throw "Daily workflow script not found: $ScriptPath"
}

$action = New-ScheduledTaskAction `
  -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`"" `
  -WorkingDirectory $ProjectDir
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -StartWhenAvailable `
  -ExecutionTimeLimit (New-TimeSpan -Hours 12)

Register-ScheduledTask `
  -TaskName $TaskName `
  -Action $action `
  -Trigger $trigger `
  -Settings $settings `
  -Description "Run AStockCrocodile daily K-line crawl, derived K-lines, blackbox predictions, and portfolio tracking." `
  -Force | Out-Null

Write-Host "Installed scheduled task '$TaskName' to run daily at $At."
