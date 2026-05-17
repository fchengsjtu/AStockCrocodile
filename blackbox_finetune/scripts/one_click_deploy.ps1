$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

function Invoke-Step {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$CommandArgs)
  & $VenvPython @CommandArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code ${LASTEXITCODE}: $VenvPython $($CommandArgs -join ' ')"
  }
}

$Mode = if ($args.Count -gt 0) { $args[0] } else { "smoke" }
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } elseif (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { ".\.venv-blackbox-finetune" }

if (!(Test-Path $VenvDir)) {
  & $PythonBin -m venv $VenvDir
}
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
Invoke-Step -m pip install --upgrade pip wheel "setuptools<82"
Invoke-Step -m pip install -r .\blackbox_finetune\requirements.txt

$BaseModel = if ($env:BASE_MODEL) { $env:BASE_MODEL } else { "Qwen/Qwen2.5-0.5B-Instruct" }
$DataDir = if ($env:DATA_DIR) { $env:DATA_DIR } else { "blackbox_finetune/data" }
$ValidationDir = if ($env:VALIDATION_DATA_DIR) { $env:VALIDATION_DATA_DIR } else { "blackbox_finetune/data_validation" }
$OutputDir = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "blackbox_finetune/runs/qwen2.5-0.5b-blackbox-lora" }
$MinRecall = if ($env:MIN_POSITIVE_RECALL) { $env:MIN_POSITIVE_RECALL } else { "0.60" }

if ($Mode -eq "smoke") {
  $TrainStart = "20110101"
  $TrainEnd = "20151231"
  $ValidationStart = "20260101"
  $ValidationEnd = "20260131"
  $PositiveLimit = if ($env:SMOKE_POSITIVE_LIMIT) { $env:SMOKE_POSITIVE_LIMIT } else { "12" }
  $Epochs = if ($env:EPOCHS) { $env:EPOCHS } else { "3" }
  $MaxSeqLength = if ($env:MAX_SEQ_LENGTH) { $env:MAX_SEQ_LENGTH } else { "512" }
  $GradSteps = if ($env:GRADIENT_ACCUMULATION_STEPS) { $env:GRADIENT_ACCUMULATION_STEPS } else { "1" }
} else {
  $TrainStart = "20110101"
  $TrainEnd = "20241231"
  $ValidationStart = "20260101"
  $ValidationEnd = "20260430"
  $PositiveLimit = $env:POSITIVE_LIMIT
  $Epochs = if ($env:EPOCHS) { $env:EPOCHS } else { "1" }
  $MaxSeqLength = if ($env:MAX_SEQ_LENGTH) { $env:MAX_SEQ_LENGTH } else { "2048" }
  $GradSteps = if ($env:GRADIENT_ACCUMULATION_STEPS) { $env:GRADIENT_ACCUMULATION_STEPS } else { "8" }
}

$BuildArgs = @("-m", "blackbox_finetune.build_dataset", "--output-dir", $DataDir, "--start-date", $TrainStart, "--end-date", $TrainEnd, "--negative-ratio", "1.0")
if ($PositiveLimit) { $BuildArgs += @("--positive-limit", $PositiveLimit) }
Invoke-Step @BuildArgs

$ValArgs = @("-m", "blackbox_finetune.build_validation_dataset", "--output-dir", $ValidationDir, "--start-date", $ValidationStart, "--end-date", $ValidationEnd, "--negative-ratio", "1.0")
if ($Mode -eq "smoke") { $ValArgs += @("--positive-limit", $PositiveLimit) }
Invoke-Step @ValArgs

Invoke-Step -m blackbox_finetune.train --base-model $BaseModel --data-dir $DataDir --output-dir $OutputDir --max-seq-length $MaxSeqLength --epochs $Epochs --batch-size 1 --gradient-accumulation-steps $GradSteps --learning-rate 2e-4 --no-4bit
Invoke-Step -m blackbox_finetune.evaluate --base-model $BaseModel --adapter-dir "$OutputDir/adapter" --data-dir $ValidationDir --threshold 0.50 --min-positive-recall $MinRecall
Invoke-Step -m unittest tests.test_blackbox_finetune -v
