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
$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { ".\.venv-qwen-finetune" }

if (!(Test-Path $VenvDir)) {
  & $PythonBin -m venv $VenvDir
}
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
Invoke-Step -m pip install --upgrade pip wheel "setuptools<82"
Invoke-Step -m pip install -r .\llm_finetune\requirements.txt

if (Test-Path .\llm_finetune\config.env) {
  Get-Content .\llm_finetune\config.env | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)\s*$') {
      [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
  }
}

$BaseModel = if ($env:BASE_MODEL) { $env:BASE_MODEL } else { "Qwen/Qwen2.5-0.5B-Instruct" }
$DataDir = if ($env:DATA_DIR) { $env:DATA_DIR } else { "llm_finetune/data" }
$OutputDir = if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "llm_finetune/runs/qwen2.5-0.5b-stock-lora" }
$MinSuccessRate = if ($env:MIN_SUCCESS_RATE) { $env:MIN_SUCCESS_RATE } else { "0.20" }
$DefaultMaxSeqLength = if ($Mode -eq "smoke") { "512" } else { "2048" }
$DefaultEpochs = if ($Mode -eq "smoke") { "5" } else { "1" }
$DefaultGradientAccumulationSteps = if ($Mode -eq "smoke") { "1" } else { "8" }
$MaxSeqLength = if ($env:MAX_SEQ_LENGTH) { $env:MAX_SEQ_LENGTH } else { $DefaultMaxSeqLength }
$Epochs = if ($env:EPOCHS) { $env:EPOCHS } else { $DefaultEpochs }
$SmokeLimit = if ($env:SMOKE_POSITIVE_LIMIT) { $env:SMOKE_POSITIVE_LIMIT } else { "200" }
$GradientAccumulationSteps = if ($env:GRADIENT_ACCUMULATION_STEPS) { $env:GRADIENT_ACCUMULATION_STEPS } else { $DefaultGradientAccumulationSteps }
$BatchSize = if ($env:BATCH_SIZE) { $env:BATCH_SIZE } else { "1" }
$LearningRate = if ($env:LEARNING_RATE) { $env:LEARNING_RATE } else { "2e-4" }
$EvalMaxSamples = if ($env:EVAL_MAX_SAMPLES) { $env:EVAL_MAX_SAMPLES } else { "200" }

$DataArgs = @("--output-dir", $DataDir, "--negative-ratio", "1.0", "--batch-size", "30")
if ($Mode -eq "smoke") {
  $DataArgs += @("--positive-limit", $SmokeLimit)
}

Invoke-Step -m llm_finetune.build_dataset @DataArgs
Invoke-Step -m llm_finetune.train --base-model $BaseModel --data-dir $DataDir --output-dir $OutputDir --max-seq-length $MaxSeqLength --epochs $Epochs --batch-size $BatchSize --gradient-accumulation-steps $GradientAccumulationSteps --learning-rate $LearningRate --no-4bit
Invoke-Step -m llm_finetune.evaluate --base-model $BaseModel --adapter-dir "$OutputDir/adapter" --data-dir $DataDir --min-success-rate $MinSuccessRate --max-samples $EvalMaxSamples
Invoke-Step -m unittest tests.test_llm_finetune -v
