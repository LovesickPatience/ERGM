param(
    [int]$ProcessId = 0,
    [switch]$Kill
)

$ErrorActionPreference = "SilentlyContinue"

function Get-TrainProc {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'main_with_selector' }
}

$procs = @()
if ($ProcessId -gt 0) {
    $p = Get-Process -Id $ProcessId
    if ($p) { $procs = @([pscustomobject]@{ ProcessId = $ProcessId }) }
} else {
    $procs = @(Get-TrainProc)
}

if ($Kill) {
    if (-not $procs) {
        Write-Host "[KILL] no running main_with_selector process found" -ForegroundColor Yellow
        exit
    }
    foreach ($p in $procs) {
        Write-Host "[KILL] stopping PID $($p.ProcessId) ..." -ForegroundColor Yellow
        Stop-Process -Id $p.ProcessId -Force
    }
    Start-Sleep -Seconds 6
    $left = @(Get-TrainProc)
    if ($left) {
        Write-Host "[KILL] still alive: $($left.ProcessId -join ', ')" -ForegroundColor Red
    } else {
        Write-Host "[KILL] all stopped" -ForegroundColor Green
    }
    $g = nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    Write-Host "[GPU ] $g"
    exit
}

Write-Host "===== RaMRA training monitor  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =====" -ForegroundColor Cyan

if ($procs) {
    Write-Host "[PROC] RUNNING  PID=$(($procs.ProcessId) -join ', ')" -ForegroundColor Green
} else {
    Write-Host "[PROC] not found (finished or not started)" -ForegroundColor Red
}

$gpu = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
if ($gpu) { Write-Host "[GPU ] $gpu" -ForegroundColor Yellow }

$errLog = Join-Path $PSScriptRoot "train_selector_iemocap.err.log"
if (Test-Path $errLog) {
    $bar = (Get-Content $errLog -Raw) -split "`r" | Where-Object { $_ -match 's/it|it/s' } | Select-Object -Last 1
    if ($bar) {
        Write-Host "[PROG] $($bar.Trim())"
    } else {
        Write-Host "[PROG] no progress bar yet (loading)"
    }
}

$logDir = Join-Path $PSScriptRoot "logs"
$logFile = Get-ChildItem $logDir -Filter "train_*.log" -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($logFile) {
    Write-Host "[LOG ] $($logFile.Name)" -ForegroundColor Cyan
    Get-Content $logFile.FullName -Encoding utf8 | Select-String -Pattern "epoch \d+/" | ForEach-Object {
        "       " + $_.Line.Trim()
    }
}

$ckptDirs = Get-ChildItem $PSScriptRoot -Directory | Where-Object { $_.Name -like "saved_models*" }
$ckpts = $ckptDirs | ForEach-Object { Get-ChildItem $_.FullName -Recurse -File -Filter *.ckpt } |
    Sort-Object LastWriteTime -Descending | Select-Object -First 3
if ($ckpts) {
    Write-Host "[CKPT] latest:" -ForegroundColor Cyan
    $ckpts | ForEach-Object { "       {0}  {1:0.00} GB  {2:HH:mm:ss}" -f $_.Name, ($_.Length / 1GB), $_.LastWriteTime }
}

Write-Host ""
Write-Host "stop training -> .\watch_training.ps1 -Kill" -ForegroundColor DarkGray
