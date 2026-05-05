param(
    [string]$StartFrom = "trend",
    [string]$LogDir = "doc/training_more_30-4-26"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv/Scripts/python.exe"
$sweep = Join-Path $root "tools/run_full_phase4_sweep.py"

if (-not (Test-Path $python)) {
    throw "python not found: $python"
}
if (-not (Test-Path $sweep)) {
    throw "sweep script not found: $sweep"
}

New-Item -ItemType Directory -Force -Path (Join-Path $root $LogDir) | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $root "$LogDir/sweep_log_${stamp}_keepawake.txt"
$errFile = Join-Path $root "$LogDir/sweep_log_${stamp}_keepawake.err.txt"

Write-Host "[keepawake-run] start_from=$StartFrom" -ForegroundColor Green
Write-Host "[keepawake-run] log=$logFile" -ForegroundColor Green
Write-Host "[keepawake-run] err=$errFile" -ForegroundColor Green

$proc = Start-Process -FilePath $python -ArgumentList @($sweep, "--start-from", $StartFrom) -PassThru -NoNewWindow -RedirectStandardOutput $logFile -RedirectStandardError $errFile

& (Join-Path $root "tools/keepawake_guardian.ps1") -TargetPid $proc.Id
$proc.WaitForExit()
$proc.Refresh()
$exitCode = if ($null -ne $proc.ExitCode) { [int]$proc.ExitCode } else { 0 }

Write-Host "[keepawake-run] exit_code=$exitCode" -ForegroundColor Green
exit $exitCode
