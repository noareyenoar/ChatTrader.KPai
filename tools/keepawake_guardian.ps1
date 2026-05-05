param(
    [Parameter(Mandatory = $true)]
    [int]$TargetPid
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

namespace Win32 {
    public static class Power {
        [DllImport("kernel32.dll")]
        public static extern uint SetThreadExecutionState(uint esFlags);
    }
}
"@

# ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
$keepAwakeFlags = [Convert]::ToUInt32("80000041", 16)
[Win32.Power]::SetThreadExecutionState($keepAwakeFlags) | Out-Null
Write-Host "[keepawake] active for pid=$TargetPid" -ForegroundColor Cyan

try {
    if (Get-Process -Id $TargetPid -ErrorAction SilentlyContinue) {
        Wait-Process -Id $TargetPid
    }
    else {
        Write-Host "[keepawake] target pid already exited: $TargetPid" -ForegroundColor Yellow
    }
}
finally {
    # Clear sticky execution-state flags
    [Win32.Power]::SetThreadExecutionState([Convert]::ToUInt32("80000000", 16)) | Out-Null
    Write-Host "[keepawake] released for pid=$TargetPid" -ForegroundColor Cyan
}
