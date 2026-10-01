<#
.SYNOPSIS
    Register (or remove) a daily Windows scheduled task that runs
    scripts/download_av_earnings.py.

.DESCRIPTION
    The task runs as the current user, only while logged on (no password is
    stored), without a console window. If the computer is off or asleep at the
    scheduled time, it runs at the next opportunity. Output is appended to
    data/logs/av_download.log.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\schedule_av_download.ps1
    powershell -ExecutionPolicy Bypass -File scripts\schedule_av_download.ps1 -Time 07:30
    powershell -ExecutionPolicy Bypass -File scripts\schedule_av_download.ps1 -Remove
#>
param(
    [string]$Time = "10:00",
    [string]$Python = "",
    [string]$TaskName = "auto_researcher_av_earnings",
    [switch]$Remove
)

$ErrorActionPreference = "Stop"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "Removed scheduled task '$TaskName'."
    return
}

$Root = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $exe = (Get-Command python -ErrorAction Stop).Source
    $windowless = Join-Path (Split-Path -Parent $exe) "pythonw.exe"
    $Python = if (Test-Path $windowless) { $windowless } else { $exe }
}
$Script = Join-Path $Root "scripts\download_av_earnings.py"
$Log = Join-Path $Root "data\logs\av_download.log"

$action = New-ScheduledTaskAction -Execute $Python `
    -Argument "`"$Script`" --log-file `"$Log`"" -WorkingDirectory $Root
$trigger = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 1) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description "Daily Alpha Vantage earnings download (auto_researcher)" -Force | Out-Null

Write-Output "Registered '$TaskName': daily at $Time with $Python"
Write-Output "Log: $Log"
