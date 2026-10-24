param(
    [Parameter(Mandatory=$true)][string]$RepoRoot,
    [Parameter(Mandatory=$true)][string]$PythonExe,
    [Parameter(Mandatory=$true)][string]$StateRoot,
    [string]$TaskName = 'MemecoinPredictionPaper',
    [ValidateRange(1024,65535)][int]$Port = 18082
)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $env:ProgramData 'MemecoinPaper'))
$taskRepo = [IO.Path]::GetFullPath($RepoRoot)
$taskState = [IO.Path]::GetFullPath($StateRoot)
$taskPython = [IO.Path]::GetFullPath($PythonExe)
foreach ($taskPath in @($taskRepo,$taskState,$taskPython)) {
    if (-not $taskPath.StartsWith($taskRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Dedicated paper paths must stay inside ProgramData\MemecoinPaper'
    }
}
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) { throw 'Task exists; refusing to replace it' }
$taskLauncher = Join-Path $taskRepo 'deploy\windows\start-prediction-paper.ps1'
if (-not (Test-Path -LiteralPath $taskLauncher -PathType Leaf) -or -not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    throw 'Pinned release or dedicated paper Python runtime missing'
}
New-Item -ItemType Directory -Path $taskState -Force | Out-Null
# No Administrator logon persistence, passwords or exported wallet material.
# LocalService can read the code/runtime and modify only its own paper state.
& icacls.exe $taskRoot /inheritance:r /grant:r '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F' '*S-1-5-19:(OI)(CI)RX' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Paper root ACL failed' }
& icacls.exe $taskState /grant:r '*S-1-5-19:(OI)(CI)M' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Paper state ACL failed' }
$taskArgs = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy RemoteSigned -File "{0}" -RepoRoot "{1}" -PythonExe "{2}" -StateRoot "{3}" -Port {4}' -f $taskLauncher,$taskRepo,$taskPython,$taskState,$Port
$taskAction = New-ScheduledTaskAction -Execute (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') -Argument $taskArgs -WorkingDirectory $taskRepo
$taskPrincipal = New-ScheduledTaskPrincipal -UserId 'S-1-5-19' -LogonType ServiceAccount -RunLevel Limited
$taskTrigger = New-ScheduledTaskTrigger -AtStartup
$taskSettings = New-ScheduledTaskSettingsSet -Priority 5 -MultipleInstances IgnoreNew -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable
Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Principal $taskPrincipal -Trigger $taskTrigger -Settings $taskSettings -Description 'Unattended public prediction-market paper research under LocalService; no real orders or wallet keys.' | Out-Null
Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State,@{Name='Account';Expression={$_.Principal.UserId}},@{Name='Logon';Expression={[string]$_.Principal.LogonType}}
