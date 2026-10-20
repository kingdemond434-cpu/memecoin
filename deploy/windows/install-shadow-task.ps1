param(
    [Parameter(Mandatory=$true)][string]$RepoRoot,
    [Parameter(Mandatory=$true)][string]$PythonExe,
    [Parameter(Mandatory=$true)][string]$LogRoot,
    [string]$TaskName = 'MemecoinShadowCodex',
    [ValidateRange(1024,65535)][int]$HealthPort = 18081
)
$ErrorActionPreference = 'Stop'
$taskRepo = [IO.Path]::GetFullPath($RepoRoot)
$taskPython = [IO.Path]::GetFullPath($PythonExe)
$taskLogs = [IO.Path]::GetFullPath($LogRoot)
$taskLauncher = Join-Path $taskRepo 'deploy\windows\start-shadow.ps1'
if (-not (Test-Path -LiteralPath $taskLauncher -PathType Leaf) -or
    -not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    throw 'Launcher or staging interpreter is missing'
}
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    throw 'Task already exists; refusing to replace an existing deployment'
}
$taskUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$taskArguments = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy RemoteSigned -File "{0}" -RepoRoot "{1}" -PythonExe "{2}" -LogRoot "{3}" -HealthPort {4}' -f $taskLauncher,$taskRepo,$taskPython,$taskLogs,$HealthPort
$taskAction = New-ScheduledTaskAction -Execute (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') -Argument $taskArguments -WorkingDirectory $taskRepo
$taskTrigger = New-ScheduledTaskTrigger -AtLogOn -User $taskUser
$taskPrincipal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
$taskSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable
Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $taskTrigger -Principal $taskPrincipal -Settings $taskSettings -Description 'Isolated memecoin public observation and paper trading; live submission disabled.' | Out-Null
Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
