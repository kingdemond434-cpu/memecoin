param(
    [string]$VpsHost = '62.171.172.249',
    [string]$VpsUser = 'administrator',
    [string]$IdentityFile = (Join-Path $env:USERPROFILE '.ssh\id_ed25519_quantbox'),
    [string]$RemoteState = 'C:/ProgramData/MemecoinPaper/state-v2'
)
$ErrorActionPreference = 'Stop'
if ($VpsHost -notmatch '^[a-zA-Z0-9.-]{1,253}$' -or $VpsUser -notmatch '^[a-zA-Z0-9_-]{1,64}$' -or
    $RemoteState -notmatch '^C:/ProgramData/MemecoinPaper/state-v[0-9]+$') { throw 'Unexpected tunnel target' }
if (-not (Test-Path -LiteralPath $IdentityFile -PathType Leaf)) { throw 'Configured SSH identity is missing' }
$taskSsh = (Get-Command ssh.exe -ErrorAction Stop).Source
$taskScp = (Get-Command scp.exe -ErrorAction Stop).Source
$taskLocal = Join-Path $env:LOCALAPPDATA 'MemecoinPaper'
New-Item -ItemType Directory -Path $taskLocal -Force | Out-Null
$taskUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
& icacls.exe $taskLocal /inheritance:r /grant:r ($taskUser + ':(OI)(CI)F') '*S-1-5-18:(OI)(CI)F' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Cannot secure local pairing-token directory' }
$taskTokenPath = Join-Path $taskLocal 'gateway.token'
& $taskScp -q -i $IdentityFile -o BatchMode=yes ($VpsUser + '@' + $VpsHost + ':' + $RemoteState + '/gateway.token') $taskTokenPath
if ($LASTEXITCODE -ne 0) { throw 'Cannot retrieve gateway pairing token over authenticated SSH' }
$taskToken = [IO.File]::ReadAllText($taskTokenPath).Trim()
if ($taskToken -notmatch '^[A-Za-z0-9_-]{40,100}$') { throw 'Invalid gateway pairing token' }
$taskListeners = @(Get-NetTCPConnection -LocalPort 18082 -State Listen -ErrorAction SilentlyContinue)
if ($taskListeners.Count) {
    $taskOwned = Get-Content -LiteralPath (Join-Path $taskLocal 'tunnel-pid.txt') -ErrorAction SilentlyContinue
    if (-not $taskOwned -or $taskListeners[0].OwningProcess -ne [int]$taskOwned) { throw 'Local port 18082 belongs to another process' }
    $taskProcess = Get-Process -Id ([int]$taskOwned) -ErrorAction Stop
    if ($taskProcess.Path -ne $taskSsh) { throw 'Saved tunnel PID no longer belongs to SSH' }
} else {
    $taskArgs = @('-i', ('"' + [IO.Path]::GetFullPath($IdentityFile) + '"'), '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
                  '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', '-N', '-L', '127.0.0.1:18082:127.0.0.1:18082', ($VpsUser + '@' + $VpsHost))
    $taskTunnel = Start-Process -FilePath $taskSsh -ArgumentList $taskArgs -WindowStyle Hidden -PassThru
    [IO.File]::WriteAllText((Join-Path $taskLocal 'tunnel-pid.txt'), [string]$taskTunnel.Id)
    for ($taskAttempt = 0; $taskAttempt -lt 15; $taskAttempt++) {
        if ($taskTunnel.HasExited) { throw 'SSH tunnel exited before opening the operator page' }
        if (Get-NetTCPConnection -LocalPort 18082 -State Listen -ErrorAction SilentlyContinue) { break }
        Start-Sleep -Milliseconds 500
    }
}
# The fragment is removed by the page after pairing; do not print this URL.
Start-Process -FilePath ('http://127.0.0.1:18082/#pair=' + [uri]::EscapeDataString($taskToken))
Write-Output 'Opened the private Phantom operator page. Connect and approve inside your own wallet.'
