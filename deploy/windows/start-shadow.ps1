param(
    [Parameter(Mandatory=$true)][string]$RepoRoot,
    [Parameter(Mandatory=$true)][string]$PythonExe,
    [Parameter(Mandatory=$true)][string]$LogRoot,
    [ValidateRange(1024,65535)][int]$HealthPort = 18081,
    [string]$ConfigPath = '',
    [string]$ConnectionFile = '',
    [string]$RpcBudgetPath = '',
    [string]$TorExe = '',
    [string]$TorConfig = '',
    [string]$TorExeSha256 = '',
    [string]$TorProxyUrl = 'socks5://127.0.0.1:9152'
)
$ErrorActionPreference = 'Stop'
$taskRepo = [IO.Path]::GetFullPath($RepoRoot)
$taskPython = [IO.Path]::GetFullPath($PythonExe)
$taskLogs = [IO.Path]::GetFullPath($LogRoot)
if (-not (Test-Path -LiteralPath (Join-Path $taskRepo 'src\main.py'))) {
    throw 'Repository entry point is missing'
}
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    throw 'Staging Python interpreter is missing'
}
New-Item -ItemType Directory -Path $taskLogs -Force | Out-Null
Add-Type -Path (Join-Path $PSScriptRoot 'ShadowProcessJob.cs')
[MemecoinShadowProcessJob]::AttachCurrentProcess()
if($ConnectionFile){
    . (Join-Path $PSScriptRoot 'ReadBotConnections.ps1')
    Import-BotConnections -Path $ConnectionFile
}
if($RpcBudgetPath){$env:MEMECOIN_RPC_BUDGET_DB=[IO.Path]::GetFullPath($RpcBudgetPath)}
Set-Location -LiteralPath $taskRepo
# This launcher only collects and simulates. It cannot inherit an armed
# wallet or an isolated signer from the machine's environment.
$env:ALLOW_LIVE_TRADING = ''
$env:SOLANA_PRIVATE_KEY = ''
$env:MEMECOIN_SIGNER_SOCKET = ''
$env:MEMECOIN_WALLET_MODE = ''
$env:HEALTH_HOST = '127.0.0.1'
$env:HEALTH_PORT = [string]$HealthPort
$env:PYTHONUNBUFFERED = '1'
$env:LOG_LEVEL = 'WARNING'
$taskDate = Get-Date -Format 'yyyy-MM-dd'
$taskOut = Join-Path $taskLogs ($taskDate + '-stdout.log')
$taskErr = Join-Path $taskLogs ($taskDate + '-stderr.log')
$taskCliArguments = @('-m', 'src.main', '--dry-run')
if ($ConfigPath) {
    $taskConfig = [IO.Path]::GetFullPath($ConfigPath)
    if (-not (Test-Path -LiteralPath $taskConfig -PathType Leaf)) { throw 'Explicit deployment configuration is missing' }
    $taskCliArguments += @('--config', $taskConfig)
}
$ErrorActionPreference = 'Continue'
$taskTorProcess = $null
$taskExitCode = 1
try {
    if ($TorExe -or $TorConfig) {
        $ErrorActionPreference = 'Stop'
        if (-not $TorExe -or -not $TorConfig -or $TorExeSha256 -notmatch '^[0-9a-fA-F]{64}$') {
            throw 'Optional Tor client requires executable, config and verified executable SHA256'
        }
        $taskTorExe = [IO.Path]::GetFullPath($TorExe)
        $taskTorConfig = [IO.Path]::GetFullPath($TorConfig)
        if (-not (Test-Path -LiteralPath $taskTorExe -PathType Leaf) -or
            -not (Test-Path -LiteralPath $taskTorConfig -PathType Leaf)) { throw 'Tor client files missing' }
        if ((Get-FileHash -LiteralPath $taskTorExe -Algorithm SHA256).Hash -ne $TorExeSha256) {
            throw 'Tor executable differs from verified deployment hash'
        }
        if ($TorProxyUrl -notmatch '^socks5://127\.0\.0\.1:[0-9]{2,5}$') {
            throw 'Managed Windows Tor proxy must bind to localhost'
        }
        $taskTorProcess = Start-Process -FilePath $taskTorExe -ArgumentList @('-f', ('"' + $taskTorConfig + '"')) -WindowStyle Hidden -PassThru
        $env:MEMECOIN_TOR_PROXY_URL = $TorProxyUrl
        $ErrorActionPreference = 'Continue'
    }
    & $taskPython @taskCliArguments 1>> $taskOut 2>> $taskErr
    $taskExitCode = $LASTEXITCODE
} finally {
    # The job object also owns this child if Task Scheduler ends the launcher.
    # A normal stop closes only the Tor process created by this invocation.
    if ($taskTorProcess -and -not $taskTorProcess.HasExited) {
        Stop-Process -Id $taskTorProcess.Id -ErrorAction SilentlyContinue
    }
}
exit $taskExitCode
