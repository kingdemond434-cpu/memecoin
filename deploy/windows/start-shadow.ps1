param(
    [Parameter(Mandatory=$true)][string]$RepoRoot,
    [Parameter(Mandatory=$true)][string]$PythonExe,
    [Parameter(Mandatory=$true)][string]$LogRoot,
    [ValidateRange(1024,65535)][int]$HealthPort = 18081,
    [string]$ConfigPath = ''
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
Set-Location -LiteralPath $taskRepo
# This launcher only collects and simulates. It cannot inherit an armed
# wallet or an isolated signer from the machine's environment.
$env:ALLOW_LIVE_TRADING = ''
$env:SOLANA_PRIVATE_KEY = ''
$env:MEMECOIN_SIGNER_SOCKET = ''
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
& $taskPython @taskCliArguments 1>> $taskOut 2>> $taskErr
exit $LASTEXITCODE
