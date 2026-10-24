param(
    [Parameter(Mandatory=$true)][string]$RepoRoot,
    [Parameter(Mandatory=$true)][string]$PythonExe,
    [Parameter(Mandatory=$true)][string]$StateRoot,
    [ValidateRange(1024,65535)][int]$Port = 18082,
    [string]$ConnectionFile = '',
    [string]$RpcBudgetPath = '',
    [ValidateSet('phantom','legacy-paper')][string]$PredictionScope = 'phantom'
)
$ErrorActionPreference = 'Stop'
$taskRepo = [IO.Path]::GetFullPath($RepoRoot)
$taskState = [IO.Path]::GetFullPath($StateRoot)
$taskPython = [IO.Path]::GetFullPath($PythonExe)
if (-not (Test-Path -LiteralPath (Join-Path $taskRepo 'src\runtime\market_gateway.py'))) { throw 'Paper release missing' }
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) { throw 'Python runtime missing' }
New-Item -ItemType Directory -Path $taskState -Force | Out-Null
Add-Type -Path (Join-Path $PSScriptRoot 'ShadowProcessJob.cs')
[MemecoinShadowProcessJob]::AttachCurrentProcess()
if($ConnectionFile){
    . (Join-Path $PSScriptRoot 'ReadBotConnections.ps1')
    Import-BotConnections -Path $ConnectionFile
}
if($RpcBudgetPath){$env:MEMECOIN_RPC_BUDGET_DB=[IO.Path]::GetFullPath($RpcBudgetPath)}
Set-Location -LiteralPath $taskRepo
$env:ALLOW_LIVE_TRADING = ''
$env:SOLANA_PRIVATE_KEY = ''
$env:MEMECOIN_SIGNER_SOCKET = ''
$env:MEMECOIN_WALLET_MODE = ''
$env:MEMECOIN_LOCAL_DECISION_DIR = Join-Path $taskState 'local_decisions'
$env:PYTHONUNBUFFERED = '1'
$taskLog = Join-Path $taskState ('paper-' + [DateTime]::UtcNow.ToString('yyyy-MM-dd') + '.log')
$ErrorActionPreference = 'Continue'
& $taskPython -m src.runtime.market_gateway --state-dir $taskState --port $Port --prediction-scope $PredictionScope *>> $taskLog
exit $LASTEXITCODE
