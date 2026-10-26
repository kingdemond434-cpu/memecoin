param(
    [Parameter(Mandatory=$true)][string]$RepoRoot,
    [Parameter(Mandatory=$true)][string]$PythonExe,
    [Parameter(Mandatory=$true)][string]$StateRoot,
    [Parameter(Mandatory=$true)][string]$ConfigPath,
    [string]$ConnectionFile='',
    [string]$RpcBudgetPath='',
    [ValidateRange(1024,65535)][int]$Port=18084
)
$ErrorActionPreference='Stop'
$taskRepo=[IO.Path]::GetFullPath($RepoRoot)
$taskPython=[IO.Path]::GetFullPath($PythonExe)
$taskState=[IO.Path]::GetFullPath($StateRoot)
$taskConfig=[IO.Path]::GetFullPath($ConfigPath)
if(-not(Test-Path -LiteralPath (Join-Path $taskRepo 'src/runtime/program_capture.py')) -or
   -not(Test-Path -LiteralPath $taskPython -PathType Leaf) -or
   -not(Test-Path -LiteralPath $taskConfig -PathType Leaf)){throw 'Capture source, interpreter or configuration missing'}
New-Item -ItemType Directory -Path $taskState -Force | Out-Null
Add-Type -Path (Join-Path $PSScriptRoot 'ShadowProcessJob.cs')
[MemecoinShadowProcessJob]::AttachCurrentProcess()
if($ConnectionFile){
 . (Join-Path $PSScriptRoot 'ReadBotConnections.ps1')
 Import-BotConnections -Path $ConnectionFile
}
if($RpcBudgetPath){$env:MEMECOIN_RPC_BUDGET_DB=[IO.Path]::GetFullPath($RpcBudgetPath)}
$env:ALLOW_LIVE_TRADING=''
$env:SOLANA_PRIVATE_KEY=''
$env:MEMECOIN_SIGNER_SOCKET=''
$env:MEMECOIN_WALLET_MODE=''
$env:PYTHONUNBUFFERED='1'
Set-Location -LiteralPath $taskRepo
$taskOut=Join-Path $taskState ('capture-'+[DateTime]::UtcNow.ToString('yyyy-MM-dd')+'-stdout.log')
$taskErr=Join-Path $taskState ('capture-'+[DateTime]::UtcNow.ToString('yyyy-MM-dd')+'-stderr.log')
$ErrorActionPreference='Continue'
& $taskPython -m src.runtime.program_capture --config $taskConfig --state-dir $taskState --port $Port 1>> $taskOut 2>> $taskErr
exit $LASTEXITCODE
