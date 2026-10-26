param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$Release,
    [Parameter(Mandatory=$true)][string]$LogDirectory
)
$ErrorActionPreference = 'Stop'
# This process needs no provider credentials or signing context. Do not import connections.env.
Get-ChildItem Env: | Where-Object {
    $_.Name -match '(KEY|SECRET|TOKEN|PASSWORD|SIGNER|WALLET|MNEMONIC|RPC_URL|LIVE_TRAD|PROXY)'
} | ForEach-Object { Remove-Item -LiteralPath ('Env:' + $_.Name) }
Set-Location -LiteralPath $Release
if (-not (Test-Path -LiteralPath $LogDirectory -PathType Container)) {
    throw 'Pre-created private log directory required'
}
& $Python -m src.terminal.server --port 18083 1>> (Join-Path $LogDirectory 'terminal.stdout.log') 2>> (Join-Path $LogDirectory 'terminal.stderr.log')
exit $LASTEXITCODE
