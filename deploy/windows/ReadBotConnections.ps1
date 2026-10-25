function Import-BotConnections {
    param([Parameter(Mandatory=$true)][string]$Path)
    $taskAllowed=@('HELIUS_API_KEY','YELLOWSTONE_GRPC_URL','YELLOWSTONE_GRPC_TOKEN','YELLOWSTONE_X_TOKEN',
                  'MEMECOIN_RPC_RPS','MEMECOIN_RPC_DAILY_UNITS','MEMECOIN_RPC_MONTHLY_UNITS','GMGN_API_KEY')
    $taskFile=[IO.Path]::GetFullPath($Path)
    if(-not(Test-Path -LiteralPath $taskFile -PathType Leaf)){throw 'Private connection file missing'}
    if((Get-Item -LiteralPath $taskFile).Length -gt 16384){throw 'Private connection file exceeds size bound'}
    $taskSeen=@{}
    foreach($taskLine in [IO.File]::ReadAllLines($taskFile)){
        $taskText=$taskLine.Trim()
        if(-not $taskText -or $taskText.StartsWith('#')){continue}
        if($taskText -notmatch '^([A-Z][A-Z0-9_]*)=(.*)$'){throw 'Malformed connection declaration'}
        $taskName=$Matches[1];$taskValue=$Matches[2].Trim()
        $taskResearchRpc=$taskName -match '^SOLANA_RESEARCH_(RPC|WS)_URL(_[1-9][0-9]*)?$'
        if(($taskName -notin $taskAllowed -and -not $taskResearchRpc) -or $taskSeen.ContainsKey($taskName)){throw 'Unsupported or duplicate connection field'}
        $taskSeen[$taskName]=$true
        if($taskValue.Length -gt 4096 -or $taskValue.Contains([char]0)){throw 'Invalid connection value'}
        if($taskName -eq 'HELIUS_API_KEY' -and $taskValue -and $taskValue -notmatch '^[A-Za-z0-9_.-]{10,256}$'){throw 'Invalid RPC credential format'}
        if($taskResearchRpc -and $taskValue){
            $taskUri=$null
            if(-not [Uri]::TryCreate($taskValue,[UriKind]::Absolute,[ref]$taskUri)){throw 'Invalid research RPC endpoint'}
            $taskSchemes=if($taskName -match '_WS_'){@('wss')}else{@('https')}
            if($taskUri.IsLoopback){$taskSchemes+=if($taskName -match '_WS_'){'ws'}else{'http'}}
            if($taskUri.Scheme -notin $taskSchemes -or -not $taskUri.Host -or $taskUri.Fragment){throw 'Invalid research RPC endpoint'}
        }
        # Assign literal data; never dot-source or evaluate credential text.
        if($taskValue){[Environment]::SetEnvironmentVariable($taskName,$taskValue,'Process')}
    }
    if($env:YELLOWSTONE_X_TOKEN -and -not $env:YELLOWSTONE_GRPC_TOKEN){$env:YELLOWSTONE_GRPC_TOKEN=$env:YELLOWSTONE_X_TOKEN}
}
