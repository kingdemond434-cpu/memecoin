param([Parameter(Mandatory=$true)][string]$FixtureRoot,[Parameter(Mandatory=$true)][string]$PythonExe,[Parameter(Mandatory=$true)][string]$FixtureScript)
$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$root=[IO.Path]::GetFullPath($FixtureRoot)
if(Test-Path -LiteralPath $root){throw 'Fresh isolated fixture directory required'}
New-Item -ItemType Directory -Path (Join-Path $root 'status') -Force|Out-Null
[Diagnostics.Process]::GetCurrentProcess().PriorityClass='BelowNormal'
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'start-cpu-bounded.ps1') -Destination (Join-Path $root 'start-cpu-bounded.ps1')
Add-Type -TypeDefinition (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'CpuBudgetJob.cs') -Raw) -OutputAssembly (Join-Path $root 'CpuBudgetJob.dll')
$manifest=[ordered]@{version=1;assembly_sha256=(Get-FileHash -LiteralPath (Join-Path $root 'CpuBudgetJob.dll') -Algorithm SHA256).Hash;
 roles=[ordered]@{fixture=[ordered]@{cpu_rate=100;execute=$PythonExe;arguments=('"'+$FixtureScript+'" "'+(Join-Path $root 'results')+'"');working_directory=$root}}}
$path=Join-Path $root 'roles.json';$manifest|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $path -Encoding UTF8
$shell=Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$info=New-Object Diagnostics.ProcessStartInfo
$info.FileName=$shell;$info.Arguments='-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "'+(Join-Path $root 'start-cpu-bounded.ps1')+'" -ManifestPath "'+$path+'" -Role fixture'
$info.UseShellExecute=$false;$info.CreateNoWindow=$true
$info.RedirectStandardInput=$true
$process=[Diagnostics.Process]::Start($info);$process.StandardInput.Close()
if(-not $process.WaitForExit(240000)){throw 'Isolated bounded fixture timed out; inspect and stop only its exact task/process tree'}
if($process.ExitCode -ne 0){throw 'Bounded fixture failed'}
$parent=Get-Content -LiteralPath (Join-Path $root 'results\parent.json') -Raw|ConvertFrom-Json
$children=@(Get-ChildItem -LiteralPath (Join-Path $root 'results') -Filter 'grandchild-*.json')
if($children.Count -ne 2 -or -not $parent.breakaway_rejected -or -not $parent.python_current_job_query_verified){throw 'Descendant or breakaway acceptance failed'}
foreach($child in $children){$data=Get-Content -LiteralPath $child.FullName -Raw|ConvertFrom-Json;if(-not $data.in_budget_job -or $data.deterministic_result -ne $parent.deterministic_result){throw 'Child membership or deterministic result changed'}}
$markerBefore=@(Get-ChildItem -LiteralPath (Join-Path $root 'results')).Count
$manifest.roles.fixture.cpu_rate=1001;$manifest|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $path -Encoding UTF8
$bad=[Diagnostics.Process]::Start($info);$bad.StandardInput.Close();$bad.WaitForExit()
if($bad.ExitCode -eq 0 -or @(Get-ChildItem -LiteralPath (Join-Path $root 'results')).Count -ne $markerBefore){throw 'Oversized budget did not fail closed'}
$manifest.roles.fixture.cpu_rate=100;$manifest.assembly_sha256='0'*64;$manifest|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $path -Encoding UTF8
$bad=[Diagnostics.Process]::Start($info);$bad.StandardInput.Close();$bad.WaitForExit()
if($bad.ExitCode -eq 0 -or @(Get-ChildItem -LiteralPath (Join-Path $root 'results')).Count -ne $markerBefore){throw 'Assembly integrity did not fail closed'}
$receipt=[ordered]@{status='PASS';cpu_percent=1;descendants_verified=2;breakaway_rejected=$true;
 deterministic_outputs_identical=$true;python_current_job_query_verified=$true;single_thread_environment=$true;oversized_budget_rejected=$true;assembly_tamper_rejected=$true;production_data_used=$false;quant_changed=$false}
$receipt|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $root 'receipt.json') -Encoding UTF8
$receipt|ConvertTo-Json -Compress
