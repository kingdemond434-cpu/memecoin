param([string]$Root='C:\ProgramData\MemecoinResourceLimits',[switch]$StartPaper)
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$source=[IO.Path]::GetFullPath($PSScriptRoot)
$rootPath=[IO.Path]::GetFullPath($Root)
if(Test-Path -LiteralPath (Join-Path $rootPath 'roles.json')){throw 'Existing CPU manifest requires explicit reviewed update; originals must not be nested or overwritten'}
$roles=[ordered]@{spot=@('MemecoinShadowCodex',500);prediction=@('MemecoinPredictionPaper',100);
 capture=@('MemecoinProgramCapture',150);terminal=@('MemecoinPaperTerminal',50);development=@('MemecoinAutonomousDevelopment',200)}
$backup=Join-Path $rootPath ('backup-'+[DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $backup -Force|Out-Null
& icacls.exe $rootPath /inheritance:r /grant:r 'Administrator:(OI)(CI)F' '*S-1-5-18:(OI)(CI)F' '*S-1-5-19:(OI)(CI)RX'|Out-Null
if($LASTEXITCODE -ne 0){throw 'CPU governor ACL setup failed'}
$states=@();$manifest=[ordered]@{version=1;roles=[ordered]@{}}
foreach($role in $roles.Keys){
 $name=$roles[$role][0];$task=Get-ScheduledTask -TaskName $name
 if($task.Actions.Count -ne 1){throw 'Exactly one existing task action required'}
 $action=$task.Actions[0]
 if($action.Arguments -match 'start-cpu-bounded'){throw 'Task already CPU wrapped'}
 $states+=[ordered]@{task=$name;enabled=$task.Settings.Enabled;user=$task.Principal.UserId;run_level=[string]$task.Principal.RunLevel}
 Export-ScheduledTask -TaskName $name|Set-Content -LiteralPath (Join-Path $backup ($name+'.xml')) -Encoding UTF8
 $execute=[Environment]::ExpandEnvironmentVariables($action.Execute)
 if($execute -eq 'powershell.exe'){$execute=Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'}
 if(-not [IO.Path]::IsPathRooted($execute) -or -not(Test-Path -LiteralPath $execute -PathType Leaf)){throw 'Absolute existing executable required'}
 $manifest.roles[$role]=[ordered]@{task=$name;cpu_rate=$roles[$role][1];execute=$execute;
  arguments=$action.Arguments;working_directory=$action.WorkingDirectory}
}
$states|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $backup 'prior-task-state.json') -Encoding UTF8
foreach($role in $roles.Keys){$name=$roles[$role][0];Disable-ScheduledTask -TaskName $name|Out-Null;Stop-ScheduledTask -TaskName $name}
Copy-Item -LiteralPath (Join-Path $source 'CpuBudgetJob.cs') -Destination (Join-Path $rootPath 'CpuBudgetJob.cs')
Copy-Item -LiteralPath (Join-Path $source 'start-cpu-bounded.ps1') -Destination (Join-Path $rootPath 'start-cpu-bounded.ps1')
[Diagnostics.Process]::GetCurrentProcess().PriorityClass='BelowNormal'
Add-Type -TypeDefinition (Get-Content -LiteralPath (Join-Path $rootPath 'CpuBudgetJob.cs') -Raw) -OutputAssembly (Join-Path $rootPath 'CpuBudgetJob.dll')
$manifest.assembly_sha256=(Get-FileHash -LiteralPath (Join-Path $rootPath 'CpuBudgetJob.dll') -Algorithm SHA256).Hash
$manifest|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $rootPath 'roles.json') -Encoding UTF8
New-Item -ItemType Directory -Path (Join-Path $rootPath 'status') -Force|Out-Null
& icacls.exe (Join-Path $rootPath 'status') /grant:r '*S-1-5-19:(OI)(CI)M'|Out-Null
if($LASTEXITCODE -ne 0){throw 'CPU status ACL setup failed'}
$shell=Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
foreach($role in $roles.Keys){
 $name=$roles[$role][0]
 $args='-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "'+(Join-Path $rootPath 'start-cpu-bounded.ps1')+'" -ManifestPath "'+(Join-Path $rootPath 'roles.json')+'" -Role '+$role
 $action=New-ScheduledTaskAction -Execute $shell -Argument $args -WorkingDirectory $rootPath
 $task=Get-ScheduledTask -TaskName $name;$settings=$task.Settings;$settings.Priority=7
 Set-ScheduledTask -TaskName $name -Action $action -Settings $settings|Out-Null
 Disable-ScheduledTask -TaskName $name|Out-Null
 $after=Get-ScheduledTask -TaskName $name;$before=$states|Where-Object task -eq $name
 if($after.Principal.UserId -ne $before.user -or [string]$after.Principal.RunLevel -ne $before.run_level){throw 'Task principal changed'}
}
if($StartPaper){foreach($role in @('spot','prediction','capture','terminal')){$name=$roles[$role][0];Enable-ScheduledTask -TaskName $name|Out-Null;Start-ScheduledTask -TaskName $name}}
$receipt=[ordered]@{status='CPU_BUDGETS_INSTALLED';root=$rootPath;total_percent=10;
 per_role_percent=[ordered]@{spot=5;prediction=1;capture=1.5;terminal=0.5;development=2};
 paper_start_requested=[bool]$StartPaper;development_enabled=$false;quant_tasks_changed=$false;principals_changed=$false;backup=$backup}
$receipt|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $rootPath 'installation.json') -Encoding UTF8
$receipt|ConvertTo-Json -Depth 6 -Compress
