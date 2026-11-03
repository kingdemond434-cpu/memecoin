# Read-only compact health summary for the hourly pass (quant item 39).
# Starts, stops and changes nothing. Writes one JSON file (default stdout).
# Every field is observed; unknowns are null, never zero.
param([string]$Out='',[string]$Root='C:\ProgramData\MemecoinResourceLimits')
$ErrorActionPreference='Continue';$ProgressPreference='SilentlyContinue'
[Diagnostics.Process]::GetCurrentProcess().PriorityClass='BelowNormal'
$now=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$roles=[ordered]@{spot=@('MemecoinShadowCodex',18081);prediction=@('MemecoinPredictionPaper',18082);capture=@('MemecoinProgramCapture',18084);terminal=@('MemecoinPaperTerminal',18083);development=@('MemecoinAutonomousDevelopment',0)}
function Get-Health($port){
 if(-not $port){return $null}
 $raw=& curl.exe -s --noproxy '*' -m 10 ("http://127.0.0.1:$port/health") 2>$null
 if($LASTEXITCODE -ne 0 -or -not $raw){return @{ok=$false;error=('curl_exit_'+$LASTEXITCODE)}}
 try{$j=($raw -join "`n")|ConvertFrom-Json}catch{return @{ok=$false;error='bad_json'}}
 $h=[ordered]@{ok=$true;status=$j.status;uptime_s=$j.uptime_seconds;dry_run=$j.dry_run;live_locked=$j.live_submission_locked}
 if($j.portfolio){$h.paper=[ordered]@{total_pnl=$j.portfolio.total_pnl;trade_count=$j.portfolio.trade_count;successful_exits=$j.portfolio.successful_exits}}
 $mesh=$j.source_mesh;if(-not $mesh -and $j.subsystems){$mesh=$j.subsystems.source_mesh}
 if($mesh){$ages=@($mesh.oldest_pending_age_by_source.PSObject.Properties|ForEach-Object{[double]$_.Value})
  $h.queue=[ordered]@{pending=$mesh.pending_events;due=$mesh.due_events;acked=$mesh.canonical_acknowledged;accepted=$mesh.accepted_observations;
   leased_this_run=$mesh.leased_events_this_run;oldest_pending_age_s=$(if($ages.Count){($ages|Measure-Object -Maximum).Maximum}else{$null})}}
 return $h}
$tasks=[ordered]@{};$red=@()
foreach($r in $roles.Keys){
 $name=$roles[$r][0];$t=Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
 $row=[ordered]@{task=$name;state=$null;last_result=$null;last_run=$null;cpu_state=$null;cpu_percent=$null;status_age_s=$null;status_write_errors=0;health=$null}
 if($t){$i=$t|Get-ScheduledTaskInfo;$row.state=[string]$t.State;$row.last_result=('0x{0:X}' -f $i.LastTaskResult);$row.last_run=$i.LastRunTime.ToUniversalTime().ToString('o')}else{$row.state='MISSING'}
 $sp=Join-Path $Root ('status\'+$r+'.json')
 if(Test-Path -LiteralPath $sp){try{$s=Get-Content -LiteralPath $sp -Raw|ConvertFrom-Json;$row.cpu_state=$s.state;$row.cpu_percent=$s.cpu_percent;$row.status_age_s=$now-[long]$s.observed_at}catch{}}
 if(Test-Path -LiteralPath ($sp+'.errors')){$row.status_write_errors=@(Get-Content -LiteralPath ($sp+'.errors')).Count}
 $row.health=Get-Health $roles[$r][1]
 if($r -ne 'development'){
  if($row.cpu_state -ne 'BOUNDED'){$red+=($r+': cpu_state '+$row.cpu_state)}
  elseif($row.status_age_s -gt 120){$red+=($r+': status stale '+$row.status_age_s+'s')}
  if($row.health -and -not $row.health.ok){$red+=($r+': health '+$row.health.error)}
  elseif($row.health -and @('ok','healthy') -notcontains ([string]$row.health.status).ToLower()){$red+=($r+': health status '+$row.health.status)}
  if(@('Ready','Running') -notcontains $row.state){$red+=($r+': task '+$row.state)}
  elseif(@('0x0','0x41301','0x41303','0x800710E0') -notcontains $row.last_result){$red+=($r+': last_result '+$row.last_result)}
  if($row.status_write_errors -gt 0){$red+=($r+': status_write_errors '+$row.status_write_errors)}}
 $tasks[$r]=$row}
$summary=[ordered]@{schema='memecoin_health/1';observed_at=$now;red=$red;verdict=$(if($red.Count){'RED'}else{'GREEN'});roles=$tasks;
 cpu_budget_percent=[ordered]@{spot=5;prediction=1;capture=1.5;terminal=0.5;development=2};
 not_yet_measured=@('discovery_latency_p50_p95','coverage_gaps','paper_pnl_7d')}
$json=$summary|ConvertTo-Json -Depth 8 -Compress
if($Out){$tmp=$Out+'.tmp';[IO.File]::WriteAllText($tmp,$json);Move-Item -LiteralPath $tmp -Destination $Out -Force}else{$json}
