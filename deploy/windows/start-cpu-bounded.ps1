param(
 [Parameter(Mandatory=$true)][string]$ManifestPath,
 [Parameter(Mandatory=$true)][ValidatePattern('^[a-z]+$')][string]$Role
)
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$root=Split-Path -Parent ([IO.Path]::GetFullPath($ManifestPath))
$manifest=Get-Content -LiteralPath $ManifestPath -Raw|ConvertFrom-Json
$rates=@($manifest.roles.psobject.Properties|ForEach-Object {$_.Value.cpu_rate})
if($manifest.version -ne 1 -or $rates.Count -lt 1 -or
   @($rates|Where-Object {$_ -isnot [int] -and $_ -isnot [long]}).Count -gt 0 -or
   @($rates|Where-Object {$_ -lt 1 -or $_ -gt 1000}).Count -gt 0 -or
   ($rates|Measure-Object -Sum).Sum -gt 1000){throw 'CPU budget manifest invalid or exceeds ten percent'}
$entry=$manifest.roles.psobject.Properties[$Role]
if(-not $entry){throw 'CPU role missing'}
$entry=$entry.Value
$dll=Join-Path $root 'CpuBudgetJob.dll'
if((Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash -ne $manifest.assembly_sha256){throw 'CPU governor assembly hash mismatch'}
$statusPath=Join-Path $root ('status\'+$Role+'.json')
function Save-CpuStatus($state,$snapshot,$childId,$code){
 $row=[ordered]@{observed_at=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds();role=$Role;state=$state;
  launcher_pid=$PID;child_pid=$childId;cpu_rate=$entry.cpu_rate;cpu_percent=([double]$entry.cpu_rate/100);
  total_configured_percent=(($rates|Measure-Object -Sum).Sum/100);exit_code=$code;
  control_flags=$null;priority_class=$null;process_ids=@();live_authority_changed=$false}
 if($snapshot){$row.control_flags=$snapshot.ControlFlags;$row.priority_class=$snapshot.PriorityClass;$row.process_ids=@($snapshot.ProcessIds)}
 $temporary=$statusPath+'.tmp';$row|ConvertTo-Json -Depth 6|Set-Content -LiteralPath $temporary -Encoding UTF8
 Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}
try {
 Add-Type -Path $dll
 [MemecoinCpuBudgetJob]::Attach(('Local\MemecoinCPU-'+$Role),[uint32]$entry.cpu_rate)
 $snapshot=[MemecoinCpuBudgetJob]::Current()
 if($snapshot.ControlFlags -ne 5 -or $snapshot.CpuRate -ne $entry.cpu_rate -or $snapshot.PriorityClass -ne 16384){throw 'Kernel CPU cap verification failed'}
 foreach($name in @('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','RAYON_NUM_THREADS','VECLIB_MAXIMUM_THREADS')){
  [Environment]::SetEnvironmentVariable($name,'1','Process')
 }
 $env:TOKENIZERS_PARALLELISM='false'
 $info=New-Object Diagnostics.ProcessStartInfo
 $info.FileName=[Environment]::ExpandEnvironmentVariables([string]$entry.execute)
 $info.Arguments=[string]$entry.arguments
 $info.WorkingDirectory=[string]$entry.working_directory
 $info.UseShellExecute=$false;$info.CreateNoWindow=$true
 $info.RedirectStandardInput=$true
 $child=[Diagnostics.Process]::Start($info)
 $child.StandardInput.Close()
 $snapshot=[MemecoinCpuBudgetJob]::Current()
 if(@($snapshot.ProcessIds) -notcontains [long]$child.Id -and -not $child.HasExited){throw 'Child escaped CPU job'}
 Save-CpuStatus 'BOUNDED' $snapshot $child.Id $null
 while(-not $child.WaitForExit(5000)){
  Save-CpuStatus 'BOUNDED' ([MemecoinCpuBudgetJob]::Current()) $child.Id $null
 }
 $code=$child.ExitCode
 while(([MemecoinCpuBudgetJob]::Current()).ActiveProcesses -gt 1){Start-Sleep -Seconds 2}
 Save-CpuStatus 'EXITED' ([MemecoinCpuBudgetJob]::Current()) $child.Id $code
 [MemecoinCpuBudgetJob]::ReleaseAfterChildrenExit()
 exit $code
}catch{
 # Never print original task arguments/configuration or launch an uncapped fallback.
 try {Save-CpuStatus ('FAILED_'+$_.Exception.GetType().Name) $null $null 1}catch{}
 exit 1
}
