# Windows paper deployment

Use a dedicated checkout and virtual environment. The launcher clears the live acknowledgement, wallet key and signer socket, passes `--dry-run`, and binds health/status to loopback. It does not grant a promotion stage.

An explicit `-ConfigPath` on the installer or launcher selects a dedicated
deployment configuration. This permits bounded research/native observation
declarations without editing the tracked default configuration on the server.
It does not change the interactive task account or its authority.

`install-shadow-task.ps1` registers a new task for the current Windows user's interactive session and starts it. It resumes at that user's next logon, allows one instance and retries failures three times. It requires no password in a script. It refuses to overwrite an existing task. A signed-out user has no interactive session, so this is not an unattended boot service.

```powershell
& .\deploy\windows\install-shadow-task.ps1 `
  -RepoRoot 'C:\path\to\repo' `
  -PythonExe 'C:\path\to\.venv\Scripts\python.exe' `
  -LogRoot 'C:\path\to\logs'
```

The launcher places itself and its descendants in a Windows job with kill-on-close, so stopping the task also ends its Python children. It refuses to launch if that containment cannot be established. [Windows job objects](https://learn.microsoft.com/windows/win32/procthread/job-objects) document this lifetime behavior.

Inspect `http://127.0.0.1:18081/status` from the server, the named log directory and `Get-ScheduledTaskInfo MemecoinShadowCodex`. Close the bot gracefully when possible so ledgers and sockets flush. To stop automatic restarts and interrupt this deployment, disable and stop only its named task:

```powershell
Disable-ScheduledTask -TaskName MemecoinShadowCodex
Stop-ScheduledTask -TaskName MemecoinShadowCodex
```

Check retained evidence after a forced interruption. Keep the original bot directory and datasets until migration has been verified. Missing source credentials, labeled episodes, replay manifests, native routes and forward evidence remain explicit blocks in `/status`; starting this task does not supply them.
