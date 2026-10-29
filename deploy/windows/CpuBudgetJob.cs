using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

// A hard CPU budget covers this launcher and every descendant, including
// descendants that create their own nested jobs. No breakaway flags are set.
public static class MemecoinCpuBudgetJob
{
    private static IntPtr owner;
    private const uint KillOnClose = 0x2000, PriorityLimit = 0x20, BackgroundPriority = 0x4000;
    [StructLayout(LayoutKind.Sequential)] private struct BasicLimits {
        public long ProcessTime, JobTime;
        public uint Flags;
        public UIntPtr MinimumWorkingSet, MaximumWorkingSet;
        public uint ActiveProcesses;
        public UIntPtr Affinity;
        public uint Priority, Scheduling;
    }
    [StructLayout(LayoutKind.Sequential)] private struct IoCounters {
        public ulong ReadOperations, WriteOperations, OtherOperations;
        public ulong ReadBytes, WriteBytes, OtherBytes;
    }
    [StructLayout(LayoutKind.Sequential)] private struct ExtendedLimits {
        public BasicLimits Basic;
        public IoCounters Io;
        public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
    }
    [StructLayout(LayoutKind.Sequential)] private struct CpuRate { public uint Flags, Rate; }
    [StructLayout(LayoutKind.Sequential)] private struct Accounting {
        public long UserTime, KernelTime, PeriodUserTime, PeriodKernelTime;
        public uint PageFaults, TotalProcesses, ActiveProcesses, TerminatedProcesses;
    }
    public sealed class Snapshot {
        public uint ControlFlags, CpuRate, ActiveProcesses, PriorityClass, LimitFlags;
        public long CpuTime100ns;
        public long[] ProcessIds;
    }
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    private static extern IntPtr OpenJobObject(uint access, bool inherit, string name);
    [DllImport("kernel32.dll", SetLastError=true)]
    private static extern bool SetInformationJobObject(IntPtr job, int kind, IntPtr value, uint length);
    [DllImport("kernel32.dll", SetLastError=true)]
    private static extern bool QueryInformationJobObject(IntPtr job, int kind, IntPtr value, uint length, IntPtr returned);
    [DllImport("kernel32.dll", SetLastError=true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll")] private static extern IntPtr GetCurrentProcess();
    [DllImport("kernel32.dll")] private static extern bool CloseHandle(IntPtr handle);

    private static void Set<T>(IntPtr job, int kind, T value) {
        int size=Marshal.SizeOf(typeof(T)); IntPtr buffer=Marshal.AllocHGlobal(size);
        try {
            Marshal.StructureToPtr(value, buffer, false);
            if (!SetInformationJobObject(job, kind, buffer, (uint)size))
                throw new Win32Exception(Marshal.GetLastWin32Error());
        } finally { Marshal.FreeHGlobal(buffer); }
    }
    private static T Query<T>(IntPtr job, int kind) {
        int size=Marshal.SizeOf(typeof(T)); IntPtr buffer=Marshal.AllocHGlobal(size);
        try {
            if (!QueryInformationJobObject(job, kind, buffer, (uint)size, IntPtr.Zero))
                throw new Win32Exception(Marshal.GetLastWin32Error());
            return (T)Marshal.PtrToStructure(buffer, typeof(T));
        } finally { Marshal.FreeHGlobal(buffer); }
    }
    public static void Attach(string name, uint rate) {
        if (owner != IntPtr.Zero) throw new InvalidOperationException("CPU budget already attached");
        if (rate < 1 || rate > 10000 || String.IsNullOrEmpty(name))
            throw new ArgumentException("A named, positive CPU budget is required");
        IntPtr job=CreateJobObject(IntPtr.Zero, name);
        int error=Marshal.GetLastWin32Error();
        if (job==IntPtr.Zero) throw new Win32Exception(error);
        if (error==183) { CloseHandle(job); throw new InvalidOperationException("CPU role already exists"); }
        try {
            CpuRate cpu=new CpuRate(); cpu.Flags=5; cpu.Rate=rate; // ENABLE | HARD_CAP
            Set(job, 15, cpu);
            ExtendedLimits limits=new ExtendedLimits();
            limits.Basic.Flags=KillOnClose | PriorityLimit;
            limits.Basic.Priority=BackgroundPriority;
            Set(job, 9, limits);
            if (!AssignProcessToJobObject(job, GetCurrentProcess()))
                throw new Win32Exception(Marshal.GetLastWin32Error());
            owner=job;
        } catch { CloseHandle(job); throw; }
    }
    private static Snapshot Read(IntPtr job) {
        CpuRate cpu=Query<CpuRate>(job, 15);
        Accounting accounting=Query<Accounting>(job, 1);
        ExtendedLimits limits=Query<ExtendedLimits>(job, 9);
        long[] ids=null;
        for(int capacity=32; capacity<=65536; capacity*=2) {
            int size=8+capacity*IntPtr.Size; IntPtr buffer=Marshal.AllocHGlobal(size);
            try {
                if (!QueryInformationJobObject(job, 3, buffer, (uint)size, IntPtr.Zero)) {
                    int error=Marshal.GetLastWin32Error();
                    if(error==234) continue;
                    throw new Win32Exception(error);
                }
                int count=Marshal.ReadInt32(buffer, 4);
                if(count<0 || count>capacity) throw new InvalidOperationException("Invalid job census");
                ids=new long[count];
                for(int i=0;i<count;i++) ids[i]=Marshal.ReadIntPtr(buffer,8+i*IntPtr.Size).ToInt64();
                break;
            } finally { Marshal.FreeHGlobal(buffer); }
        }
        if(ids==null) throw new InvalidOperationException("Job census exceeds inspection capacity");
        return new Snapshot { ControlFlags=cpu.Flags, CpuRate=cpu.Rate,
            ActiveProcesses=accounting.ActiveProcesses, PriorityClass=limits.Basic.Priority,
            LimitFlags=limits.Basic.Flags, CpuTime100ns=accounting.UserTime+accounting.KernelTime,
            ProcessIds=ids };
    }
    public static Snapshot Current() {
        if(owner==IntPtr.Zero) throw new InvalidOperationException("CPU budget is not attached");
        return Read(owner);
    }
    public static Snapshot Inspect(string name) {
        IntPtr job=OpenJobObject(4, false, name); // JOB_OBJECT_QUERY
        if(job==IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
        try { return Read(job); } finally { CloseHandle(job); }
    }
    public static void ReleaseAfterChildrenExit() {
        if(owner==IntPtr.Zero) return;
        if(Current().ActiveProcesses!=1) throw new InvalidOperationException("Children still running");
        // Avoid terminating our own successful launcher at normal exit. On
        // abnormal owner termination KILL_ON_JOB_CLOSE still kills the tree.
        ExtendedLimits limits=Query<ExtendedLimits>(owner, 9);
        limits.Basic.Flags &= ~KillOnClose;
        Set(owner, 9, limits);
        CloseHandle(owner); owner=IntPtr.Zero;
    }
}
