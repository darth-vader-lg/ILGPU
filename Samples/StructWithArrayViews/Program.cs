using ILGPU;
using ILGPU.Runtime;
using StructWithArrayViews;
using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;

// Create main context
using var context = Context.Create(builder =>
{
    // Need to enable IO operations.
    //
    // Need to enable debugging of optimized kernels. By default,
    // the optimisation is at level 1, which would exclude IO operations.
    builder.Default().DebugConfig(
        enableIOOperations: true,
        forceDebuggingOfOptimizedKernels: true);
});

var device = context.Devices.OrderBy(d => d.AcceleratorType switch { AcceleratorType.Cuda => 0, AcceleratorType.OpenCL => 1, AcceleratorType.Velocity => 2, AcceleratorType.CPU => 3, _ => 4 }).First();
using var accelerator = device.CreateAccelerator(context);
Console.WriteLine($"Performing operations on {accelerator}");
var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, Struct, ArrayView1D<(double center, int length), Stride1D.Dense>>(Struct.Kernel);
var firstArray = accelerator.Allocate1D(Enumerable.Range(0, 100).Select(i => i).ToArray());
var secondArray = accelerator.Allocate1D(Enumerable.Range(0, 100).Select(i => (double)i).ToArray());
var complexArray = accelerator.Allocate1D(Enumerable.Range(0, 100).Select(i => (center: (double)i, length: i)).ToArray());
var clusters = ((double center, int[] indices)[])[
    (10.0, [1, 2, 4]),
    (20.0, [2, 1, 8, 10]),
    (30.0, []),
    (40.0, [5, 2]),
    ];
var clusterCenter = accelerator.Allocate1D(clusters.Select(c => c.center).ToArray());
var offsets = new int[clusters.Length + 1];
for (var (i, offset) = (0, 0); i <= clusters.Length; i++)
{
    offsets[i] = offset;
    if (i < clusters.Length)
        offset += clusters[i].indices.Length;
}
var clusterOffset = accelerator.Allocate1D(offsets);
var clusterIndices = accelerator.Allocate1D(clusters.SelectMany(c => c.indices).ToArray());

var input = new Struct { clusterCenter = clusterCenter, clusterIndices = clusterIndices, clusterOffset = clusterOffset, firstArray = firstArray, secondArray = secondArray };
kernel(1, input, complexArray.View);
accelerator.DefaultStream.Synchronize();
var timer = new Stopwatch();
timer.Start();
kernel(1, input, complexArray.View);
accelerator.DefaultStream.Synchronize();
timer.Stop();
Console.WriteLine($"Time = {timer.Elapsed.TotalMilliseconds} ms");
return;

namespace StructWithArrayViews
{
    public struct StructClustersData
    {
        public Struct owner;
        public StructCluster this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => new() { owner = owner, offset = owner.clusterOffset[index] };
        }
    }

    public struct StructCluster
    {
        public Struct owner;
        public int offset;
        public ref int this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref owner.clusterIndices[offset + index];
        }
    }

    public struct Struct
    {
        public ArrayView1D<double, Stride1D.Dense> clusterCenter;
        public ArrayView1D<int, Stride1D.Dense> clusterOffset;
        public ArrayView1D<int, Stride1D.Dense> clusterIndices;
        public ArrayView1D<int, Stride1D.Dense> firstArray;
        public ArrayView1D<double, Stride1D.Dense> secondArray;
        //private ClustersData Clusters => new() { owner = this };
        //public ref struct ClustersData
        //{
        //    public ref Struct owner;
        //    public ref struct Cluster
        //    {
        //        public ref Struct owner;
        //        public ref int offset;
        //        public ref int this[int index]
        //        {
        //            [MethodImpl(MethodImplOptions.AggressiveInlining)]
        //            get => ref owner.clusterIndices[offset + index];
        //        }
        //    }
        //    public Cluster this[int index]
        //    {
        //        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        //        get => new() { owner = owner, offset = owner.clusterOffset[index] };
        //    }
        //}
        private readonly ArrayView1D<int, Stride1D.Dense> GetCluster(int index) => clusterIndices.SubView(clusterOffset[index], clusterOffset[index + 1] - clusterOffset[index]);
        private void Kernel(Index1D index)
        {
            Interop.WriteLine("Index: {0}", index);
            Interop.WriteLine("First array:");
            for (var i = 0; i < firstArray.Length; i++)
                Interop.Write("{0},", firstArray[i]);
            Interop.Write("\r\n");
            Interop.WriteLine("Second array:");
            for (var i = 0; i < secondArray.Length; i++)
                Interop.Write("{0},", secondArray[i]);
            //for (var i = 0; i < 100000000; i++)
            //    GetCluster(1)[2] = GetCluster(1)[2];
            var c = new StructClustersData() { owner = this }[1];
            for (var i = 0; i < 100000000; i++)
            {
                c[2] = c[2] + 1;
            }
            var cd = new StructClustersData() { owner = this };
            Interop.WriteLine("Cluster[1][2] = {0}", cd[1][2]);
        }

        public static void Kernel(Index1D index, Struct input, ArrayView1D<(double center, int length), Stride1D.Dense> complexArray)
        {
            input.Kernel(index);
        }
    }
}
