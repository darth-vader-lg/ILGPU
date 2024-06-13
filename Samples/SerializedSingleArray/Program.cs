global using ILGPU;
global using ILGPU.Runtime;
using SerializedSingleArray;
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

var particleSystem = new ParticleSystem(accelerator, [[1, 2, 3], [4.5, 2], [30, -10, 1, 7], [], [0, 1, 2, 3]]);
var particleKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<(int offset, int count)>, ArrayView<double>>(ParticleSystem.ParticleKernel);
particleKernel(1, particleSystem.clustersArray, particleSystem.pointsArray);
accelerator.DefaultStream.Synchronize();



var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>>(Kernel);
using var input = accelerator.Allocate1D(new InputData().ToArray());

kernel(1, input.View);

// Interop.WriteLine may require a call to stream.Synchronize() for the
// contents to be flushed.
accelerator.DefaultStream.Synchronize();

var timer = new Stopwatch();
timer.Start();
kernel(1, input.View);
accelerator.DefaultStream.Synchronize();
timer.Stop();
Console.WriteLine($"Time = {timer.Elapsed.TotalMilliseconds} ms");

return;

static unsafe void Kernel(
    Index1D index,
    ArrayView<byte> inputDataView)
{
    var inputData = new InputData(inputDataView);
    for (var i = 0; i < 10000000; ++i)
        inputData.doubles[0] = inputData.doubles[1];
    // NB: String interpolation, alignment, spacing, format and precision
    // specifiers are not currently supported. Use standard {x} placeholders.
    Interop.WriteLine("{0} = {1}, {2}, {3}", index, inputData.iValue, inputData.dptr[0], inputData.dptr[1]);
    ;
}


namespace SerializedSingleArray
{
    unsafe ref struct InputData
    {
        internal int iValue;
        internal ArrayView<double> doubles;
        internal double* dptr;

        public InputData()
        {
            iValue = 123;
        }
        public InputData(ArrayView<byte> inputData)
        {
            var vv = inputData.SubView(0, sizeof(int)).Cast<int>().VariableView(0).Value;
            iValue = vv;
            doubles = inputData.SubView(sizeof(int) * 2, sizeof(double) * 2).Cast<double>().SubView(0, 2);
            fixed (double* ptr = &doubles[0]) dptr = ptr;
        }
        public readonly byte[] ToArray()
        {
            using var stream = new MemoryStream();
            using var writer = new BinaryWriter(stream);
            writer.Write(iValue);
            writer.Write(iValue);
            writer.Write(1.0);
            writer.Write(2.0);
            stream.Flush();
            return stream.GetBuffer();
        }
    }


    public readonly ref struct ParticleSystem
    {
        private readonly Index1D index;
        public readonly ArrayView<(int offset, int count)> clustersArray;
        public readonly ArrayView<double> pointsArray;
        public ref struct ClustersData(ArrayView<(int offset, int count)> clusters, ArrayView<double> points)
        {
            public readonly Cluster this[int index]
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => new(clusters, points, index);
            }
            public readonly int Length
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => clusters.IntLength;
            }
        }
        public ref struct Cluster(ArrayView<(int offset, int count)> clusters, ArrayView<double> points, int cluster)
        {
            public readonly ref double this[int index]
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => ref points[clusters[cluster].offset + index];
            }
            public readonly int Length
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => clusters[cluster].count;
            }
        }
        public ParticleSystem(Accelerator accelerator, double[][] p)
        {
            var count = p.Length;
            var _clusters = new (int offset, int count)[count];
            var _points = new double[p.Select(points => points.Length).Sum()];
            for (var i = 0; i < _clusters.Length; i++)
            {
                _clusters[i] = (i == 0 ? 0 : _clusters[i - 1].offset + _clusters[i - 1].count, p[i].Length);
                p[i].CopyTo(_points, _clusters[i].offset);
            }
            clustersArray = accelerator.Allocate1D(_clusters).AsArrayView<(int offset, int count)>(0, _clusters.Length);
            pointsArray = accelerator.Allocate1D(_points).AsArrayView<double>(0, _points.Length);
        }
        public ParticleSystem(Index1D index, ArrayView<(int offset, int count)> clusters, ArrayView<double> points)
        {
            this.index = index;
            this.clustersArray = clusters;
            this.pointsArray = points;
        }
        private void Kernel()
        {
            var clusters = new ClustersData(clustersArray, pointsArray);
            // NB: String interpolation, alignment, spacing, format and precision
            // specifiers are not currently supported. Use standard {x} placeholders.
            for (var i = 0; i < clustersArray.Length; i++)
            {
                Interop.WriteLine("Cluster {0}:", i);
                for (var j = 0; j < clusters[i].Length; j++)
                    Interop.Write("{0}, ", clusters[i][j]);
                Interop.Write("\r\n");
            }
        }

        public static void ParticleKernel(Index1D index, ArrayView<(int offset, int count)> clusters, ArrayView<double> points)
        {
            var particleKernel = new ParticleSystem(index, clusters, points);
            particleKernel.Kernel();
        }
    }
}
