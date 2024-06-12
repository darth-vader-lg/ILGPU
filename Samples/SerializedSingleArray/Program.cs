global using ILGPU;
global using ILGPU.Runtime;
using SerializedSingleArray;
using System.Diagnostics;

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
}
