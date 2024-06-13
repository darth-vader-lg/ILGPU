using ILGPU.Runtime;
using ILGPU;
using StructWithArrayViews;

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
var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, Struct>(Struct.Kernel);
var firstArray = accelerator.Allocate1D(Enumerable.Range(0, 100).ToArray());
var secondArray = accelerator.Allocate1D(Enumerable.Range(0, 100).Select(i => (double)i).ToArray());
var input = new Struct { firstArray = firstArray, secondArray = secondArray };
kernel(1, input);
accelerator.DefaultStream.Synchronize();
return;

namespace StructWithArrayViews
{
    public struct Struct
    {
        public ArrayView1D<int, Stride1D.Dense> firstArray;
        public ArrayView1D<double, Stride1D.Dense> secondArray;
        private readonly void Kernel(Index1D index)
        {
            Interop.WriteLine("Index: {0}", index);
            Interop.WriteLine("First array:");
            for (var i = 0; i < firstArray.Length; i++)
                Interop.Write("{0},", firstArray[i]);
            Interop.Write("\r\n");
            Interop.WriteLine("Second array:");
            for (var i = 0; i < secondArray.Length; i++)
                Interop.Write("{0},", secondArray[i]);
        }

        public static void Kernel(Index1D index, Struct input)
        {
            input.Kernel(index);
        }
    }
}
