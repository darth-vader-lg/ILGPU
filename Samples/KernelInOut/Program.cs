global using ILGPU;
global using ILGPU.Runtime;
using KernelInOut;

using var context = Context.CreateDefault();
// Fastest device...
var device = context.Devices.OrderBy(d => d.AcceleratorType switch { AcceleratorType.Cuda => 0, AcceleratorType.OpenCL => 1, AcceleratorType.Velocity => 2, AcceleratorType.CPU => 3, _ => 4 }).First();
// Create accelerator for the given device
using var accelerator = device.CreateAccelerator(context);
Console.WriteLine($"Performing operations on {accelerator}");

var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, LambdaClosure>(Kernel);
using var input = accelerator.Allocate1D<byte>(1024);
using var output = accelerator.Allocate1D<byte>(1024);
var lambaClosure = new LambdaClosure();
kernel((int)input.Length, input.View, output.View, lambaClosure);

var data = output.GetAsArray1D();
;

static void Kernel<TKernelFunction>(
    Index1D index,
    ArrayView<byte> input,
    ArrayView<byte> output,
    TKernelFunction function)
    where TKernelFunction : struct, IKernelFunction
{
    // Invoke the custom "lambda function"
    function.Compute(index);
}

namespace KernelInOut
{
    public interface IKernelFunction
    {
        void Compute(Index1D index);
    }

    /// <summary>
    /// Implements a custom lambda closure
    /// </summary>
    /// <remarks>
    /// Constructs a new lambda closure.
    /// </remarks>
    /// <param name="offset">The offset to use.</param>
    public readonly struct LambdaClosure(long offset) : IKernelFunction
    {
        /// <summary>
        /// Returns the offset to add to all elements.
        /// </summary>
        public long Offset { get; } = offset;
        /// <summary cref="IKernelFunction{T}.ComputeValue(Index1D, int)"/>
        public readonly void Compute(Index1D index)
        {
            return;
        }
    }
}
