global using ILGPU;
global using ILGPU.Runtime;
using KernelParameters;

const int DataSize = 1024;

using var context = Context.CreateDefault();
// For each available device...

foreach (var device in context)
{
    // Create accelerator for the given device
    using var accelerator = device.CreateAccelerator(context);
    Console.WriteLine($"Performing operations on {accelerator}");

    var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, int, LambdaClosure>(Kernel);
    using var buffer = accelerator.Allocate1D<long>(DataSize);
    kernel((int)buffer.Length, buffer.View, 1, new LambdaClosure(20));

    var data = buffer.GetAsArray1D();
}

/// <summary>
/// A generic kernel that uses generic arguments to emulate a lambda-function delegate.
/// </summary>
/// <typeparam name="TKernelFunction">The custom kernel functionality.</typeparam>
/// <typeparam name="T">The element type.</typeparam>
/// <param name="index">The element index.</param>
/// <param name="data">The target data array.</param>
/// <param name="value">The constant input value.</param>
/// <param name="function">The domain and context-specific kernel lambda function.</param>
static void Kernel<TKernelFunction, T>(
    Index1D index,
    ArrayView<T> data,
    int value,
    TKernelFunction function)
    where TKernelFunction : struct, IKernelFunction<T>
    where T : unmanaged
{
    // Invoke the custom "lambda function"
    data[index] = function.ComputeValue(index, value);
}

namespace KernelParameters
{
    /// <summary>
    /// An interface constraint for the <see cref="Kernel{TKernelFunction, T}(Index1D, ArrayView{T}, int, TKernelFunction)"/> function.
    /// This helps to emulate a lambda-function delegate that is passed to a kernel in a type safe way.
    /// </summary>
    /// <typeparam name="T">The element type that is returned by the <see cref="ComputeValue(Index1D, int)"/> function.</typeparam>
    public interface IKernelFunction<T>
        where T : struct
    {
        /// <summary>
        /// Computes a domain-specific value.
        /// </summary>
        /// <param name="index">The element index.</param>
        /// <param name="value">The kernel-context specific value.</param>
        /// <returns>The computed value.</returns>
        T ComputeValue(Index1D index, int value);
    }

    /// <summary>
    /// Implements a custom lambda closure
    /// </summary>
    /// <remarks>
    /// Constructs a new lambda closure.
    /// </remarks>
    /// <param name="offset">The offset to use.</param>
    public readonly struct LambdaClosure(long offset) : IKernelFunction<long>
    {
        /// <summary>
        /// Returns the offset to add to all elements.
        /// </summary>
        public long Offset { get; } = offset;
        /// <summary cref="IKernelFunction{T}.ComputeValue(Index1D, int)"/>
        public long ComputeValue(Index1D index, int value) =>
            Offset + value * index;
    }
}
