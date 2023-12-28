global using ILGPU;
global using ILGPU.Runtime;
using KernelInOut;
using System.Collections;
using System.Runtime.CompilerServices;

using var context = Context.CreateDefault();
var device = context.Devices.OrderBy(d => d.AcceleratorType switch { AcceleratorType.Cuda => 0, AcceleratorType.OpenCL => 1, AcceleratorType.Velocity => 2, AcceleratorType.CPU => 3, _ => 4 }).First();
using var accelerator = device.CreateAccelerator(context);
Console.WriteLine($"Performing operations on {accelerator}");

var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<Sys>, ArrayView<Sys>>(Kernel);
using var input = accelerator.Allocate1D(Enumerable.Range(0, 1024).Select(i => new Sys { p = (i, i, i) }).ToArray());
using var output = accelerator.Allocate1D<Sys>(input.Length);

kernel((int)input.Length, input.View, output.View);

var data = output.GetAsArray1D();
;

static void Kernel(
    Index1D index,
    ArrayView<Sys> inputPtr,
    ArrayView<Sys> outputPtr)
{
    var input = new GPUView<Sys>(inputPtr);
    var output = new GPUView<Sys>(outputPtr);
    for (var (i, iLen) = (0, Math.Min(input.Length, output.Length)); i < iLen; ++i)
        output[i] = input[i];
}

namespace KernelInOut
{
    public struct Sys
    {
        public (double x, double y, double z) x;
        public (double x, double y, double z) y;
        public (double x, double y, double z) z;
        public (double x, double y, double z) p;
    }

    public interface IEnumerableArrayView<T> : IEnumerable<T> where T : unmanaged
    {
        int Length { get; }
        ref T this[int index] { get; }
    }

    public struct GPUView<T>(ArrayView<T> view) : IEnumerableArrayView<T> where T : unmanaged
    {
        public readonly ref T this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref view[index];
        }

        public readonly int Length
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => view.IntLength;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly IEnumerator<T> GetEnumerator()
        {
            for (var (i, iLen) = (0, view.IntLength); i < iLen; ++i)
                yield return view[i];
        }

        readonly IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    public struct CPUView<T>(T[] view) : IEnumerableArrayView<T> where T : unmanaged
    {
        public readonly ref T this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref view[index];
        }

        public readonly int Length
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => view.Length;
        }

        public readonly IEnumerator<T> GetEnumerator()
        {
            return view.AsEnumerable().GetEnumerator();
        }

        readonly IEnumerator IEnumerable.GetEnumerator()
        {
            return view.GetEnumerator();
        }
    }
}
