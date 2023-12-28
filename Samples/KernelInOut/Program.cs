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
    var input = new EnumerableArrayView<Sys>(inputPtr);
    var output = new EnumerableArrayView<Sys>(outputPtr);
    for (var (i, iLen) = (0, Math.Min(input.Count, output.Count)); i < iLen; ++i)
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
        int Count { get; }
        ref T this[int index] { get; }
    }

    public struct EnumerableArrayView<T>(ArrayView<T> view) : IEnumerableArrayView<T> where T : unmanaged
    {
        public ref T this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref view[index];
        }

        public int Count
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => view.IntLength;
        }

        public IEnumerator<T> GetEnumerator()
        {
            for (var (i, iLen) = (0, view.IntLength); i < iLen; ++i)
                yield return view[i];
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
