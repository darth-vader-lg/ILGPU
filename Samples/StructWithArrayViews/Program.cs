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

// Selezione del dispositivo
var device = context.Devices.OrderBy(d => d.AcceleratorType switch { AcceleratorType.Cuda => 0, AcceleratorType.OpenCL => 1, AcceleratorType.Velocity => 2, AcceleratorType.CPU => 3, _ => 4 }).First();
// Crea l'acceleratore
using var accelerator = device.CreateAccelerator(context);
Console.WriteLine($"Performing operations on {accelerator}");
// Crea la funzione di run del kernel
var kernelRun = accelerator.LoadAutoGroupedStreamKernel((Action<Index1D, StructWithArrayViews.Kernel>)StructWithArrayViews.Kernel.Run);
var clusters = ((double center, int[] indices)[])[
    (10.0, [1, 2, 4]),
    (20.0, [2, 1, 8, 10]),
    (30.0, []),
    (40.0, [5, 2]),
    ];
// Enumeratore di clusters
IEnumerable<ClusterInfo> ClustersEnumerator()
{
    if (clusters.Length == 0)
        yield break;
    yield return new()
    {
        center = clusters[0].center,
        offset = 0,
        length = clusters[0].indices.Length
    };
    for (var (i, offset) = (1, clusters[0].indices.Length); i < clusters.Length; offset += clusters[i].indices.Length, i++)
    {
        yield return new()
        {
            center = clusters[i].center,
            offset = offset,
            length = clusters[i].indices.Length
        };
    }
}
// Crea la struttura dati del kernel
var kernel = new StructWithArrayViews.Kernel
{
    clusters = accelerator.Allocate1D(ClustersEnumerator().ToArray()),
    clusterIndices = accelerator.Allocate1D(clusters.SelectMany(c => c.indices).ToArray())
};
// Avvia cronometro
var timer = new Stopwatch();
timer.Start();
// Avvia il kernel ed attende il termine
kernelRun(1, kernel);
accelerator.DefaultStream.Synchronize();
// Stoppa il timer e visualizza il tempo
timer.Stop();
Console.WriteLine($"Time = {timer.Elapsed.TotalMilliseconds} ms");
return;

namespace StructWithArrayViews
{
    /// <summary>
    /// Elenco di clusters
    /// </summary>
    public struct Clusters
    {
        #region Fields
        /// <summary>
        /// Informazioni sui clusters
        /// </summary>
        private readonly ArrayView1D<ClusterInfo, Stride1D.Dense> clusters;
        /// <summary>
        /// Indici dei vertici nei clusters
        /// </summary>
        private ArrayView1D<int, Stride1D.Dense> clusterIndices;
        #endregion
        #region Properties
        /// <summary>
        /// Lunghezza array di clusters
        /// </summary>
        public readonly int Length
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => clusters.IntLength;
        }
        /// <summary>
        /// Indicizzatore di cluster
        /// </summary>
        /// <param name="index">Indice cluster</param>
        /// <returns>Il cluster</returns>
        public Cluster this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => new(ref clusters[index], ref clusterIndices);
        }
        #endregion
        #region Methods
        /// <summary>
        /// Costruttore
        /// </summary>
        /// <param name="clusters">Informazioni sui clusters</param>
        /// <param name="clusterIndices">Indici dei vertici nei clusters</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Clusters(ref ArrayView1D<ClusterInfo, Stride1D.Dense> clusters, ref ArrayView1D<int, Stride1D.Dense> clusterIndices)
        {
            this.clusters = clusters;
            this.clusterIndices = clusterIndices;
        }
        #endregion
    }

    public struct Cluster
    {
        #region Fields
        /// <summary>
        /// Informazioni sul cluster
        /// </summary>
        private ClusterInfo cluster;
        /// <summary>
        /// Indici dei vertici nel cluster
        /// </summary>
        private readonly ArrayView1D<int, Stride1D.Dense> indices;
        #endregion
        #region Properties
        /// <summary>
        /// Centroide del cluster
        /// </summary>
        public double Center
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => cluster.center;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => cluster.center = value;
        }
        /// <summary>
        /// Numero di indici di vertici
        /// </summary>
        public readonly int Length
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => cluster.length;
        }
        /// <summary>
        /// Indicizzatore di indice di vertice
        /// </summary>
        /// <param name="index">Indice</param>
        /// <returns>L'indice del vertice</returns>
        public readonly ref int this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref indices[index];
        }
        #endregion
        #region Methods
        /// <summary>
        /// Costruttore
        /// </summary>
        /// <param name="cluster">Informazioni sul cluster</param>
        /// <param name="clusterIndices">Array globale di indici</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Cluster(ref ClusterInfo cluster, ref ArrayView1D<int, Stride1D.Dense> clusterIndices)
        {
            this.cluster = cluster;
            indices = clusterIndices.SubView(cluster.offset, cluster.length);
        }
        #endregion
    }

    /// <summary>
    /// Informazioni sul cluster
    /// </summary>
    public struct ClusterInfo
    {
        #region Fields
        /// <summary>
        /// Centroide
        /// </summary>
        public double center;
        /// <summary>
        /// Lunghezza array di indici
        /// </summary>
        public int length;
        /// <summary>
        /// Offset di inizio indici vertici nell'array globale di vertici
        /// </summary>
        public int offset;
        #endregion
    }

    /// <summary>
    /// Kernel acceleratore
    /// </summary>
    public struct Kernel
    {
        #region Fields
        /// <summary>
        /// Array di cluster
        /// </summary>
        public ArrayView1D<ClusterInfo, Stride1D.Dense> clusters;
        /// <summary>
        /// Array di indici di vertici nei clusters
        /// </summary>
        public ArrayView1D<int, Stride1D.Dense> clusterIndices;
        #endregion
        #region Properties
        /// <summary>
        /// Elenco di clusters
        /// </summary>
        public Clusters Clusters
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => new(ref clusters, ref clusterIndices);
        }
        #endregion
        /// <summary>
        /// Funzione principale del kernel
        /// </summary>
        /// <param name="index">Indice del task</param>
        private void Run(Index1D index)
        {
            //for (var i = 0; i < 100000000; i++)
            //    Clusters[1][2] = Clusters[1][2] + 1;
            //Interop.WriteLine("Cluster[1][2] = {0}", Clusters[1][2]);
            var cluster = Clusters[1];
            for (var i = 0; i < 100000000; i++)
                cluster[2] = cluster[2] + 1;
            Interop.WriteLine("Cluster[1][2] = {0}", cluster[2]);
            //var clusterView = clusterIndices.SubView(clusters[1].offset, clusters[1].len);
            //for (var i = 0; i < 100000000; i++)
            //    clusterView[2] = clusterView[2] + 1;
            //Interop.WriteLine("Cluster[1][2] = {0}", clusterView[2]);
        }
        /// <summary>
        /// Funzione principale del kernel 
        /// </summary>
        /// <param name="index">Indice del task</param>
        /// <param name="kernel">Dati per il kernel</param>
        public static void Run(Index1D index, Kernel kernel)
        {
            kernel.Run(index);
        }
    }
}
