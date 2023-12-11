namespace Mistral;

/// <summary>Some user-adjustable parameters</summary>
public sealed class PerformanceParams
{
	/// <summary>When true, might fail with DXGI timeout exceptions on slow integrated GPUs, will run faster on discrete GPUs.</summary>
	/// <remarks>The default is false.</remarks>
	public bool isFastGpu { get; set; }

	internal PerformanceParams()
	{
		isFastGpu = false;
	}
}