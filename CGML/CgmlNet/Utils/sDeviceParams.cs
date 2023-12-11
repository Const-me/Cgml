namespace Cgml;
using System.Runtime.InteropServices;

/// <summary>Initialization parameters for DirectCompute device</summary>
public struct sDeviceParams
{
	/// <summary>Name or 0-based index of the GPU to use</summary>
	[MarshalAs( UnmanagedType.LPWStr )]
	public readonly string? adapter;

	/// <summary>GPU queue depth limit</summary>
	public readonly int queueDepth;

	/// <summary>GPU queue length to use by default</summary>
	public const int defaultQueueDepth = 4;

	/// <summary>Miscellaneous initialization flags</summary>
	[Flags]
	public enum eDeviceFlags: byte
	{
		/// <summary>No special flags</summary>
		None = 0,
		/// <summary>Sacrifice a bit of performance to improve power efficiency</summary>
		PowerSaver = 1,
	}

	/// <summary>Miscellaneous initialization flags</summary>
	public readonly eDeviceFlags flags;

	/// <summary>Create the structure</summary>
	public sDeviceParams( string? adapter = null, int? queueDepth = null, eDeviceFlags flags = eDeviceFlags.None )
	{
		this.adapter = adapter;
		this.queueDepth = queueDepth ?? defaultQueueDepth;
		this.flags = flags;
	}
}