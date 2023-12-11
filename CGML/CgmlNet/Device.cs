namespace Cgml;
using System.Runtime.InteropServices;

/// <summary>Utility structure which keeps both CGML device, and device context</summary>
[StructLayout( LayoutKind.Auto )]
public readonly struct Device: IDisposable
{
	/// <summary>The device interface represents a virtual adapter; it is used to create resources.</summary>
	public readonly iDevice device;

	/// <summary>Represents a device context which generates rendering commands</summary>
	public readonly iContext context;

	internal Device( iDevice device, iContext context )
	{
		this.device = device;
		this.context = context;
	}

	/// <summary>Release both COM objects</summary>
	public void Dispose()
	{
		context?.Dispose();
		device?.Dispose();
	}
}