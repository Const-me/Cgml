namespace Cgml;

/// <summary>Loosely corresponds to <c>D3D11_USAGE</c> enumeration</summary>
/// <seealso href="https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_usage" />
public enum eBufferUse: byte
{
	/// <summary>Immutable tensor, readable from GPU</summary>
	Immutable = 0,
	/// <summary>Read+write tensor, readable and writable on GPU</summary>
	ReadWrite = 1,
	/// <summary>Read+write tensor, readable and writable on GPU, which supports downloads from GPU</summary>
	ReadWriteDownload = 2,
	/// <summary>The tensor is accessible by both GPU (read only) and CPU (write only). Optimized for resources frequently updated from CPU.</summary>
	Dynamic = 3,
}