namespace Cgml;

/// <summary>VRAM layout of the tensor</summary>
public enum eTensorLayout: byte
{
	/// <summary>The tensor is dense</summary>
	Dense = 0,

	/// <summary>The weights are compressed with a special BCML1 lossy codec.</summary>
	/// <remarks>The weights are quantized into 4 bits per element.<br />
	/// Each block of 32 elements is consuming 20 bytes of VRAM: 4 bytes for block header, and 16 bytes for the quantized weights.</remarks>
	BCML1 = 1,
}