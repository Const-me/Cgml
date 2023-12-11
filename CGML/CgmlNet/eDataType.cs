namespace Cgml;

/// <summary>Element type for all these tensors</summary>
public enum eDataType: byte
{
	/// <summary>Half-precision floats, a.k.a. IEEE 754 binary16</summary>
	/// <seealso href="https://en.wikipedia.org/wiki/Half-precision_floating-point_format" />
	FP16 = 0,
	/// <summary>32-bit floats</summary>
	FP32 = 1,
	/// <summary>32-bit integers</summary>
	U32 = 2,
	/// <summary>Non-standard half-precision floats</summary>
	/// <seealso href="https://en.wikipedia.org/wiki/Bfloat16_floating-point_format" />
	BF16 = 3,
}