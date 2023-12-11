namespace Cgml;

/// <summary>Optional format transformation while creating immutable tensors</summary>
/// <seealso cref="iDevice.loadImmutableTensor" />
public enum eLoadTransform: byte
{
	/// <summary>Don't transform anything</summary>
	None = 0,
	/// <summary>Convert <see cref="eDataType.BF16" /> elements into <see cref="eDataType.FP16" /></summary>
	Fp16MakeIeee = 1,
}