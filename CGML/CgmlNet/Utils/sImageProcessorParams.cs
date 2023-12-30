namespace Cgml;
using System.Numerics;

/// <summary>Input parameters for importing images as a tensor</summary>
public struct sImageProcessorParams
{
	/// <summary><c>xy</c> size of the output tensor</summary>
	/// <remarks><c>z</c> size is 3, because 3 RGB channels.</remarks>
	public int width, height;

	/// <summary>The implementation uses the following formula to normalize the numbers written to the output tensor:<br />
	/// <c>rgb = ( rgb - imageMean ) / imageStd</c></summary>
	/// <remarks>To disable the normalization, set <c>imageMean</c> to zero, and <c>imageStd</c> to one.</remarks>
	public Vector3 imageMean, imageStd;
}