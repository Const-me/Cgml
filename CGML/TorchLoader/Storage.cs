namespace Torch;
using Cgml;
using System.Runtime.InteropServices;

/// <summary>Storage of a tensor</summary>
[StructLayout( LayoutKind.Auto )]
public readonly struct Storage
{
	/// <summary>Type of the tensor elements</summary>
	public readonly eDataType dataType;

	/// <summary>Name of the ZIP entry with the payload data of the tensor</summary>
	public readonly string payload;

	/// <summary>Memory required for the tensor data, but I don't know in which units.<br />
	/// For BF16, this seem to be count of elements, not bytes.</summary>
	public readonly int totalBytes;

	internal Storage( eDataType dt, object[] args )
	{
		if( args.Length != 3 )
			throw new ArgumentException();

		dataType = dt;
		payload = (string)args[ 0 ];
		// args[ 1 ] is deviceType string, values like "cpu" or "cuda:0"
		// We don't need these values.
		totalBytes = (int)args[ 2 ];
	}
}