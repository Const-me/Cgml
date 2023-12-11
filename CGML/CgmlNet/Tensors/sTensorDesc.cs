namespace Cgml;
using System.Diagnostics.CodeAnalysis;

/// <summary>Describes size and memory layout of a tensor in VRAM</summary>
public struct sTensorDesc: IEquatable<sTensorDesc>
{
	/// <summary>Size and stride</summary>
	public TensorShape shape;

	/// <summary>Type of elements</summary>
	public eDataType dataType;

	/// <summary>Usage flags for the GPU buffer</summary>
	public eBufferUse usage;

	/// <summary>VRAM layout of the tensor</summary>
	public eTensorLayout layout;

	/// <summary>Count of elements, up to 4 coordinates</summary>
	/// <remarks>The unused coordinates are set to 1</remarks>
	public Int128 size => shape.size;

	/// <summary>Strides of the tensor, expressed in elements</summary>
	/// <remarks>For dense row major tensor, these numbers are [ 1, size[0], size[0]*size[1], size[0]*size[1]*size[2] ]</remarks>
	public Int128 stride => shape.stride;

	/// <summary>Compare for equality</summary>
	public bool Equals( sTensorDesc other )
	{
		if( !shape.Equals( other.shape ) )
			return false;
		if( dataType != other.dataType )
			return false;
		if( usage != other.usage ) return false;
		if( layout != other.layout ) return false;
		return true;
	}

	/// <summary>Compare for equality</summary>
	public static bool operator ==( in sTensorDesc a, in sTensorDesc b ) => a.Equals( b );

	/// <summary>Compare for inequality</summary>
	public static bool operator !=( in sTensorDesc a, in sTensorDesc b ) => !a.Equals( b );

	/// <summary>Compare for equality</summary>
	public override bool Equals( [NotNullWhen( true )] object? obj )
	{
		if( obj is sTensorDesc a )
			return Equals( a );
		return false;
	}

	/// <summary>Compute a hash code</summary>
	public override int GetHashCode() =>
		HashCode.Combine( shape, dataType, usage, layout );
}