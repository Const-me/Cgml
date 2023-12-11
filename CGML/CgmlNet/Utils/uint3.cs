namespace Cgml;
using System.Diagnostics.CodeAnalysis;

/// <summary>C# projection of <c>uint3</c> HLSL data type</summary>
public readonly struct uint3: IEquatable<uint3>
{
	/// <summary>Elements of the structure</summary>
	public readonly uint x, y, z;

	/// <summary>Create from 3 numbers</summary>
	public uint3( uint x, uint y, uint z )
	{
		this.x = x;
		this.y = y;
		this.z = z;
	}

	/// <summary>Create from 3 numbers</summary>
	public uint3( int x, int y, int z )
	{
		this.x = (uint)x;
		this.y = (uint)y;
		this.z = (uint)z;
	}

	/// <summary>String representation for debugger</summary>
	public override string ToString() => $"[ {x}, {y}, {z} ]";

	/// <summary>Compare for equality</summary>
	public bool Equals( uint3 other ) =>
		0 == ( ( x ^ other.x ) | ( y ^ other.y ) | ( z ^ other.z ) );

	/// <summary>Compare for equality</summary>
	public static bool operator ==( in uint3 a, in uint3 b ) => a.Equals( b );

	/// <summary>Compare for inequality</summary>
	public static bool operator !=( in uint3 a, in uint3 b ) => !a.Equals( b );

	/// <summary>Compare for equality</summary>
	public override bool Equals( [NotNullWhen( true )] object? obj )
	{
		if( obj is uint3 a )
			return Equals( a );
		return false;
	}

	/// <summary>Compute a hash code</summary>
	public override int GetHashCode() => HashCode.Combine( x, y, z );
}