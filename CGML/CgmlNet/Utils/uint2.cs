namespace Cgml;
using System.Diagnostics.CodeAnalysis;

/// <summary>C# projection of <c>uint2</c> HLSL data type</summary>
public readonly struct uint2: IEquatable<uint2>
{
	/// <summary>Elements of the structure</summary>
	public readonly uint x, y;

	/// <summary>Create from 2 numbers</summary>
	public uint2( uint x, uint y )
	{
		this.x = x;
		this.y = y;
	}

	/// <summary>Create from 2 numbers</summary>
	public uint2( int x, int y )
	{
		this.x = (uint)x;
		this.y = (uint)y;
	}

	/// <summary>Bitcast FP64 number into <c>uint2</c> structure</summary>
	public static explicit operator uint2( double fp64 )
	{
		ulong u = BitConverter.DoubleToUInt64Bits( fp64 );
		uint low = unchecked((uint)( u ));
		uint high = (uint)( u >> 32 );
		return new uint2( low, high );
	}

	/// <summary>String representation for debugger</summary>
	public override string ToString() => $"[ {x}, {y} ]";

	/// <summary>Compare for equality</summary>
	public bool Equals( uint2 other ) =>
		0 == ( ( x ^ other.x ) | ( y ^ other.y ) );

	/// <summary>Compare for equality</summary>
	public static bool operator ==( in uint2 a, in uint2 b ) => a.Equals( b );

	/// <summary>Compare for inequality</summary>
	public static bool operator !=( in uint2 a, in uint2 b ) => !a.Equals( b );

	/// <summary>Compare for equality</summary>
	public override bool Equals( [NotNullWhen( true )] object? obj )
	{
		if( obj is uint2 a )
			return Equals( a );
		return false;
	}

	/// <summary>Compute a hash code</summary>
	public override int GetHashCode() => HashCode.Combine( x, y );
}