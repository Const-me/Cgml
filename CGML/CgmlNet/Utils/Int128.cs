namespace Cgml;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Serialization;

/// <summary>4 integers, 16 bytes in total</summary>
/// <remarks>Also a C# projection of <c>uint4</c> and <c>int4</c> HLSL data types.</remarks>
[DataContract]
public readonly struct Int128: IEquatable<Int128>
{
	/// <summary>Elements of the structure</summary>
	[DataMember]
	public readonly int x, y, z, w;

	/// <summary>Create from 4 integers</summary>
	public Int128( int x, int y, int z, int w )
	{
		this.x = x;
		this.y = y;
		this.z = z;
		this.w = w;
	}

	/// <summary>Create from SIMD vector</summary>
	public static implicit operator Int128( Vector128<int> vec )
	{
		Int128 res;
		unsafe
		{
			Sse2.Store( (int*)&res, vec );
		}
		return res;
	}

	/// <summary>Get first element</summary>
	public int ToScalar() => x;

	/// <summary>Get element by index</summary>
	[MethodImpl( MethodImplOptions.AggressiveInlining )]
	public int GetElement( byte index ) => index switch
	{
		0 => x,
		1 => y,
		2 => z,
		3 => w,
		_ => throw new ArgumentOutOfRangeException( nameof( index ) )
	};

	/// <summary>Load into a SIMD vector with uint32 elements</summary>
	public Vector128<uint> AsUInt32()
	{
		unsafe
		{
			fixed( Int128* rsi = &this )
				return Sse2.LoadVector128( (uint*)rsi );
		}
	}

	/// <summary>Product of all 4 numbers</summary>
	/// <remarks>Throws <see cref="OverflowException" /> if the result exceeds 2G elements</remarks>
	public int horizontalProduct()
	{
		Vector128<uint> a = AsUInt32();
		Vector128<uint> b = Sse2.ShiftRightLogical128BitLane( a, 4 );
		Vector128<ulong> p2 = Sse2.Multiply( a, b );
		ulong res = p2.GetElement( 1 );
		res *= p2.ToScalar();
		return (int)res;
	}

	/// <summary>String representation for debugger</summary>
	public override string ToString() => $"[ {x}, {y}, {z}, {w} ]";

	/// <summary>Compare for equality</summary>
	[MethodImpl( MethodImplOptions.AggressiveInlining )]
	public bool Equals( Int128 other )
	{
		unsafe
		{
			fixed( Int128* pi = &this )
			{
				Int128* pa = &other;
				var i = Sse2.LoadVector128( (int*)pi );
				var a = Sse2.LoadVector128( (int*)pa );
				var xx = Sse2.Xor( a, i );
				return Sse41.TestZ( xx, xx );
			}
		}
	}

	/// <summary>Compare for equality</summary>
	public static bool operator ==( in Int128 a, in Int128 b ) => a.Equals( b );

	/// <summary>Compare for inequality</summary>
	public static bool operator !=( in Int128 a, in Int128 b ) => !a.Equals( b );

	/// <summary>Compare for equality</summary>
	public override bool Equals( [NotNullWhen( true )] object? obj )
	{
		if( obj is Int128 a )
			return Equals( a );
		return false;
	}

	/// <summary>Compute a hash code</summary>
	public override int GetHashCode() => HashCode.Combine( x, y, z, w );

	/// <summary>Make a 3D vector with [ y, z, w ] fields</summary>
	[IgnoreDataMember]
	public uint3 yzw => new uint3( y, z, w );

	/// <summary>Make a 2D vector with [ z, w ] fields</summary>
	[IgnoreDataMember]
	public uint2 zw => new uint2( z, w );

	/// <summary>Make a 2D vector with [ x, y ] fields</summary>
	[IgnoreDataMember]
	public uint2 xy => new uint2( x, y );

	/// <summary>Make a 2D vector with [ y, z ] fields</summary>
	[IgnoreDataMember]
	public uint2 yz => new uint2( y, z );

	/// <summary>Make a 2D vector with [ y, x ] fields</summary>
	[IgnoreDataMember]
	public uint2 yx => new uint2( y, x );

	/// <summary>Copy 4 integers from this structure into the supplied span</summary>
	public void store( Span<int> span )
	{
		if( span.Length == 4 )
		{
			unsafe
			{
				fixed( Int128* rsi = &this )
				fixed( int* rdi = span )
				{
					var v = Sse2.LoadVector128( (int*)rsi );
					Sse2.Store( rdi, v );
					return;
				}
			}
		}
		throw new ArgumentException();
	}

	/// <summary>Create from 4 integers in memory</summary>
	public static Int128 load( ReadOnlySpan<int> span )
	{
		if( span.Length == 4 )
		{
			unsafe
			{
				fixed( int* rsi = span )
					return Sse2.LoadVector128( rsi );
			}
		}
		throw new ArgumentException();
	}
}