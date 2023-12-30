namespace Cgml;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Serialization;

/// <summary>Size and memory layout of a tensor</summary>
[DataContract]
public readonly struct TensorShape: IEquatable<TensorShape>
{
	/// <summary>Count of elements, up to 4 coordinates</summary>
	/// <remarks>The unused coordinates are set to 1</remarks>
	[DataMember]
	public readonly Int128 size;

	/// <summary>Strides of the tensor, expressed in elements</summary>
	/// <remarks>For dense row major tensor, these numbers are [ 1, size[0], size[0]*size[1], size[0]*size[1]*size[2] ]<br/>
	/// For dense column major matrix, these numbers are [ size[1], 1, size[0]*size[1], size[0]*size[1], size[0]*size[1] ]</remarks>
	[DataMember]
	public readonly Int128 stride;

	/// <summary>True if this tensor is a vector</summary>
	[IgnoreDataMember]
	public bool isVector => size.yzw == new uint3( 1, 1, 1 );

	/// <summary>True if this tensor is a matrix or vector</summary>
	[IgnoreDataMember]
	public bool isMatrix => size.zw == new uint2( 1, 1 );

	/// <summary>A string for debugger</summary>
	public override string ToString()
	{
		if( isVector )
			return $"Vector [ {size.ToScalar()} ]";
		if( isMatrix )
		{
			if( 1 == stride.ToScalar() )
				return $"Matrix [ {size.ToScalar()}, {size.GetElement( 1 )} ], row major";
			else if( 1 == stride.GetElement( 1 ) )
				return $"Matrix [ {size.ToScalar()}, {size.GetElement( 1 )} ], column major";
			else
				return $"Matrix [ {size.ToScalar()}, {size.GetElement( 1 )} ]";
		}
		return $"Size {size}, strides {stride}";
	}

	/// <summary>Shorter human-readable string</summary>
	public string description()
	{
		if( isVector )
			return $"Vector [ {size.x} ]";
		if( isMatrix )
			return $"Matrix [ {size.x}, {size.y} ]";
		if( size.w == 1 )
			return $"Tensor [ {size.x}, {size.y}, {size.z} ]";
		return $"Tensor {size}";
	}

	TensorShape( Vector128<int> size, Vector128<int> stride )
	{
		this.size = size;
		this.stride = stride;
	}

	/// <summary>Create dense row-major tensor shape of the specified size, without paddings</summary>
	public TensorShape( in Int128 size )
	{
		this.size = size;
		stride = new Int128( 1, size.x, size.x * size.y, size.x * size.y * size.z );
	}

	/// <summary>Create arbitrary shape</summary>
	public TensorShape( in Int128 size, in Int128 stride )
	{
		this.size = size;
		this.stride = stride;
	}

	/// <summary>Create from the values deserialized from PyTorch models</summary>
	public static TensorShape unpickle( object[] size, object[] stride )
	{
		int len = size.Length;
		if( len != stride.Length )
			throw new ArgumentException();

		switch( len )
		{
			case 1:
				return makeVector( (int)size[ 0 ], (int)stride[ 0 ] );
			case 2:
				return makeMatrix( (int)size[ 0 ], (int)size[ 1 ], (int)stride[ 0 ], (int)stride[ 1 ] );
			case 3:
				return makeTensor( (int)size[ 0 ], (int)size[ 1 ], (int)size[ 2 ],
					(int)stride[ 0 ], (int)stride[ 1 ], (int)stride[ 2 ] );
			case 4:
				return makeTensor4( (int)size[ 0 ], (int)size[ 1 ], (int)size[ 2 ], (int)size[ 3 ],
					(int)stride[ 0 ], (int)stride[ 1 ], (int)stride[ 2 ], (int)stride[ 3 ] );
			default:
				throw new NotImplementedException();
		}
	}

	static TensorShape makeVector( int ne, int nb )
	{
		Vector128<int> size = Vector128.Create( ne, 1, 1, 1 );
		if( nb != 1 )
			throw new ArgumentException();

		Vector128<int> stride = Vector128.Create( 1, ne, ne, ne );
		return new TensorShape( size, stride );
	}

	static TensorShape makeMatrix( int ne0, int ne1, int nb0, int nb1 )
	{
		if( nb1 == 1 )
		{
			// Transposing column major matrix into row major
			Vector128<int> size = Vector128.Create( ne1, ne0, 1, 1 );
			int complete = nb0 * ne0;
			Vector128<int> stride = Vector128.Create( 1, nb0, complete, complete );
			return new TensorShape( size, stride );
		}

		throw new NotImplementedException();
	}

	static TensorShape makeTensor( int ne0, int ne1, int ne2, int nb0, int nb1, int nb2 )
	{
		if( nb2 == 1 )
		{
			Vector128<int> size = Vector128.Create( ne2, ne1, ne0, 1 );
			Vector128<int> stride = Vector128.Create( 1, nb1, nb0, nb0 * ne0 );
			return new TensorShape( size, stride );
		}
		throw new NotImplementedException();
	}

	static TensorShape makeTensor4( int ne0, int ne1, int ne2, int ne3, int nb0, int nb1, int nb2, int nb3 )
	{
		if( nb3 == 1 )
		{
			Int128 size = new Int128( ne3, ne2, ne1, ne0 );
			TensorShape shape = new TensorShape( size );
			Int128 expectedStride = new Int128( nb3, nb2, nb1, nb0 );
			if( shape.stride != expectedStride )
				throw new ApplicationException();
			return shape;
		}
		throw new NotImplementedException();
	}

	/// <summary>Create tensor shape for a dense row major matrix; the first argument is width.</summary>
	public static TensorShape rowMajorMatrix( int ne0, int ne1 )
	{
		Vector128<int> size = Vector128.Create( ne0, ne1, 1, 1 );
		int complete = ne0 * ne1;
		Vector128<int> stride = Vector128.Create( 1, ne0, complete, complete );
		return new TensorShape( size, stride );
	}

	/// <summary>Create tensor shape for dense row major tensor</summary>
	public static TensorShape rowMajor( int x, int y = 1, int z = 1 )
	{
		if( x <= 0 || y <= 0 || z <= 0 )
			throw new ArgumentOutOfRangeException();

		Vector128<int> size = Vector128.Create( x, y, z, 1 );
		int p0 = x * y;
		int p1 = p0 * z;
		Vector128<int> stride = Vector128.Create( 1, x, p0, p1 );
		return new TensorShape( size, stride );
	}

	/// <summary>Count of elements in the tensor</summary>
	public int countElements() =>
		size.horizontalProduct();

	/// <summary>Count of rows in the tensor</summary>
	public int countRows() => size.y * size.z * size.w;

	/// <summary>Permute dimensions of the tensor</summary>
	/// <remarks>Allows to implement transpositions and similar view-only operations on tensors</remarks>
	public TensorShape permute( byte x, byte y, byte z, byte w )
	{
		if( ( x | y | z | w ) >= 4 )
			throw new ArgumentOutOfRangeException( "Permutation indices must be in [ 0 .. 3 ] range each" );
		if( x == y || x == z || x == w || y == z || y == w || z == w )
			throw new ArgumentException( "Permutation indices must be unique" );

		Vector128<int> size, stride;
		unsafe
		{
			fixed( TensorShape* rsi = &this )
			{
				int* pSize = (int*)( &rsi->size );
				int* pStride = (int*)( &rsi->stride );

				if( Avx2.IsSupported )
				{
					size = Sse2.LoadVector128( pSize );
					stride = Sse2.LoadVector128( pStride );

					Vector128<int> perm = Vector128.Create( x, y, z, w );
					size = Avx.PermuteVar( size.AsSingle(), perm ).AsInt32();
					stride = Avx.PermuteVar( stride.AsSingle(), perm ).AsInt32();
				}
				else
				{
					size = Vector128.Create( pSize[ x ], pSize[ y ], pSize[ z ], pSize[ w ] );
					stride = Vector128.Create( pStride[ x ], pStride[ y ], pStride[ z ], pStride[ w ] );
				}
			}
		}

		return new TensorShape( size, stride );
	}

	/// <summary>Reduce specified dimension of the tensor</summary>
	public TensorShape trim( byte dim, int newSize )
	{
		if( dim >= 4 )
			throw new ArgumentOutOfRangeException( "Trim dimension must be in [ 0 .. 3 ] range" );

		Span<int> size = stackalloc int[ 4 ];
		this.size.store( size );

		int currSize = size[ dim ];
		if( currSize > newSize )
		{
			// Can trim
			size[ dim ] = newSize;
			return new TensorShape( Int128.load( size ), stride );
		}
		else if( currSize == newSize )
		{
			// Already of the correct size, nothing to do
			return this;
		}
		else
			throw new ArgumentOutOfRangeException( $"Current dimention is {currSize}, smaller than the requested size {newSize}" );
	}

	/// <summary>Compare for equality</summary>
	public bool Equals( TensorShape other )
	{
		unsafe
		{
			fixed( TensorShape* pi = &this )
			{
				TensorShape* pa = &other;

				var i = Sse2.LoadVector128( (int*)&pi->size );
				var a = Sse2.LoadVector128( (int*)&pa->size );
				var xx = Sse2.Xor( a, i );

				i = Sse2.LoadVector128( (int*)&pi->stride );
				a = Sse2.LoadVector128( (int*)&pa->stride );
				xx = Sse2.Or( xx, Sse2.Xor( a, i ) );

				return Sse41.TestZ( xx, xx );
			}
		}
	}

	/// <summary>Compare for equality</summary>
	public static bool operator ==( in TensorShape a, in TensorShape b ) => a.Equals( b );

	/// <summary>Compare for inequality</summary>
	public static bool operator !=( in TensorShape a, in TensorShape b ) => !a.Equals( b );

	/// <summary>Compare for equality</summary>
	public override bool Equals( [NotNullWhen( true )] object? obj )
	{
		if( obj is TensorShape a )
			return Equals( a );
		return false;
	}

	/// <summary>Compute a hash code</summary>
	public override int GetHashCode() => HashCode.Combine( size, stride );
}