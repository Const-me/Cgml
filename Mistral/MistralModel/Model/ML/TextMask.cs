namespace Mistral.Model;
using Cgml;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>Utility class which implements 2D array of bits in system memory</summary>
sealed class TextMask
{
	int x, y;
	int xElements;
	uint[] array = new uint[ 0 ];

	/// <summary>Resize, and fill with zeros</summary>
	public void resize( int x, int y )
	{
		if( x < 0 || y < 0 ) throw new ArgumentOutOfRangeException();
		xElements = ( ( x + 31 ) / 32 );
		int elements = xElements * y;
		if( array.Length < elements )
			array = new uint[ elements ];

		Span<uint> span = array;
		span = span.Slice( 0, elements );
		span.Fill( 0 );

		this.x = x;
		this.y = y;
	}

	/// <summary>Set specified row of the bitmap to the result of following comparison: source != cmp</summary>
	public void setNotEqual( int y, ReadOnlySpan<int> source, int cmp )
	{
		if( y < 0 || y >= this.y )
			throw new ArgumentOutOfRangeException();
		if( source.Length > x )
			throw new ArgumentOutOfRangeException();

		int lenBlocks = ( source.Length / 32 );
		Vector256<int> cmpVec = Vector256.Create( cmp );
		Vector256<int> perm = Vector256.Create( 0, 4, 1, 5, 2, 6, 3, 7 );
		unsafe
		{
			fixed( int* pinnedSource = source )
			fixed( uint* pinnedDest = array )
			{
				uint* rdi = pinnedDest + y * xElements;
				int* rsiEndAligned = pinnedSource + lenBlocks * 32;
				int* rsiEnd = pinnedSource + source.Length;
				int* rsi = pinnedSource;
				for( ; rsi < rsiEndAligned; rdi++ )
				{
					Vector256<int> e0 = Avx2.CompareEqual( cmpVec, Avx.LoadVector256( rsi ) );
					Vector256<int> e1 = Avx2.CompareEqual( cmpVec, Avx.LoadVector256( rsi + 8 ) );
					Vector256<int> e2 = Avx2.CompareEqual( cmpVec, Avx.LoadVector256( rsi + 16 ) );
					Vector256<int> e3 = Avx2.CompareEqual( cmpVec, Avx.LoadVector256( rsi + 24 ) );
					rsi += 32;

					Vector256<short> e01 = Avx2.PackSignedSaturate( e0, e1 );
					Vector256<short> e23 = Avx2.PackSignedSaturate( e2, e3 );
					Vector256<sbyte> res = Avx2.PackSignedSaturate( e01, e23 );
					res = Avx2.PermuteVar8x32( res.AsInt32(), perm ).AsSByte();

					uint bmp = unchecked((uint)Avx2.MoveMask( res ));
					*rdi = ~bmp;
				}

				if( rsi >= rsiEnd )
					return;

				uint last = 0;
				uint bit = 1;
				for( ; rsi < rsiEnd; rsi++, bit += bit )
				{
					if( *rsi != cmp )
						last |= bit;
				}
				*rdi = last;
			}
		}
	}

	/// <summary>Get bit with 2D index</summary>
	public bool this[ int x, int y ]
	{
		get
		{
			if( x >= 0 && y >= 0 && x < this.x && y < this.y )
			{
				uint elt = array[ ( y * xElements ) + ( x / 32 ) ];
				uint bit = 1u << ( x % 32 );
				return 0 != ( elt & bit );
			}
			throw new IndexOutOfRangeException();
		}
	}

	/// <summary>Upload data into an immutable tensor with 32-bit elements, 1 bit per element</summary>
	public iTensor upload( iDevice device )
	{
		ReadOnlySpan<uint> span = array;
		span = span.Slice( 0, xElements * y );
		ReadOnlySpan<byte> bytes = MemoryMarshal.Cast<uint, byte>( span );

		sTensorDesc desc = new sTensorDesc()
		{
			shape = TensorShape.rowMajorMatrix( xElements, y ),
			dataType = eDataType.U32,
			usage = eBufferUse.Immutable,
			layout = eTensorLayout.Dense
		};

		return device.uploadImmutableTensor( desc, bytes );
	}
}