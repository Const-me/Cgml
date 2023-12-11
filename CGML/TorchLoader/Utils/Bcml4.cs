namespace Torch;
using Cgml;
using System.Diagnostics;

/// <summary>Utility class to reshape GPTQ tensors into BCML4 data format</summary>
/// <remarks>The source data must use 4-bit quantization, with block size 128 elements</remarks>
public static class Bcml4
{
	const int panelHeight = 64;

	static int elementsPerRow( int width )
	{
		int zeros = ( width + 1023 ) / 1024;
		int scales = ( width + 255 ) / 256;
		int weights = ( width + 7 ) / 8;
		int eltsPerRow = zeros + scales + weights;
		return eltsPerRow;
	}

	static int bytesPerPanel( int width )
	{
		int elts = elementsPerRow( width );
		return elts * ( panelHeight * 4 );
	}

	static int panelsCount( int y ) =>
		( y + panelHeight - 1 ) / panelHeight;

	/// <summary>Compute count of bytes required for BCML3 tensor of the specified size</summary>
	public static int tensorByteWidth( in Int128 size )
	{
		int bytesPanel = bytesPerPanel( size.x );
		int panels = panelsCount( size.y );
		return bytesPanel * panels * size.z * size.w;
	}

	/// <summary>Initialize tensor shape of the compressed tensor</summary>
	public static TensorShape tensorShape( in Int128 size )
	{
		int bytesPanel = bytesPerPanel( size.x );
		int panels = panelsCount( size.y );
		int bytesLayer = bytesPanel * panels;

		Int128 strides = new Int128( 0, bytesPanel, bytesLayer, bytesLayer * size.z );
		return new TensorShape( size, strides );
	}

	static uint loadZero( MatrixView<uint> qzeros, int x, int y )
	{
		uint elt = qzeros[ x / 8, y ];
		elt >>= ( x % 8 ) * 4;
		return elt & 0xFu;
	}

	static void gatherZeros( Span<uint> rdi, MatrixView<uint> qzeros, int xOut, int yBaseOut, int height )
	{
		// Need this clamp because sometimes width of the output matrix ain't multiple of 1024
		// The last qzeros is incomplete in that case, the uint contains less than 8 4-bit values
		int innerBlocks = Math.Min( 8, (int)qzeros.size.y - xOut );
		Debug.Assert( innerBlocks > 0 );

		// 1 iteration = 1 output element 
		for( int i = 0; i < height; i++ )
		{
			uint res = 0;
			int y = yBaseOut + i;

			for( int j = 0; j < innerBlocks; j++ )
			{
				uint e = loadZero( qzeros, y, xOut + j );
				e <<= j * 4;
				res |= e;
			}
			rdi[ i ] = res;
		}
	}

	/// <summary>Reshape matrix into BCML4</summary>
	/// <remarks>This does not recompress the matrix: no data is lost in the process</remarks>
	public static void reshape( Span<uint> bc4, MatrixView<uint> qweight, MatrixView<ushort> scales, MatrixView<uint> qzeros, ReadOnlySpan<uint> index )
	{
		Int128 destSize = new Int128( (int)qweight.size.y * 8, (int)qweight.size.x, 1, 1 );
		Debug.Assert( bc4.Length == tensorByteWidth( destSize ) / 4 );

		int rows = (int)qweight.size.x;
		int panels = panelsCount( rows );
		int rowLengthIntegers = (int)qweight.size.y;

		// 1 iteration = 1 panel = [ destSize.x, panelHeight ] sub-matrix
		for( int p = 0; p < panels; p++ )
		{
			int yBase = p * panelHeight;
			int height = rows - yBase;
			if( height >= panelHeight )
				height = panelHeight;
			else
				bc4.Fill( 0 );

			// 1 iteration = [ 8, panelHeight ]  slice of the current panel
			for( int c = 0; c < rowLengthIntegers; c++ )
			{
				if( 0 == ( c % 128 ) )
				{
					// First column of a 1024-long sequence, pack quantized zeros
					gatherZeros( bc4, qzeros, c / 16, yBase, height );
					bc4 = bc4.Slice( panelHeight );
				}

				if( 0 == ( c % 32 ) )
				{
					// First column of 256-long sequence, pack 2 (or sometimes, very rarely, 1) column with the scales
					int scalesColumn = c / 16;

					// First column of the two
					ReadOnlySpan<ushort> rsiF16 = scales.rowSpan( yBase, scalesColumn, height );
					for( int i = 0; i < height; i++ )
						bc4[ i ] = rsiF16[ i ];

					if( c + 16 < rowLengthIntegers )
					{
						// Second column of the two
						rsiF16 = scales.rowSpan( yBase, scalesColumn + 1, height );
						for( int i = 0; i < height; i++ )
						{
							uint src = rsiF16[ i ];
							src <<= 16;
							bc4[ i ] |= src;
						}
					}

					bc4 = bc4.Slice( panelHeight );
				}

				{
					// Copy quantized weights
					ReadOnlySpan<uint> rsi = qweight.rowSpan( yBase, c, height );
					rsi.CopyTo( bc4 );
					bc4 = bc4.Slice( panelHeight );
				}
			}
		}

		Debug.Assert( bc4.IsEmpty );
	}
}