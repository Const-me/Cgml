// Initial implementation of vector * compressed matrix compute shader
// The compression algorithm is selected with `BCML_CODEC` macro
Buffer<float> tensor: register( t0 );
ByteAddressBuffer mat: register( t1 );
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint rowLength: packoffset( c0.x );
	uint rowsCount: packoffset( c0.y );
	uint2 arg0Strides: packoffset( c0.z );
	uint2 resultStrides: packoffset( c1.x );
	uint matrixStride: packoffset( c1.z );
}

static const uint THREADS = 512;

groupshared float rowBuffer[ THREADS ];

#include "miscUtils.hlsli"
#include "bcml.hlsli"

inline void loadRow( uint rsi, uint rdi )
{
	rowBuffer[ rdi ] = tensor[ rsi ];
	GroupMemoryBarrierWithGroupSync();
}

static const uint THREADS_Y = THREADS / BCML_PANEL_HEIGHT;

[ numthreads( BCML_PANEL_HEIGHT, THREADS_Y, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex, uint3 thread3 : SV_GroupThreadID )
{
	uint rsiRow = dot( group.yz, arg0Strides );

	const uint groupFirstRow = group.x * THREADS;
	const uint groupCountRows = min( THREADS, rowsCount - groupFirstRow );

	// matrixStride is the count of bytes in compressed panels
	// Each thread group of this shader processes multiple panels, that's why 2D thread indices
	const uint idxPanel = group.x * THREADS_Y + thread3.y;
	uint rsiMatrix = mad( idxPanel, matrixStride, thread3.x * 4 );
	DecoderState decoder = createDecoderState( rsiMatrix );

	const uint completeBatches = rowLength / THREADS;
	const uint remainderBatch = rowLength % THREADS;

	rsiRow += thread;
	float acc = 0;
	for( uint i = 0; i < completeBatches; i++ )
	{
		// Load THREADS elements from first tensor into group shared buffer
		loadRow( rsiRow, thread );
		rsiRow += THREADS;

		// Update accumulators with the product of tiles
		// `tensor` tile is of length THREADS
		// `mat` tile is of size [ THREADS, groupCountRows ]
		[ branch ]
		if( thread < groupCountRows )
		{
			for( uint j = 0; j < THREADS; j++ )
				computeCompressedProduct( i, j, decoder, acc );
		}

		GroupMemoryBarrierWithGroupSync();
	}

	[branch]
	if( 0 != remainderBatch )
	{
		// We have an incomplete partial tile
		[branch]
		if( thread < remainderBatch )
			rowBuffer[ thread ] = tensor[ rsiRow ];
		GroupMemoryBarrierWithGroupSync();

		[branch]
		if( thread < groupCountRows )
		{
			float2 bcHeader;
			uint bcBlock;
			for( uint j = 0; j < remainderBatch; j++ )
				computeCompressedProduct( completeBatches, j, decoder, acc );
		}

		GroupMemoryBarrierWithGroupSync();
	}

	if( thread >= groupCountRows )
		return;

	// Store output slice of length `groupCountRows`
	acc = roundFp16Nearest( acc );
	uint rdi = dot( group.yz, resultStrides );
	rdi += groupFirstRow;
	rdi += thread;
	result[ rdi ] = acc;
}