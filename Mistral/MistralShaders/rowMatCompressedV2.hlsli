// Optimized implementation of vector * compressed matrix compute shader
// Specifically, for BCML2 codec this version is about 3.93x faster on my nVidia 1080 Ti
// The compression algorithm is selected with `BCML_CODEC` macro
#include "miscUtils.hlsli"

Tensor tensor : register( t0 );
ByteAddressBuffer mat: register( t1 );
OutputTensor result : register( u0 );

cbuffer Constants: register( b0 )
{
	uint rowLength: packoffset( c0.x );
	uint rowsCount: packoffset( c0.y );
	uint2 arg0Strides: packoffset( c0.z );
	uint2 resultStrides: packoffset( c1.x );
	uint matrixStride: packoffset( c1.z );
}

groupshared float rowBuffer[ THREADS ];

inline void loadRow( uint rsi, uint rdi )
{
	rowBuffer[ rdi ] = load( tensor, rsi );
	GroupMemoryBarrierWithGroupSync();
}

#include "bcml.hlsli"

static const uint THREADS_Y = THREADS / BCML_PANEL_HEIGHT;

[numthreads( BCML_PANEL_HEIGHT, THREADS_Y, 1 )]
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
			for( uint j = 0; j < THREADS; )
				computeCompressedBlock( j, decoder, acc );
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
			const uint remainderBatchAligned = roundDownToBlock( remainderBatch );

			// Process complete blocks in the remainder
			uint j = 0;
			while( j < remainderBatchAligned )
				computeCompressedBlock( j, decoder, acc );

			// Process the final, incomplete block in the remainder
			for( ; j < remainderBatch; j++ )
				accumulateRemainderElement( j, decoder, acc );
		}
		GroupMemoryBarrierWithGroupSync();
	}

	if( thread >= groupCountRows )
		return;

	// Store output slice of length `groupCountRows`
	uint rdi = dot( group.yz, resultStrides );
	rdi += groupFirstRow;
	rdi += thread;
	store( result, rdi, acc );
}