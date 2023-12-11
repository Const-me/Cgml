// Multiply rows of the tensor by a matrix, i.e. [ x, y, z ] * [ x, r ] = [ r, y, z ]
// Each thread group cooperatively computes a small slice of the output row
#include "miscUtils.hlsli"

Tensor tensor : register( t0 );
Tensor mat : register( t1 );
OutputTensor result : register( u0 );

cbuffer Constants: register( b0 )
{
	uint rowLength: packoffset( c0.x );
	uint rowsCount: packoffset( c0.y );
	uint2 arg0Strides: packoffset( c0.z );
	uint2 resultStrides: packoffset( c1.x );
}

#ifndef THREADS
static const uint THREADS = 512;
#endif

groupshared float rowBuffer[ THREADS ];

inline void loadRow( uint rsi, uint rdi )
{
	rowBuffer[ rdi ] = load( tensor, rsi );
	GroupMemoryBarrierWithGroupSync();
}

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	uint rsiRow = dot( group.yz, arg0Strides );

	const uint groupFirstRow = group.x * THREADS;
	const uint groupCountRows = min( THREADS, rowsCount - groupFirstRow );
	uint rsiMatrix = ( groupFirstRow + thread ) * rowLength;

	const uint completeBatches = rowLength / THREADS;
	const uint remainderBatch = rowLength % THREADS;

	rsiRow += thread;
	float acc = 0;
	uint i;
	for( i = 0; i < completeBatches; i++ )
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
			{
				float r = rowBuffer[ j ];
				float m = load( mat, rsiMatrix );
				rsiMatrix++;
				acc = mad( r, m, acc );
			}
		}

		GroupMemoryBarrierWithGroupSync();
	}

	[branch]
	if( 0 != remainderBatch )
	{
		// We have an incomplete partial tile
		[branch]
		if( thread < remainderBatch )
			rowBuffer[ thread ] = load( tensor, rsiRow );
		GroupMemoryBarrierWithGroupSync();

		[branch]
		if( thread < groupCountRows )
		{
			for( uint j = 0; j < remainderBatch; j++ )
			{
				float r = rowBuffer[ j ];
				float m = load( mat, rsiMatrix );
				rsiMatrix++;
				acc = mad( r, m, acc );
			}
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