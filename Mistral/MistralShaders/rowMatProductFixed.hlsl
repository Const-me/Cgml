// Multiply rows of the tensor by a matrix: [ 4096, y, z ] * [ 4096, r ] = [ r, y, z ]
// In practice, r = 32000 (at least for official models, AFAIK fine-tuned ones may add moar tokens),
// y varies depending on the input like 10-88
// z = 1
#include "miscUtils.hlsli"

Tensor tensor : register( t0 );
Tensor mat : register( t1 );
OutputTensor result : register( u0 );

cbuffer Constants : register( b0 )
{
	uint2 arg0Strides : packoffset( c0.x );
	uint2 resultStrides : packoffset( c0.z );
	uint rowsCount : packoffset( c1.x );
	uint groupOffset: packoffset( c1.y );
	uint2 arg0SizeYZ : packoffset( c1.z );
}

static const uint THREADS = 64;

// Our goal is saving memory bandwidth to the second argument
// That matrix comes from the model. It's uncompressed because the quality is sensitive to it.
// The size is [ 4096, 32000 ] elements = 250 MB of VRAM

static const uint rowLength = 4096;
groupshared float matrixRowBuffer[ rowLength ]; //< 16kb of local memory

#include "groupReduce.hlsli"	//< 256 bytes of local memory in another one

[numthreads( THREADS, 1, 1 )]
void main( uint3 group : SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint groupFirstRow = ( group.x + groupOffset ) * THREADS;
	const uint groupCountRows = min( THREADS, rowsCount - groupFirstRow );
	uint rsiMatrix = groupFirstRow * rowLength;
	
	uint rdiBase = groupFirstRow;
	for( uint r = 0; r < groupCountRows; r++, rdiBase++ )
	{
		// Load complete row from matrix into the group shared buffer
		for( uint i = thread; i < rowLength; i += THREADS )
			matrixRowBuffer[ i ] = load( mat, rsiMatrix + i );
		// Increment matrix load index to point to the next row
		rsiMatrix += rowLength;
		GroupMemoryBarrierWithGroupSync();
		
		// Load complete source tensor, and compute stuff
		uint rsiLayer2 = 0;
		uint rdiLayer2 = rdiBase;
		for( uint2 it2 = 0; it2.y < arg0SizeYZ.y; it2.y++, rsiLayer2 += arg0Strides.y, rdiLayer2 += resultStrides.y )
		{
			uint rsiLayer1 = rsiLayer2;
			uint rdiLayer1 = rdiLayer2;
			for( it2.x = 0; it2.x < arg0SizeYZ.x; it2.x++, rsiLayer1 += arg0Strides.x, rdiLayer1 += resultStrides.x )
			{
				float acc = 0;
				for( uint i = thread; i < rowLength; i += THREADS )
					acc = mad( matrixRowBuffer[ i ], load( tensor, rsiLayer1 + i ), acc );
				horizontalSum( thread, acc );
				if( 0 == thread )
				{
					store( result, rdiLayer1, acc );
				}
			}
		}

		GroupMemoryBarrierWithGroupSync();
	}
}