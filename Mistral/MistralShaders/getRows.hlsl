// Extract rows from the source tensor. Row indices to extract are taken from the second input tensor.

#include "miscUtils.hlsli"

// tok_embeddings FP16 tensor, 4096*32000 row major
Tensor source : register( t0 );
// Input tensor, w*h row major
Buffer<uint> tokens: register( t1 );
// Output FP16 tensor, 4096*w*h
OutputTensor result : register( u0 );

#ifndef THREADS
#define THREADS 256
#endif

cbuffer Constants: register( b0 )
{
	// First column from the source tensor to use
	uint firstColumn : packoffset( c0.x );
	// Distance between rows in the input tensor
	uint inputStride: packoffset( c0.y );
	// Count of rows in the source tensor
	uint sourceHeight: packoffset( c0.z );
	// Length of the rows to duplicate
	uint rowLength: packoffset( c0.w );
	// Output tensor stride
	uint outputStride: packoffset( c1.x );
}

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	// Row index to duplicate
	const uint idx = tokens[ firstColumn + group.x + group.y * inputStride ];

	if( idx < sourceHeight )
	{
		// Initialize iterators with [ source index, destination index ]
		uint2 it;
		it.x = rowLength * idx;
		const uint rsiEnd = it.x + rowLength;
		it.y = rowLength * ( group.x + group.y * outputStride );

		// memcpy() equivalent, fully coalesced loads and stores
		// We don't need loadElement / storeElement functions even for BF16 because not doing any math on these numbers
		for( it += thread; it.x < rsiEnd; it += THREADS )
			result[ it.y ] = source[ it.x ];
	}
	else
	{
		uint rdi = rowLength * ( group.x + group.y * outputStride );
		const uint rdiEnd = rdi + rowLength;
		for( rdi += thread; rdi < rdiEnd; rdi += THREADS )
			result[ rdi ] = 0;
	}
}