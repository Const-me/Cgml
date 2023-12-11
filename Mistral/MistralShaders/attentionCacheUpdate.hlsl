// Store input to the attention cache curcular buffer
#include "miscUtils.hlsli"

Tensor input : register( t0 );
OutputTensor cached : register( u0 );

cbuffer Constants: register( b0 )
{
	// .zw strides of the input tensor
	uint2 inputStride: packoffset( c0.x );
	// .zw strides of the cache tensor
	uint2 cacheStride: packoffset( c0.z );
	// First slice to load from the input tensor
	uint firstSliceLoad: packoffset( c1.x );
	// First slice to store, they are wrapped
	uint firstSliceStore: packoffset( c1.y );
	// Size of the window = wrapping
	uint slidingWindow: packoffset( c1.z );
	// Count of elements in XY slice of both tensor
	uint rowLength: packoffset( c1.w );
}

#ifndef THREADS
static const uint THREADS = 256;
#endif

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	uint destIndex = group.x + firstSliceStore;
	destIndex = destIndex % slidingWindow;	// Wrap destination slice

	// x = load index, y = store index
	uint2 it;
	it.x = dot( uint2( group.x + firstSliceLoad, group.y ), inputStride );
	it.y = dot( uint2( destIndex, group.y ), cacheStride );

	const uint rsiEnd = it.x + rowLength;
	for( it += thread; it.x < rsiEnd; it += THREADS )
	{
		cached[ it.y ] = input[ it.x ];
	}
}