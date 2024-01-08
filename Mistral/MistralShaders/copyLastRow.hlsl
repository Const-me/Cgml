// Copy the last x-directed row of the tensor into y=0 position
#include "miscUtils.hlsli"

OutputTensor result : register( u0 );

static const uint THREADS = 265;

cbuffer Constants: register( b0 )
{
	uint4 inputStrides: packoffset( c0 );
	uint width: packoffset( c1.x );
	uint lastRowIndex: packoffset( c1.y );
}

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	uint2 it;
	it.x = dot( group.yz, inputStrides.zw );
	it.y = lastRowIndex * inputStrides.y + it.x;
	const uint rdiEnd = it.x + width;

	for( it += thread; it.x < rdiEnd; it += THREADS )
		result[ it.x ] = result[ it.y ];
}