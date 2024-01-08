// Copy transposed tensor into row major destination of the same size
#include "miscUtils.hlsli"
OutputTensor result : register( u0 );
Tensor source : register( t0 );

cbuffer Constants: register( b0 )
{
	uint4 inputStrides: packoffset( c0 );
	uint width: packoffset( c1.x );
	uint3 outputStrides: packoffset( c1.y );
}

static const uint THREADS = 64;

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	uint2 it;
	it.x = dot( group, inputStrides.yzw );
	it.y = dot( group, outputStrides );

	const uint rdiEnd = it.y + width;
	const uint2 inc = uint2( inputStrides.x * THREADS, THREADS );
	it.x += thread * inputStrides.x;
	it.y += thread;

	for( ; it.y < rdiEnd; it += inc )
		result[ it.y ] = source[ it.x ];
}