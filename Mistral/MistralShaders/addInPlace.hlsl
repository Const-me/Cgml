// Element-wise <c>result += arg0</c> for two tensors of the same size and shape
#ifndef THREADS
#define THREADS 128
#endif

#include "miscUtils.hlsli"

OutputTensor result : register( u0 );
Tensor arg0 : register( t0 );

cbuffer Constants: register( b0 )
{
	uint width: packoffset( c0.x );
	uint3 strides: packoffset( c0.y );
}

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	uint rdi = dot( group, strides );
	const uint rdiEnd = rdi + width;

	for( rdi += thread; rdi < rdiEnd; rdi += THREADS )
	{
		float f = load( result, rdi );
		f += load( arg0, rdi );
		store( result, rdi, f );
	}
}