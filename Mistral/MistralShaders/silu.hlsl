// Compute x = silu( x ) * y, element-wise
#ifndef THREADS
#define THREADS 256
#endif

#include "miscUtils.hlsli"

OutputTensor tensor : register( u0 );
Tensor mul : register( t0 );

cbuffer Constants: register( b0 )
{
	uint width: packoffset( c0.x );
	uint3 strides: packoffset( c0.y );
}

inline float silu( const float v )
{
	// https://en.wikipedia.org/wiki/Sigmoid_function
	const float exponent = exp( v );
	const float sigmoid = exponent / ( exponent + 1 );
	return v * sigmoid;
}

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	uint rdi = dot( group, strides );
	const uint rdiEnd = rdi + width;

	for( rdi += thread; rdi < rdiEnd; rdi += THREADS )
	{
		float f = load( tensor, rdi );
		f = silu( f );
		f *= load( mul, rdi );
		store( tensor, rdi, f );
	}
}