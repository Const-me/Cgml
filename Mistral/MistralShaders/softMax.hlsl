// Compute softmax function for every row of the tensor, in-place
#include "miscUtils.hlsli"

OutputTensor tensor : register( u0 );

cbuffer Constants: register( b0 )
{
	uint width: packoffset( c0.x );
	uint3 strides: packoffset( c0.y );
}

#ifndef THREADS
#define THREADS 32
#endif

#include "groupReduce.hlsli"

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	uint rdi = dot( group, strides );
	const uint rdiEnd = rdi + width;
	const uint rdiStart = rdi + thread;

	float sumExp = 0;
	for( rdi = rdiStart; rdi < rdiEnd; rdi += THREADS )
		sumExp += exp( load( tensor, rdi ) );

	horizontalSum( thread, sumExp );

	if( 0 == thread )
		reductionBuffer[ 0 ] = 1.0 / sumExp;
	GroupMemoryBarrierWithGroupSync();

	const float mul = reductionBuffer[ 0 ];
	for( rdi = rdiStart; rdi < rdiEnd; rdi += THREADS )
	{
		float f = load( tensor, rdi );
		f = exp( f ) * mul;
		store( tensor, rdi, f );
	}
}