// The final softmax does very long tensors, size.x = 32000, and has a pre-scaling with a constant
#include "miscUtils.hlsli"

OutputTensor tensor : register( u0 );

cbuffer Constants: register( b0 )
{
	uint width: packoffset( c0.x );
	uint3 strides: packoffset( c0.y );
	float initialMul : packoffset( c1.x );
}

#ifndef THREADS
#define THREADS 1024
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
		sumExp += exp( load( tensor, rdi ) * initialMul );

	horizontalSum( thread, sumExp );

	if( 0 == thread )
		reductionBuffer[ 0 ] = 1.0 / sumExp;
	GroupMemoryBarrierWithGroupSync();

	const float mul = reductionBuffer[ 0 ];
	for( rdi = rdiStart; rdi < rdiEnd; rdi += THREADS )
	{
		float f = load( tensor, rdi );
		f = exp( f * initialMul ) * mul;
		store( tensor, rdi, f );
	}
}