// Compute the logarithm of the softmax function. In principle, log_softmax(x) = log(softmax(x)) but using a more accurate implementation.
#include "miscUtils.hlsli"

OutputTensor tensor : register( u0 );

cbuffer Constants: register( b0 )
{
	uint width: packoffset( c0.x );
	uint3 strides: packoffset( c0.y );
}

#ifndef THREADS
#define THREADS 256
#endif

#include "groupReduce.hlsli"

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	uint rdi = dot( group, strides );
	const uint rdiEnd = rdi + width;
	const uint rdiStart = rdi + thread;

	// First pass over the input, find maximum
	const float FLT_MAX = 3.402823466e+38F;
	float tmp = -FLT_MAX;
	for( rdi = rdiStart; rdi < rdiEnd; rdi += THREADS )
		tmp = max( tmp, load( tensor, rdi ) );

	horizontalMax( thread, tmp );
	if( 0 == thread )
		reductionBuffer[ 0 ] = tmp;
	GroupMemoryBarrierWithGroupSync();

	// Second pass, compute sum( exp( tensor[ i ] - maxVal ] ) )
	const float maxVal = reductionBuffer[ 0 ];
	tmp = 0.0;
	for( rdi = rdiStart; rdi < rdiEnd; rdi += THREADS )
		tmp += exp( load( tensor, rdi ) - maxVal );
	horizontalSum( thread, tmp );
	if( 0 == thread )
		reductionBuffer[ 0 ] = log( tmp );
	GroupMemoryBarrierWithGroupSync();
	
	// Third and final pass, transform these numbers
	const float sub = maxVal + reductionBuffer[ 0 ];
	for( rdi = rdiStart; rdi < rdiEnd; rdi += THREADS )
	{
		float f = load( tensor, rdi );
		f -= sub;
		store( tensor, rdi, f );
	}
}