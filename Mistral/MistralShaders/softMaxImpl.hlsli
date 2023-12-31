#include "groupReduce.hlsli"

[numthreads( THREADS, 1, 1 )]
void main( uint3 group : SV_GroupID, uint thread : SV_GroupIndex )
{
	uint rdi = dot( group, strides );
	const uint rdiEnd = rdi + width;
	const uint rdiStart = rdi + thread;

	// Find the maximum value in the input vector
	const float FLT_MAX = 3.402823466e+38F;
	float ax = -FLT_MAX;
	for( rdi = rdiStart; rdi < rdiEnd; rdi += THREADS )
		ax = max( ax, load( tensor, rdi ) );
	horizontalMax( thread, ax );
	if( 0 == thread )
		reductionBuffer[ 0 ] = ax * initialMul;
	GroupMemoryBarrierWithGroupSync();
	const float maxVal = reductionBuffer[ 0 ];

	// Compute sum of exponentials with the log-sum-exp trick
	float sumExp = 0.0;
	for( rdi = rdiStart; rdi < rdiEnd; rdi += THREADS )
		sumExp += exp( load( tensor, rdi ) * initialMul - maxVal );
	horizontalSum( thread, sumExp );
	if( 0 == thread )
		reductionBuffer[ 0 ] = 1.0 / sumExp;
	GroupMemoryBarrierWithGroupSync();

	// Compute softmax probabilities
	const float mul = reductionBuffer[ 0 ];
	for( rdi = rdiStart; rdi < rdiEnd; rdi += THREADS )
	{
		float f = load( tensor, rdi ) * initialMul;
		f = exp( f - maxVal ) * mul;
		store( tensor, rdi, f );
	}
}