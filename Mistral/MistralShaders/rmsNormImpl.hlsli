static const uint THREADS = 64;
#include "groupReduce.hlsli"

[numthreads( THREADS, 1, 1 )]
void main( uint3 group : SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint rsiStart = hadd( group.xy * inputStrides.yz );
	const uint rsiEnd = rsiStart + inputSize.x;
	// Compute sum of squares of the row
	float sum = 0;
	uint rsi;
	for( rsi = rsiStart + thread; rsi < rsiEnd; rsi += THREADS )
	{
		float f = load( tensor, rsi );
		f = f * f;
		sum += f;
	}
	// Reduce to scalar
	horizontalSum( thread, sum );

	// On the first thread compute rsqrt
	if( 0 == thread )
	{
		// Compute average from the sum
		sum /= (float) (int) ( inputSize.x );

		// rsqrt( x + eps )
		sum = rsqrt( sum + epsilon );

		// Save to groupshared buffer for broadcasting
		reductionBuffer[ 0 ] = sum;
	}

	// Broadcast that scaling float from the groupshared buffer
	GroupMemoryBarrierWithGroupSync();
	const float mul = reductionBuffer[ 0 ];

	uint2 it;
	it.x = rsiStart;
	it.y = 0;
	for( it += thread; it.x < rsiEnd; it += THREADS )
	{
		float f = load( tensor, it.x );
		f *= mul;
#if CUDA_COMPAT
		// f = roundFp16Nearest( f );
#endif
		f *= load( weights, it.y );
		storeResult( it.x, f );
	}
}