// <c>torch.topk</c> masking the rest of values with -inf, then <c>nn.functional.softmax</c>, and finally <c>torch.multinomial</c>
#include "miscUtils.hlsli"

Tensor tensor : register( t0 );
RWBuffer<uint> tempBuffer: register( u0 );
RWBuffer<uint> result: register( u1 );

cbuffer Constants: register( b0 )
{
	// Row width of the input tensor
	uint width: packoffset( c0.x );
	// Minimum count of the elements to sample. Actual count gonna be slightly more, because duplicate values in the input tensor.
	uint topK : packoffset( c0.y );
	// A random number in [ 0.0 .. 1.0 ] interval
	float rand : packoffset( c0.z );
}

// Count of threads in the shader, also maximum count of values to use for the sampling
static const uint THREADS = 1024;

groupshared uint prefixLocal[ THREADS ];
groupshared uint tempScalarBuffer;

// Compute exclusive prefix sum of the specified value across threads of this group
// The return values are [ exclusive sum for the current thread, total sum of all threads ]
inline uint2 exclusivePrefixSum( const uint thread, in uint val )
{
	// Keep the original value
	const uint orig = val;

	// Compute inclusive prefix sum, the algorithm is ported from that picture:
	// https://en.wikipedia.org/wiki/Prefix_sum#/media/File:Hillis-Steele_Prefix_Sum.svg
	for( uint i = 1; i < THREADS; i += i )
	{
		prefixLocal[ thread ] = val;
		GroupMemoryBarrierWithGroupSync();

		[ branch ]
		if( thread >= i )
		{
			// First iteration:  i = 1, ( thread >= i ) is true for the threads [ 1 .. THREADS - 1 ]
			// Second iteration: i = 2, ( thread >= i ) is true for the threads [ 2 .. THREADS - 1 ]
			// Third iteration:  i = 4, ( thread >= i ) is true for the threads [ 4 .. THREADS - 1 ]
			// Last iteration: i = THREADS / 2, ( thread >= i ) is true for the threads [ THREADS / 2 .. THREADS - 1 ]
			val += prefixLocal[ thread - i ];

			[ branch ]
			if( i == ( THREADS / 2 ) && thread == ( THREADS - 1 ) )
			{
				// This is the last iteration of the loop, and we're running on the last thread of the group
				// Broadcast the total sum, which is the `val` number on the last thread of the group
				tempScalarBuffer = val;
			}
		}
		GroupMemoryBarrierWithGroupSync();	
	}

	uint2 result;
	// Exclusive prefix sum = ( inclusive prefix sum ) - ( the original value )
	result.x = val - orig;
	// Total sum = inclusive sum on the last thread of the group
	result.y = tempScalarBuffer;

	return result;
}

// Load a source value, convert back to FP16, and clamp into [ 0 .. +INF ] interval
inline uint loadSource( uint rsi )
{
	float f = tensor[ rsi ];
	uint u = f32tof16( f );
	if( 0 != ( u & 0x8000 ) )
	{
		// Get rid of the negative probabilities. Not necessarily an error.
		// -0.0f is a perfectly cromulent probability, despite technically a negative number
		u = 0;
	}
	return u;
}

inline float upcast( uint u )
{
	// https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/f16tof32
	// Converts the float16 stored in the low half of the uint to a float
	// Therefore, no need to mask away the higher 2 bytes
	return f16tof32( u );
}

// This shader is dispatched with a single thread group, we don't care about local memory usage,
// as long as it doesn't exceed the 32Kb limit of D3D11
groupshared uint probsLocal[ THREADS ];
groupshared float softMaxBuffer[ THREADS ];

// Compute horisontal sum of the numbers. The result is only correct on the thread #0 of the group.
inline void horizontalSum( const uint thread, inout float sum )
{
	softMaxBuffer[ thread ] = sum;
	for( uint i = THREADS / 2; i > 1; i /= 2 )
	{
		GroupMemoryBarrierWithGroupSync();
		if( thread < i )
		{
			sum += softMaxBuffer[ thread + i ];
			softMaxBuffer[ thread ] = sum;
		}
	}
	GroupMemoryBarrierWithGroupSync();
	if( 0 == thread )
		sum += softMaxBuffer[ 1 ];
}

[ numthreads( THREADS, 1, 1 ) ]
void main( uint thread : SV_GroupIndex )
{
	uint i;
	// Write zeros to the temp buffer
	for( i = thread; i < 0x8000; i += THREADS )
		tempBuffer[ i ] = 0;
	AllMemoryBarrierWithGroupSync();

	// Count values using InterlockedAdd intrinsic
	for( i = thread; i < width; i += THREADS )
	{
		uint val = loadSource( i );
		uint bucket = 0x7FFF - val;
		InterlockedAdd( tempBuffer[ bucket ], 1 );
	}
	AllMemoryBarrierWithGroupSync();

	// Compute exclusive prefix sum of the counters array
	uint previousSum = 0;
	for( i = 0; i < 0x8000; i += THREADS )
	{
		// Can't use for( i = thread; ... ) because it won't compile, with X3663: thread sync operation found in varying flow control
		const uint rdi = i + thread;

		// Load
		uint val = tempBuffer[ rdi ];
		// Compute exclusive prefix sum, using a group shared buffer
		uint2 prefixSum = exclusivePrefixSum( thread, val );
		// Store
		tempBuffer[ rdi ] = prefixSum.x + previousSum;
		previousSum += prefixSum.y;
	}
	AllMemoryBarrierWithGroupSync();

	// Produce sorted array of elements + indices
	// We only need the initial portion of that array, with the largest MAX_SAMPLE_LENGTH elements of the input row
	for( i = thread; i < width; i += THREADS )
	{
		uint val = loadSource( i );
		uint bucket = 0x7FFF - val;
		uint destinationIndex;
		InterlockedAdd( tempBuffer[ bucket ], 1, destinationIndex );

		[branch]
		if( destinationIndex < THREADS )
		{
			// Keep source index in the upper 16 bits of these values
			val |= ( i << 16 );
			probsLocal[ destinationIndex ] = val;
		}
	}
	// ==== The rest of the shader operates entirely on the group shared buffers ====
	GroupMemoryBarrierWithGroupSync();

#if 0
	// Debug code below: produce the result which should be equal to the output of sampleMax() compute shader
	if( 0 == thread )
		result[ 0 ] = probsLocal[ 0 ] >> 16;
	return;
#else
	// Find count of elements to use for the next step
	// The input tensor is extremely likely to contain many duplicate values
	// It has 32000 positive FP16 elements: https://en.wikipedia.org/wiki/Birthday_problem
	if( 0 == thread )
	{
		const uint minProb = probsLocal[ topK - 1 ] & 0xFFFFu;
		for( i = topK; i < THREADS; i++ )
		{
			uint cp = probsLocal[ i ] & 0xFFFFu;
			if( cp != minProb )
				break;	// Next element has smaller probability, which means we have found actual value for the TopK
		}
		tempScalarBuffer = i;
	}
	GroupMemoryBarrierWithGroupSync();
	const uint actualTopK = tempScalarBuffer;

	// ==== Compute softmax of the probsLocal values [ 0 .. actualTopK - 1 ] ====

	// That buffer is sorted in descending order, we can fetch the maximum value from the first element without another loop
	const float maxVal = upcast( probsLocal[ 0 ] );
	// Compute sum of exponentials with the log-sum-exp trick
	float sumExp = 0.0;
	[branch]
	if( thread < actualTopK )
	{
		float e = upcast( probsLocal[ thread ] );
		sumExp += exp( e - maxVal );
	}
	horizontalSum( thread, sumExp );
	if( 0 == thread )
		softMaxBuffer[ 0 ] = 1.0 / sumExp;
	GroupMemoryBarrierWithGroupSync();
	// Compute softmax probabilities
	const float mul = softMaxBuffer[ 0 ];
	float prob;
	[branch]
	if( thread < actualTopK )
	{
		float f = upcast( probsLocal[ thread ] );
		f = exp( f - maxVal ) * mul;
		prob = f;
	}
	else
	{
		// The original version masks out-of-top-k values with -inf, exp( -inf ) == 0.0
		prob = 0.0;
	}
	softMaxBuffer[ thread ] = prob;
	GroupMemoryBarrierWithGroupSync();

	// The rest of the algorithm only runs on 1 thread
	if( 0 != thread )
		return;

	// Finally, implement the sampling here.
	// After softmax, sum of all elements equals to 1.0, no need to compute another sum and scale the input random float.
	float acc = 0;

	// Reverse iteration order for better numerical accuracy
	// This way we're adding smaller numbers first, FP values have exponentially better precision near zero
	for( i = actualTopK - 1; i > 0; i-- )
	{
		const float prob = softMaxBuffer[ i ];
		acc += prob;
		[branch]
		if( acc > rand )
		{
			result[ 0 ] = probsLocal[ i ] >> 16;
			return;
		}
	}

	result[ 0 ] = probsLocal[ 0 ] >> 16;
#endif
}