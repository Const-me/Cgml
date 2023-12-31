// <c>torch.topk</c> masking the rest of values with -inf, then <c>nn.functional.softmax</c>, and finally <c>torch.multinomial</c>
#include "miscUtils.hlsli"

Tensor tensor : register( t0 );
RWBuffer<uint> tempBuffer: register( u0 );
RWBuffer<uint> result: register( u1 );

cbuffer Constants: register( b0 )
{
	// Row width of the input tensor
	uint width: packoffset( c0.x );
	// Affects how many elements to sample
	uint topK : packoffset( c0.y );
	// A random number in [ 0.0 .. 1.0 ] interval
	float rand : packoffset( c0.z );
}

// Count of threads in the shader, also maximum count of values to use for the sampling
static const uint THREADS = 1024;

groupshared uint prefixLocal[ THREADS ];

// Compute exclusive prefix sum of the prefixLocal array, in-place
// The argument is SV_GroupIndex integer, the range is [ 0 .. THREADS - 1 ]
// The return value is the complete sum of that array
inline uint exclusivePrefixSum( uint thread )
{
	// The algorithm is from that article:
	// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
	// The HTML at that link is broken, however.
	// To get a good version in CHM format use some BitTorrent search, the keyword for that is "GPU Gems 3 - (Malestrom)"

	// ---- Up-Sweep, or Reduce, Phase of a Work-Efficient Sum Scan Algorithm ----
	uint val = prefixLocal[ thread ];
	GroupMemoryBarrierWithGroupSync();
	uint2 it = uint2( 2, 1 );
	// See "Figure 39-3" in the book
	while( true )
	{
		const uint mask = it.x - 1;
		// First iteration: it = [ 2, 1 ], mask = 1, the condition is true for threads [ 1, 3, 5, 7, .. ]
		// Second one: it = [ 4, 2 ], mask = 3, the condition is true for threads [ 3, 7, 11, 15, .. ]
		// Last one: it = [ 256, 128 ], mask = 255, the condition is only true for the last thread 255
		if( ( thread & mask ) == mask )
		{
			val += prefixLocal[ thread ^ it.y ];
			prefixLocal[ thread ] = val;
		}
		GroupMemoryBarrierWithGroupSync();

		const uint next = it.x * 2;
		if( next > THREADS )
			break;
		it.y = it.x;
		it.x = next;
	}
	// Keep the result of the reduction
	const uint result = prefixLocal[ THREADS - 1 ];
	GroupMemoryBarrierWithGroupSync();

	// ---- Down-Sweep Phase of the Work-Efficient Parallel Sum Scan Algorithm ----
	// See "Figure 39-4" in the book
	[ branch ]
	if( thread == THREADS - 1 )
	{
		// This branch implements first 2 layers illustrated on that picture
		prefixLocal[ thread ] = 0;
		const uint rsi = ( THREADS / 2 ) - 1;
		val = prefixLocal[ rsi ];
		prefixLocal[ rsi ] = 0;
		prefixLocal[ thread ] = val;
	}
	GroupMemoryBarrierWithGroupSync();

	it = uint2( THREADS / 2, THREADS / 4 );
	while( true )
	{
		const uint mask = it.x - 1;
		// First iteration: it = [ 128, 64 ], mask = 127, the condition is true for threads [ 127, 255 ]
		// Second iteration: it = [ 64, 32 ], mask = 63, the condition is true for threads [ 63, 127, 191, 255 ]
		// Last iteration: it = [ 2, 1 ], mask = 1, the condition is true for threads [ 1, 3, 5, 7, .. ]
		if( ( thread & mask ) == mask )
		{
			// Compute index on the left side
			const uint rsi = thread ^ it.y;

			const uint oldVal = prefixLocal[ thread ];	// black arrow
			val = oldVal + prefixLocal[ rsi ];		// blue arrow
			prefixLocal[ rsi ] = oldVal;	// orange arrow
			prefixLocal[ thread ] = val;	// update current value in the shared buffer
		}

		GroupMemoryBarrierWithGroupSync();

		const uint next = it.y / 2;
		if( 0 == next )
			break;
		it.x = it.y;
		it.y = next;
	}

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
	// Converts the float16 stored in the low-half of the uint to a float
	// Therefore, no need to mask away the higher 2 bytes
	return f16tof32( u );
}

groupshared uint probsLocal[ THREADS ];
groupshared float probsPrefixSum[ THREADS ];

// Compute inclusive prefix sum of the probsPrefixSum array
// The argument is SV_GroupIndex integer, the range is [ 0 .. THREADS - 1 ]
inline void computeProbsPrefixSum( uint thread )
{
	const float orig = probsPrefixSum[ thread ];
	// ---- Up-Sweep, or Reduce, Phase of a Work-Efficient Sum Scan Algorithm ----

	// The iteration is a bit tricky because we wanna avoid integer multiplications or variable shifts inside these loops
	// The iterator is a vector of 4 uint values:
	// * x is iteration stride: 2 for leaves of the tree, 1024 for the root of the tree
	// * y is offset between elements being accumulated: 1 for leaves of the tree, 512 for the root of the tree
	// * z is current thread index, multiplied by x
	// * w is count of threads in the shader, again multiplied by x 
	// These 4 integers allow both loops to run without expensive integer instructions.
	// The HLSL compiler optimizes `it *= 2` and `it /= 2` into shifts, `ishl` and `ushr` respectively
	uint4 it = uint4( 2, 1, THREADS * 2, thread * 2 );
	uint i;
	for( ; it.x <= THREADS; it *= 2 )
	{
		for( i = it.x - 1 + it.w; i < THREADS; i += it.z )
		{
			float f = probsPrefixSum[ i ];
			f += probsPrefixSum[ i - it.y ];
			probsPrefixSum[ i ] = f;
		}

		GroupMemoryBarrierWithGroupSync();
	}

	// ---- Down-Sweep Phase of the Work-Efficient Parallel Sum Scan Algorithm ----
	if( 0 == thread )
	{
		// This branch implements first 2 layers illustrated on that picture
		const uint first = THREADS / 2 - 1;
		const uint last = THREADS - 1;
		probsPrefixSum[ last ] = probsPrefixSum[ first ];
		probsPrefixSum[ first ] = 0.0;
	}
	GroupMemoryBarrierWithGroupSync();

	// Implement the rest of the steps
	for( it /= 4; it.y > 0; it /= 2 )
	{
		for( i = it.x - 1 + it.w; i < THREADS; i += it.z )
		{
			const uint first = i - it.y;
			const float a = probsPrefixSum[ first ];
			const float b = probsPrefixSum[ i ];
			probsPrefixSum[ first ] = b;
			probsPrefixSum[ i ] = a + b;
		}
		GroupMemoryBarrierWithGroupSync();
	}

	// The last step to implement inclusivity
	probsPrefixSum[ thread ] += orig;
	GroupMemoryBarrierWithGroupSync();
}

groupshared uint actualTopKBuffer;

// Compute horisontal sum of the numbers. The result is only correct on the thread #0 of the group.
inline void horizontalSum( const uint thread, inout float sum )
{
	probsPrefixSum[ thread ] = sum;
	for( uint i = THREADS / 2; i > 1; i /= 2 )
	{
		GroupMemoryBarrierWithGroupSync();
		if( thread < i )
		{
			sum += probsPrefixSum[ thread + i ];
			probsPrefixSum[ thread ] = sum;
		}
	}
	GroupMemoryBarrierWithGroupSync();
	if( 0 == thread )
		sum += probsPrefixSum[ 1 ];
}

[ numthreads( THREADS, 1, 1 ) ]
void main( uint thread : SV_GroupIndex )
{
	uint i;
	// Write zeros to the temp buffer
	for( i = thread; i < 0x8000; i += THREADS )
		tempBuffer[ i ] = 0;
	AllMemoryBarrierWithGroupSync();

	// Count values using interlocked add intrinsics
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

		// Copy batch to the local buffer
		uint val = tempBuffer[ rdi ];
		prefixLocal[ thread ] = val;
		AllMemoryBarrierWithGroupSync();

		// Prefix sums of the buffer
		val = exclusivePrefixSum( thread );

		// Store prefix sum of the batch
		tempBuffer[ rdi ] = prefixLocal[ thread ] + previousSum;
		previousSum += val;
		AllMemoryBarrierWithGroupSync();
	}

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
	GroupMemoryBarrierWithGroupSync();

	// ==== The rest of the shader operates entirely on the group shared buffers ====
	
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
		actualTopKBuffer = i;
	}
	GroupMemoryBarrierWithGroupSync();
	const uint actualTopK = actualTopKBuffer;

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
		probsPrefixSum[ 0 ] = 1.0 / sumExp;
	GroupMemoryBarrierWithGroupSync();
	// Compute softmax probabilities
	const float mul = probsPrefixSum[ 0 ];
	[branch]
	if( thread < actualTopK )
	{
		float f = upcast( probsLocal[ thread ] );
		f = exp( f - maxVal ) * mul;
		probsPrefixSum[ thread ] = f;
	}
	else
	{
		// The original version masks the out-of-top-k values with -inf, and exp( -inf ) is 0.0
		probsPrefixSum[ thread ] = 0.0;
	}
	GroupMemoryBarrierWithGroupSync();

	// Compute inclusive prefix sums of the batch
	computeProbsPrefixSum( thread );

	// The rest of the algorithm only runs on 1 thread
	if( 0 != thread )
		return;

	// Implement the actual sampling here
	const float completeSum = probsPrefixSum[ actualTopK - 1 ];
	const float scaledRand = completeSum * rand;
	float acc = 0;

	// Reverse iteration order for better numerical accuracy
	// This way we're adding smaller numbers first, FP values have exponentially better precision near zero
	for( i = actualTopK - 1; i > 0; i-- )
	{
		const uint e = probsLocal[ i ];
		const float prob = upcast( e );
		acc += prob;
		if( acc > scaledRand )
		{
			result[ 0 ] = e >> 16;
			return;
		}
	}

	result[ 0 ] = probsLocal[ 0 ] >> 16;
}