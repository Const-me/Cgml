// Implements topP sampling
#include "miscUtils.hlsli"

Tensor tensor : register( t0 );
RWBuffer<uint> tempBuffer: register( u0 );
RWBuffer<uint> result: register( u1 );

cbuffer Constants: register( b0 )
{
	// Row width of the input tensor
	uint width: packoffset( c0.x );
	// Distance between rows of the tensor
	uint tensorStride: packoffset( c0.y );
	// Affects how many elements to sample
	float topP : packoffset( c0.z );
	// A random number in [ 0.0 .. 1.0 ] interval
	float rand : packoffset( c0.w );
}

static const uint THREADS = 256;
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
#if USE_BF16
	uint u = tensor[ rsi ];
#else
	float f = tensor[ rsi ];
	uint u = f32tof16( f );
#endif	
	if( 0 != ( u & 0x8000 ) )
	{
		// Get rid of the negative probabilities. Not necessarily an error.
		// -0.0f is a perfectly cromulent probability, despite technically a negative number
		u = 0;
	}
	return u;
}

#if USE_BF16
inline float upcast( uint u )
{
	return asfloat( u << 16 );
}
#else
inline float upcast( uint u )
{
	// https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/f16tof32
	// Converts the float16 stored in the low-half of the uint to a float
	// Therefore, no need to mask away the higher 2 bytes
	return f16tof32( u );
}
#endif

static const uint MAX_SAMPLE_LENGTH_LOG2 = 10;
// 1024
static const uint MAX_SAMPLE_LENGTH = 1u << MAX_SAMPLE_LENGTH_LOG2;

groupshared uint probsLocal[ MAX_SAMPLE_LENGTH ];
groupshared float probsPrefixSum[ MAX_SAMPLE_LENGTH ];

// Compute inclusive prefix sum of the probsPrefixSum array
// The argument is SV_GroupIndex integer, the range is [ 0 .. THREADS - 1 ]
inline void computeProbsPrefixSum( uint thread )
{
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
	for( ; it.x <= MAX_SAMPLE_LENGTH; it *= 2 )
	{
		for( i = it.x - 1 + it.w; i < MAX_SAMPLE_LENGTH; i += it.z )
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
		const uint first = MAX_SAMPLE_LENGTH / 2 - 1;
		const uint last = MAX_SAMPLE_LENGTH - 1;
		probsPrefixSum[ last ] = probsPrefixSum[ first ];
		probsPrefixSum[ first ] = 0.0;
	}
	GroupMemoryBarrierWithGroupSync();

	// Implement the rest of the steps
	for( it /= 4; it.y > 0; it /= 2 )
	{
		for( i = it.x - 1 + it.w; i < MAX_SAMPLE_LENGTH; i += it.z )
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
	for( i = thread; i < MAX_SAMPLE_LENGTH; i += THREADS )
	{
		float a = probsPrefixSum[ i ];
		float b = upcast( probsLocal[ i ] );
		probsPrefixSum[ i ] = a + b;
	}

	GroupMemoryBarrierWithGroupSync();
}

inline uint findSampleLength()
{
	const float threshold = topP;
	if( probsPrefixSum[ 1 ] > threshold )
		return 1;

	uint left = 1;
	uint right = MAX_SAMPLE_LENGTH;
	while( true )
	{
		uint mid = ( right + left ) / 2;
		if( mid == left )
			return mid;

		float f = probsPrefixSum[ mid ];
		if( f < threshold )
			left = mid;
		else
			right = mid;
	}
}

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint baseTemp = group.x * 0x8000;
	const uint baseSource = group.x * tensorStride;
	uint i;

	// Write zeros to the temp buffer
	for( i = thread; i < 0x8000; i += THREADS )
		tempBuffer[ baseTemp + i ] = 0;
	AllMemoryBarrierWithGroupSync();

	// Count values using interlocked add intrinsics
	for( i = thread; i < width; i += THREADS )
	{
		uint val = loadSource( baseSource + i );
		uint bucket = 0x7FFF - val;
		InterlockedAdd( tempBuffer[ baseTemp + bucket ], 1 );
	}
	AllMemoryBarrierWithGroupSync();

	// Compute exclusive prefix sum of the counters array
	uint previousSum = 0;
	for( i = 0; i < 0x8000; i += THREADS )
	{
		const uint rdi = baseTemp + i + thread;

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
		uint val = loadSource( baseSource + i );
		uint bucket = 0x7FFF - val;
		uint destinationIndex;
		InterlockedAdd( tempBuffer[ baseTemp + bucket ], 1, destinationIndex );

		[branch]
		if( destinationIndex >= MAX_SAMPLE_LENGTH )
			continue;
		// Keep source index in the upper 16 bits of these values
		val |= ( i << 16 );
		probsLocal[ destinationIndex ] = val;
	}
	GroupMemoryBarrierWithGroupSync();

	// Copy upcasted probabilities from probsLocal to probsPrefixSum array
	for( i = thread; i < MAX_SAMPLE_LENGTH; i += THREADS )
		probsPrefixSum[ i ] = upcast( probsLocal[ i ] );
	GroupMemoryBarrierWithGroupSync();

	// Compute inclusive prefix sums of the batch
	computeProbsPrefixSum( thread );

	// The rest of the algorithm only runs on 1 thread
	if( 0 != thread )
		return;

	// Implement the actual sampling here
	const uint sampleLength = findSampleLength();
	if( sampleLength < 2 )
	{
		// The model is pretty confident what's next, nothing to sample
		// Produce index of the element with the maximum probability
		result[ group.x ] = ( probsLocal[ 0 ] >> 16 );
		return;
	}

	// Need that random sampling
	const float topSumInv = 1.0 / probsPrefixSum[ sampleLength - 1 ];
	float acc = 0;

	// Reverse iteration order for better numerical accuracy
	// This way we're adding smaller numbers first, FP values have exponentially better precision near zero
	for( i = sampleLength - 1; i > 0; i-- )
	{
		const uint e = probsLocal[ i ];
		float prob = upcast( e );
		acc = mad( prob, topSumInv, acc );
		if( acc > rand )
		{
			result[ group.x ] = e >> 16;
			return;
		}
	}

	result[ group.x ] = probsLocal[ 0 ] >> 16;
}