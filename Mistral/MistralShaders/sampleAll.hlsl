// <c>torch.multinomial</c>
#include "miscUtils.hlsli"

Tensor tensor : register( t0 );
RWBuffer<uint> result: register( u0 );

cbuffer Constants: register( b0 )
{
	// Row width of the input tensor
	uint width: packoffset( c0.x );
	// A random number in [ 0.0 .. 1.0 ] interval
	float rand : packoffset( c0.y );
	// Same number as FP64
	uint2 rand64 : packoffset( c0.z );
}

#ifndef USE_FP64
#define USE_FP64 0
#endif

static const uint THREADS = 1024;

#if USE_FP64
groupshared uint2 sumBuffer[ THREADS ];

inline uint2 storeDouble( double x )
{
	uint2 res;
	asuint( x, res.x, res.y );
	return res;
}
inline double loadDouble( uint2 u )
{
	return asdouble( u.x, u.y );
}
#else
groupshared float sumBuffer[ THREADS ];
inline float storeDouble( float x ) { return x; }
inline float loadDouble( float x ) { return x; }
#endif

groupshared float maxValues[ THREADS ];
groupshared uint maxIndices[ THREADS ];

static const float FLT_MAX = 3.402823466e+38F;
static const uint UINT_MAX = ~(uint)0;

[ numthreads( THREADS, 1, 1 ) ]
void main( uint thread : SV_GroupIndex )
{
	uint i;
#if USE_FP64
	double acc = 0.0;
#else
	float acc = 0.0;
#endif

	float maxVal = -FLT_MAX;
	uint maxIndex = UINT_MAX;
	for( i = thread; i < width; i += THREADS )
	{
		float val = load( tensor, i );
		[branch]
		if( val > 0 )
		{
			acc += val;
			[branch]
			if( val > maxVal )
			{
				maxVal = val;
				maxIndex = i;
			}
		}
	}

	sumBuffer[ thread ] = storeDouble( acc );
	maxValues[ thread ] = maxVal;
	maxIndices[ thread ] = maxIndex;

	for( i = THREADS / 2; i != 0; i /= 2 )
	{
		GroupMemoryBarrierWithGroupSync();
		[branch]
		if( thread < i )
		{
			// Accumulate the sum of the elements
			acc += loadDouble( sumBuffer[ thread + i ] );
			sumBuffer[ thread ] = storeDouble( acc );

			// Find index of the maximum element, using two other shared buffers
			const float other = maxValues[ thread + i ];
			[branch]
			if( other > maxVal )
			{
				maxVal = other;
				maxIndex = maxIndices[ thread + i ];
				maxValues[ thread ] = maxVal;
				maxIndices[ thread ] = maxIndex;
			}
		}
	}

	// The rest of the algorithm only runs on 1 thread
	if( 0 != thread )
		return;

#if USE_FP64
	const double threshold = acc * loadDouble( rand64 );
#else
	const float threshold = acc * rand;
#endif

	acc = 0;
	for( i = 0; i < width; i++ )
	{
		float e = load( tensor, i );
		[branch]
		if( e > 0 )
		{
			acc += e;
			[branch]
			if( acc >= threshold )
			{
				result[ 0 ] = i;
				return;
			}
		}
	}

	// Floating-point issues, caused by different summation order between parallel and sequential parts of this shader
	// As a fallback, return index of the first maximum element
	result[ 0 ] = maxIndex;
}