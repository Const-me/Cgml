// Rotary position embedding
#include "miscUtils.hlsli"

OutputTensor tensor : register( u0 );

#ifndef THREADS
static const uint THREADS = 64;
#endif

#ifndef USE_FP64
#define USE_FP64 0
#endif

cbuffer Constants: register( b0 )
{
	// Size of the input/output tensor
	uint4 size: packoffset( c0 );
	// Stride of the input/output tensor
	uint4 stride: packoffset( c1 );

	// 10000.0
	float theta : packoffset( c2.x );
	// -2.0 / json.head_dim
	float minusHalfDimMul : packoffset( c2.y );

	int freqsOffset : packoffset( c2.z );
}

#include "rotaryEmbeddingFreq.hlsli"

inline float2 load2( OutputTensor buff, uint rsi )
{
	float2 res;
	res.x = load( buff, rsi );
	res.y = load( buff, rsi + 1 );
	return res;
}

inline void store2( OutputTensor buff, uint rsi, float2 vals )
{
	store( buff, rsi, vals.x );
	store( buff, rsi + 1, vals.y );
}

// Multiply two complex numbers stored as [ real, imaginary ] 2D vector
// Multiplication formula: https://en.wikipedia.org/wiki/Complex_number#Multiplication_and_square
float2 mulComplex( float2 a, float2 b )
{
	float2 res;
	res.x = a.x * b.x - a.y * b.y;
	res.y = a.x * b.y + a.y * b.x;
	return res;
}

// If this shader will show up in profiler, possible to optimize by implementing X and Z loops here in HLSL
// They need the same frequency complex number, because broadcasting
[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	uint rsi = dot( group, stride.yzw );
	const uint rsiEnd = rsi + size.x;
	rsi += thread * 2;

	int2 freqSource;
	freqSource.x = thread;
	freqSource.y = (int)group.y + freqsOffset;

	for( ; rsi < rsiEnd; rsi += 2 * THREADS, freqSource.x += THREADS )
	{
		const float2 f = computeFreqs( freqSource );

		float2 v = load2( tensor, rsi );
		v = mulComplex( v, f );
		store2( tensor, rsi, v );
	}
}