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

#if USE_FP64
// Wrap a potentialy large FP64 value around 2*pi; the result is then passed to sincos() intrinsic
inline float wrapTwoPi( double x )
{
	const double twoPi = 6.283185307179586476925286766559;
	const double invTwoPi = 0.15915494309189533576888376337251;
	double div = x * invTwoPi;
	
#if USE_FP64 > 1
	// GPU supports FP64 FMA instruction, also itod / dtoi to convert numbers between int32 / fp64
	int i = (int)div;
	x = fma( (double)i, -twoPi, x );
#else
	// GPU doesn't support FMA, but it does support add/sub/mul
	float i = floor((float)div);
	x -= twoPi * i;
#endif
	return (float)x;
}
#endif

// Generating values which replace freqs_cis pre-computed tensor which is used in the original Python version
const float2 computeFreqs( int2 i2 )
{
	// Not sure the precision is good enough for the use case. The original code uses FP64 precision for these things
	// Might need another version of this shaders, for GPUs which support FP64 math
	// OTOH, the output is then downcasted to FP16. It's also possible Facebook used FP64 math by mistake.
	float2 f2 = ( float2 )i2;

	// Disable the warning X3571 "pow(f, e) will not work for negative f"
	// The theta is from the constant buffer, and CPU-running code sets that value to a large positive number like 10000.0
#pragma warning( disable : 3571 )
	float freq = pow( theta, f2.x * minusHalfDimMul );
#if USE_FP64
	double tmp = freq;
#if USE_FP64 > 1
	tmp *= (double)i2.y;
#else
	// Microsoft neglected to documented that, but unfortunately `itod` instruction is an extended fp64 op.
	// Let's hope total count of these tokens won't exceed 16777216
	tmp *= f2.y;
#endif
	float outer = wrapTwoPi( tmp );
#else
	float outer = freq * f2.y;
#endif

	float2 result;
	sincos( outer, result.y, result.x );
	return result;
}

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