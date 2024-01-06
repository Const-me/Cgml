// Rotary position embedding for Instruct-0.2 version of the model from https://huggingface.co/
#include "miscUtils.hlsli"

OutputTensor tensor : register( u0 );

#ifndef USE_FP64
#define USE_FP64 0
#endif

cbuffer Constants: register( b0 )
{
	// <c>yzw</c> stride of the input/output tensor
	uint3 stride: packoffset( c0 );

	// 1000000.0
	float theta : packoffset( c0.w );
	// -2.0 / json.head_dim
	float minusHalfDimMul : packoffset( c1.x );

	int freqsOffset : packoffset( c1.y );
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
#else // USE_FP64 > 1
	// Microsoft neglected to documented that, but unfortunately `itod` instruction is an extended fp64 op.
	// Let's hope total count of these tokens won't exceed 16777216
	tmp *= f2.y;
#endif // USE_FP64 > 1
	float outer = wrapTwoPi( tmp );
#else // USE_FP64
	float outer = freq * f2.y;
#endif

	float2 result;
	sincos( outer, result.y, result.x );
	return result;
}

static const uint DIM = 128;
static const uint HALF_DIM = DIM / 2;	// 64

[ numthreads( HALF_DIM, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	// Compute sin and cos
	int2 freqSource;
	freqSource.x = (int)thread;
	freqSource.y = (int)group.y + freqsOffset;
	const float2 sinCos = computeFreqs( freqSource );

	// Load the row from the tensor into local variables
	// source.x is the first half, source.y is the second one
	const uint rsi = dot( group, stride );
	const float2 source = float2(
		load( tensor, rsi + thread ),
		load( tensor, rsi + thread + HALF_DIM ) );

	// First half of the row
	float result = sinCos.x * source.x;
	result = mad( sinCos.y, -source.y, result );
	store( tensor, rsi + thread, result );

	// Second half of the row
	result = sinCos.x * source.y;
	result = mad( sinCos.y, source.x, result );
	store( tensor, rsi + thread + HALF_DIM, result );
}