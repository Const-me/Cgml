// Compute softmax function for every row of the tensor, in-place
#include "miscUtils.hlsli"

OutputTensor tensor : register( u0 );

cbuffer Constants: register( b0 )
{
	// X dimension of the tensor
	uint width: packoffset( c0.x );
	// <c>yzw</c> strides of the tensor; it's assumed to be dense so the strides.x is one
	uint3 strides: packoffset( c0.y );
}

static const uint THREADS = 32;

// In release builds, HLSL compiler realizes this value is a compile-time constant 1.0.
// The unnecessary multiplications are dropped by the compiler.
static const float initialMul = 1.0f;

#include "softMaxImpl.hlsli"