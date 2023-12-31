// The final softmax does very long tensors, size.x = 32000, and has a pre-scaling with a constant
#include "miscUtils.hlsli"

OutputTensor tensor : register( u0 );

cbuffer Constants: register( b0 )
{
	// X dimension of the tensor
	uint width: packoffset( c0.x );
	// <c>yzw</c> strides of the tensor; it's assumed to be dense so the strides.x is one
	uint3 strides: packoffset( c0.y );
	// Multiplier to apply to source values; expected to be a positive number.
	float initialMul : packoffset( c1.x );
}

static const uint THREADS = 1024;

#include "softMaxImpl.hlsli"