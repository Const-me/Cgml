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

#include "rotaryEmbeddingFreq.hlsli"

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