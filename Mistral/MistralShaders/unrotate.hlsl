// Copy slice from circular buffer into a dense tensor. The input slice might have a discontinuity due to wrapping of that buffer.
#include "miscUtils.hlsli"

Tensor tensor : register( t0 );
OutputTensor result : register( u0 );

cbuffer Constants: register( b0 )
{
	// Strides of the input tensor, in elements
	uint4 inputStrides: packoffset( c0 );
	// Width of the output tensor
	uint width: packoffset( c1.x );
	// YZW strides of the output tensor, in elements
	uint3 outputStrides: packoffset( c1.y );
	// Starting offset in the input tensor; typically 0, except for rotated coordinates
	int4 inputOffset0: packoffset( c2 );
	// Length of the first slice in the input tensor; for unrotated coordinates, equal to output size
	uint4 inputLength0: packoffset( c3 );
	// For rotated coordinates, integer to add to the output position in the second slice to find input position; unused for unrotated coordinates.
	int4 inputOffset1: packoffset( c4 );
}

static const uint THREADS = 64;

inline uint3 unrotateGroup( uint3 group )
{
	int3 off = ( group < inputLength0.yzw ) ? inputOffset0.yzw : inputOffset1.yzw;
	return (uint3)( (int3)group + off );
}

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	uint rdi = dot( group, outputStrides );
	uint rsi = dot( unrotateGroup( group ), inputStrides.yzw );
	const uint rsiInc = inputStrides.x * THREADS;

	[branch]
	if( width <= inputLength0.x )
	{
		// The rotation happens over some other direction, the input stride of the row is the constant number inputStrides.x.
		const uint rdiEnd = rdi + width;
		rdi += thread;
		rsi += ( inputOffset0.x + thread ) * inputStrides.x;
		for( ; rdi < rdiEnd; rdi += THREADS, rsi += rsiInc )
			result[ rdi ] = tensor[ rsi ];
	}
	else
	{
		// Output row is composed of two slices
		rdi += thread;
		int2 off = int2(inputOffset0.x, inputOffset1.x);
		off *= (int)inputStrides.x;
		rsi += thread * inputStrides.x;

		for( uint i = thread; i < width; i += THREADS, rdi += THREADS )
		{
			int off1 = ( i < inputLength0.x ) ? off.x : off.y;
			result[ rdi ] = tensor[ (uint)( (int)rsi + off1 ) ];
		}
	}
}