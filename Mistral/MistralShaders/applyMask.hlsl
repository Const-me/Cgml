// Apply diagonel mask to the tensor, writing maskValue constnt to the top-right area above the diagonal
#include "miscUtils.hlsli"

OutputTensor tensor : register( u0 );

cbuffer Constants : register( b0 )
{
	uint width : packoffset( c0.x );
	uint3 strides : packoffset( c0.y );
	uint xOffset: packoffset( c1.x );
	int diagonal : packoffset( c1.y );
	uint maskValue : packoffset( c1.z );
}

#ifndef THREADS
#define THREADS 64
#endif

[numthreads( THREADS, 1, 1 )]
void main( uint3 group : SV_GroupID, uint thread : SV_GroupIndex )
{
	uint rdi = dot( group, strides ) + xOffset;
	const uint rdiEnd = rdi + width;

	int2 pos;
	pos.x = (int) thread;
	pos.y = (int) group.x;
	for( rdi += thread; rdi < rdiEnd; rdi += THREADS, pos.x += THREADS )
	{
		int d = pos.x - pos.y;
		[branch]
		if( d >= diagonal )
		{
#if USE_BF16
			tensor[ rdi ] = maskValue;
#else
			tensor[ rdi ] = asfloat( maskValue );
#endif
		}
	}
}