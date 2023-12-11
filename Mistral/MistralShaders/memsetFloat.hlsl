// Fill buffer with zeros
#include "miscUtils.hlsli"

OutputTensor buffer : register( u0 );

static const uint VALS_PER_GROUP = 0x10000;
static const uint THREADS = 256;

cbuffer Constants: register( b0 )
{
	uint bufferLength: packoffset( c0.x );
}

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	uint rdi = group.x * VALS_PER_GROUP;
	const uint rdiEnd = rdi + min( VALS_PER_GROUP, bufferLength - rdi );

	for( rdi += thread; rdi < rdiEnd; rdi += THREADS )
		buffer[ rdi ] = 0;
}