// Compute index of the first maximum element in each row of the tensor
#include "miscUtils.hlsli"

Tensor tensor : register( t0 );
RWBuffer<uint> result: register( u0 );

cbuffer Constants: register( b0 )
{
	// Row width of the input tensor
	uint width: packoffset( c0.x );
	// Distance between rows of the tensor
	uint tensorStride: packoffset( c0.y );
}

static const uint THREADS = 256;

static const float FLT_MAX = 3.402823466e+38F;
static const uint UINT_MAX = ~(uint)0;

groupshared float reductionValues[ THREADS ];
groupshared uint reductionIndex[ THREADS ];

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint baseSource = group.x * tensorStride;
	uint i;

	// Count values using interlocked add intrinsics
	float maxVal = -FLT_MAX;
	uint maxIndex = UINT_MAX;
	for( i = thread; i < width; i += THREADS )
	{
		const float val = load( tensor, baseSource + i );
		[branch]
		if( val > maxVal )
		{
			maxVal = val;
			maxIndex = i;
		}
	}

	reductionValues[ thread ] = maxVal;
	reductionIndex[ thread ] = maxIndex;

	for( i = THREADS / 2; i != 0; i /= 2 )
	{
		GroupMemoryBarrierWithGroupSync();
		[branch]
		if( thread < i )
		{
			const float other = reductionValues[ thread + i ];
			[branch]
			if( other > maxVal )
			{
				maxVal = other;
				maxIndex = reductionIndex[ thread + i ];
				reductionValues[ thread ] = maxVal;
				reductionIndex[ thread ] = maxIndex;
			}
		}
	}

	[branch]
	if( 0 == thread )
		result[ group.x ] = maxIndex;
}