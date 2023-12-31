// <c>( x * torch.rsqrt( x.pow( 2 ).mean( -1, keepdim = True ) + epsilon ) ) * weights</c>, in-place version
#include "miscUtils.hlsli"

OutputTensor tensor : register( u0 );
Tensor weights : register( t0 );

cbuffer Constants: register( b0 )
{
	uint4 inputSize: packoffset( c0 );
	uint4 inputStrides: packoffset( c1 );
	float epsilon : packoffset( c2.x );
}

inline void storeResult( uint idx, float elt )
{
	store( tensor, idx, elt );
}

#include "rmsNormImpl.hlsli"