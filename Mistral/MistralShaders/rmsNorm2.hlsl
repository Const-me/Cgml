// <c>( x * torch.rsqrt( x.pow( 2 ).mean( -1, keepdim = True ) + epsilon ) ) * weights</c>
#include "miscUtils.hlsli"

OutputTensor result : register( u0 );
Tensor tensor : register( t0 );
Tensor weights : register( t1 );

cbuffer Constants: register( b0 )
{
	uint4 inputSize: packoffset( c0 );
	uint4 inputStrides: packoffset( c1 );
	float epsilon : packoffset( c2.x );
}

inline void storeResult( uint idx, float elt )
{
	store( result, idx, elt );
}

#include "rmsNormImpl.hlsli"