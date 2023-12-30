Texture2D<float4> source: register( t0 );
SamplerState ss : register ( s0 );

float4 main( float2 uv: TEXCOORD0 ): SV_Target0
{
	float3 px = source.Sample ( ss, uv ).rgb;
	return float4( px, 1.0 );
}