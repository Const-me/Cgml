// Copy pixels across RGBA textures of different pixel formats, and reset alpha to 1.0
Texture2D<float4> source: register( t0 );

float4 main( float2 tc: TEXCOORD0, float4 pos: SV_Position ): SV_Target0
{
	int3 loadPos;
	loadPos.xy = (int2)pos.xy;
	loadPos.z = 0;
	float4 r = source.Load( loadPos );
	r.w = 1.0;
	return r;
}