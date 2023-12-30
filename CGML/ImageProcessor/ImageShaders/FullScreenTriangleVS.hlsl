// Generate a triangle which covers the entire render target

struct sVertex
{
	float2 tc: TEXCOORD0;
	float4 pos: SV_Position;
};

// The argument is in [ 0 .. 2 ] interval
sVertex main( uint vert: SV_VertexID )
{
	sVertex output;
	// Vertex positions to cover the entire render target
	float2 tmp = float2( (vert >> 1) & 1, vert & 1 );
	tmp = mad( tmp, 4.0, -1.0 );
	output.pos = float4( tmp, 0.5, 1.0 );
	// Texture coordinates. Note we negating Y, because in the clip space Y is directed upwards.
	output.tc = mad( tmp, float2( 0.5, -0.5 ), 0.5 );
	return output;
}