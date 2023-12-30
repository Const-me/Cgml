// Copy RGB pixels from 2D texture into a tensor

cbuffer Constants: register( b0 )
{
	// Image width; the height is count of thread groups dispatched
	uint width: packoffset( c0.x );
	// Count of pixels in the image, equal to width * height
	// This compute shader splits channels into 3 Z layers of the output tensor
	uint layerStride: packoffset( c0.y );

	float3 mean: packoffset( c1.x );
	float3 stdInv: packoffset( c2.x );
}

Texture2D<float4> source: register( t0 );
RWBuffer<float> dest: register( u0 );

static const uint THREADS = 64;

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	int3 loadPos;
	loadPos.y = (int)group.x;
	loadPos.z = 0;

	uint3 rdi;
	rdi.x = group.x * width + thread;
	rdi.y = rdi.x + layerStride;
	rdi.z = rdi.y + layerStride;

	for( uint i = thread; i < width; i += THREADS, rdi += THREADS )
	{
		loadPos.x = (int)i;
		float3 px = source.Load( loadPos ).xyz;
		px = ( px - mean ) * stdInv;
		dest[ rdi.x ] = px.x;
		dest[ rdi.y ] = px.y;
		dest[ rdi.z ] = px.z;
	}
}