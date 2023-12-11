RWBuffer<uint> result: register( u0 );
Buffer<uint> inputTensor: register( t0 );
Buffer<uint> maskTensor: register( t1 );

cbuffer Constants: register( b0 )
{
	uint height: packoffset( c0.x );
	uint curPos: packoffset( c0.y );
	uint resultStride: packoffset( c0.z );
	uint maskStride: packoffset( c0.w );
}

static const uint THREADS = 32;

[ numthreads( THREADS, 1, 1 ) ]
void main( uint thread : SV_GroupIndex )
{
	const uint maskBit = 1u << ( curPos % 32 );
	const uint maskOffset = curPos / 32;

	for( uint i = thread; i < height; i += THREADS )
	{
		const uint mask32 = maskTensor[ i * maskStride + maskOffset ];
		if( 0 != ( mask32 & maskBit ) )
			continue;

		const uint rdi = i * resultStride + curPos;
		result[ rdi ] = inputTensor[ i ];
	}
}