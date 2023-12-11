#include <stdafx.h>
#include "miscUtils.h"
#include "API/TensorShape.h"

__m128i __vectorcall bufferMemoryUsage( ID3D11Buffer* buffer )
{
	if( nullptr != buffer )
	{
		D3D11_BUFFER_DESC desc;
		buffer->GetDesc( &desc );

		if( desc.Usage != D3D11_USAGE_STAGING )
			return setHigh_size( desc.ByteWidth );
		else
			return setLow_size( desc.ByteWidth );
	}
	return _mm_setzero_si128();
}

__m128i __vectorcall resourceMemoryUsage( ID3D11ShaderResourceView* srv )
{
	if( nullptr != srv )
	{
		CComPtr<ID3D11Resource> res;
		srv->GetResource( &res );
		CComPtr<ID3D11Buffer> buff;
		if( SUCCEEDED( res.QueryInterface( &buff ) ) )
			return bufferMemoryUsage( buff );
		assert( false );	// We don't use textures in this project
	}
	return _mm_setzero_si128();
}

bool canMulMat( const Cgml::TensorShape& t0, const Cgml::TensorShape& t1 )
{
	/*
	return
	( t0.ne[ 0 ] == t1.ne[ 0 ] ) &&
	( t0.ne[ 2 ] == t1.ne[ 2 ] ) &&
	( t0.ne[ 3 ] == t1.ne[ 3 ] ); */
	__m128i a = t0.sizeVec();
	__m128i b = t1.sizeVec();
	__m128i xx = _mm_xor_si128( a, b );
	xx = _mm_shuffle_epi32( xx, _MM_SHUFFLE( 3, 2, 0, 0 ) );
	return (bool)_mm_testz_si128( xx, xx );
}