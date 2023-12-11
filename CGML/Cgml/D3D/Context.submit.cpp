#include "stdafx.h"
#include "Context.h"
#include "Tensor.h"
using namespace Cgml;

HRESULT COMLIGHTCALL Context::bindShader( uint16_t id, const uint8_t* constantBufferData, int cbSize ) noexcept
{
	if( id >= shaders.size() )
		return E_BOUNDS;

	if( cbSize < 0 )
		return E_INVALIDARG;
	else if( cbSize > 0 && constantBufferData == nullptr )
		return E_POINTER;

	context->CSSetShader( shaders.at( id ), nullptr, 0 );
	CHECK( profiler.computeShader( id ) );
	CHECK( constantBuffers.updateAndBind( device, context, constantBufferData, cbSize ) );
	return S_OK;
}

HRESULT COMLIGHTCALL Context::dispatch( int groupsX, int groupsY, int groupsZ ) noexcept
{
	constexpr int max = D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
	if( groupsX <= 0 || groupsY <= 0 || groupsZ <= 0 ||
		groupsX > max || groupsY > max || groupsZ > max )
		return E_INVALIDARG;

	context->Dispatch( (UINT)groupsX, (UINT)groupsY, (UINT)groupsZ );

	return S_OK;
}

namespace
{
	inline HRESULT makeUav( ID3D11UnorderedAccessView** rdi, iTensor* rsi )
	{
		const Tensor* tensorBase = static_cast<Tensor*>( rsi );
		ID3D11UnorderedAccessView* view = tensorBase->writeView();
		if( nullptr == view )
			return E_INVALIDARG;
		*rdi = view;
		return S_OK;
	}

	inline HRESULT makeSrv( ID3D11ShaderResourceView** rdi, iTensor* rsi )
	{
		const Tensor* tensorBase = static_cast<Tensor*>( rsi );
		ID3D11ShaderResourceView* view = tensorBase->readView();
		*rdi = view;
		return S_OK;
	}
}

HRESULT Context::bindTensors( std::initializer_list<iTensor*> uav, std::initializer_list<iTensor*> srv ) noexcept
{
	const size_t countUav = std::max( (size_t)boundUavs, uav.size() );
	const size_t countSrv = std::max( (size_t)boundSrvs, srv.size() );

	ID3D11UnorderedAccessView** const arrUav = (ID3D11UnorderedAccessView**)_alloca( countUav * sizeof( void* ) );
	ID3D11ShaderResourceView** const arrSrv = (ID3D11ShaderResourceView**)_alloca( countSrv * sizeof( void* ) );

	for( size_t i = 0; i < uav.size(); i++ )
		CHECK( makeUav( &arrUav[ i ], uav.begin()[ i ] ) );

	if( uav.size() < countUav )
		__stosq( (uint64_t*)&arrUav[ uav.size() ], 0, countUav - uav.size() );

	for( size_t i = 0; i < srv.size(); i++ )
		CHECK( makeSrv( &arrSrv[ i ], srv.begin()[ i ] ) );

	if( srv.size() < countSrv )
		__stosq( (uint64_t*)&arrSrv[ srv.size() ], 0, countSrv - srv.size() );

	context->CSSetUnorderedAccessViews( 0, (UINT)countUav, arrUav, nullptr );
	if( countSrv > 0 )
		context->CSSetShaderResources( 0, (UINT)countSrv, arrSrv );

	boundUavs = (uint8_t)uav.size();
	boundSrvs = (uint8_t)srv.size();
	return S_OK;
}

HRESULT COMLIGHTCALL Context::unbindInputs() noexcept
{
	if( boundSrvs != 0 )
	{
		ID3D11ShaderResourceView** const arrSrv = (ID3D11ShaderResourceView**)_alloca( boundSrvs * sizeof( void* ) );
		__stosq( (uint64_t*)&arrSrv[ 0 ], 0, boundSrvs );
		context->CSSetShaderResources( 0, (UINT)boundSrvs, arrSrv );
		boundSrvs = 0;
	}
	return S_OK;
}