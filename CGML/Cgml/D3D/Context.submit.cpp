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

HRESULT COMLIGHTCALL Context::bindTensors( iTensor** arr, int countWriteInt, int countReadInt ) noexcept
{
	if( countWriteInt <= 0 || countWriteInt > D3D11_PS_CS_UAV_REGISTER_COUNT )
	{
		logError( u8"Valid count of output tensors is [ 1 .. %i ]", (int)D3D11_PS_CS_UAV_REGISTER_COUNT );
		return E_INVALIDARG;
	}
	if( countReadInt < 0 || countReadInt > D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT )
	{
		logError( u8"Valid count of input tensors is [ 0 .. %i ]", (int)D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT );
		return E_INVALIDARG;
	}

	const size_t countWrite = (uint32_t)countWriteInt;
	const size_t countRead = (uint32_t)countReadInt;
	const size_t countUav = std::max( (size_t)boundUavs, countWrite );
	const size_t countSrv = std::max( (size_t)boundSrvs, countRead );

	ID3D11UnorderedAccessView** const arrUav = (ID3D11UnorderedAccessView**)_alloca( countUav * sizeof( void* ) );
	ID3D11ShaderResourceView** const arrSrv = (ID3D11ShaderResourceView**)_alloca( countSrv * sizeof( void* ) );

	for( size_t i = 0; i < countWrite; i++ )
		CHECK( makeUav( &arrUav[ i ], arr[ i ] ) );
	if( countWrite < countUav )
		__stosq( (uint64_t*)&arrUav[ countWrite ], 0, countUav - countWrite );

	for( size_t i = 0; i < countRead; i++ )
		CHECK( makeSrv( &arrSrv[ i ], arr[ countWrite + i ] ) );
	if( countRead < countSrv )
		__stosq( (uint64_t*)&arrSrv[ countRead ], 0, countSrv - countRead );

	context->CSSetUnorderedAccessViews( 0, (UINT)countUav, arrUav, nullptr );
	if( countSrv > 0 )
		context->CSSetShaderResources( 0, (UINT)countSrv, arrSrv );

	boundUavs = (uint8_t)countWrite;
	boundSrvs = (uint8_t)countRead;
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