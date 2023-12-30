#include "stdafx.h"
#include "Compressor.h"
#include "bcml1.h"
#include <D3D/tensorUtils.h>
using namespace Cgml;

HRESULT iCompressor::create( std::unique_ptr<iCompressor>& rdi, ID3D11Device* device )
{
	std::unique_ptr<Compressor> res = std::make_unique<Compressor>( device );
	CHECK( res->create() );
	rdi = std::move( res );
	return S_OK;
}

Compressor::~Compressor()
{
	if( nullptr != work )
	{
		WaitForThreadpoolWorkCallbacks( work, TRUE );
		CloseThreadpoolWork( work );
		work = nullptr;
	}
}

HRESULT Compressor::create()
{
	if( !Bcml1::checkExtensionFlags( Bcml1::eCpuExtensionFlags::AVX2 ) )
	{
		logError( u8"BCML compressor requires a CPU with AVX2 support" );
		return HRESULT_FROM_WIN32( ERROR_HV_CPUID_FEATURE_VALIDATION );
	}

	work = CreateThreadpoolWork( &workCallbackStatic, this, nullptr );
	if( nullptr == work )
		return getLastHr();
	return S_OK;
}

void __stdcall Compressor::workCallbackStatic( PTP_CALLBACK_INSTANCE Instance, PVOID Context, PTP_WORK Work )
{
	Compressor* const pc = (Compressor*)Context;
	HRESULT hr = E_UNEXPECTED;
	try
	{
		hr = pc->workCallback();
	}
	catch( HRESULT h )
	{
		hr = h;
	}
	catch( const std::bad_alloc& )
	{
		hr = E_OUTOFMEMORY;
	}
	catch( const std::exception& )
	{
		hr = E_FAIL;
	}

	if( FAILED( hr ) )
	{
		InterlockedCompareExchange( &pc->m_status, hr, S_FALSE );
		pc->condVar.notify_all();
	}
}

using Lock = std::unique_lock<std::mutex>;

HRESULT Compressor::getBuffer( std::vector<__m256i>& vec, size_t lengthBytes ) noexcept
{
	Lock lk{ mutex };
	CHECK( m_status );
	CHECK( uploadCompleteTensors() );

	std::vector<__m256i> result;
	if( !poolInputs.empty() )
	{
		result = std::move( *poolInputs.rbegin() );
		poolInputs.pop_back();
	}

	const size_t elts = ( lengthBytes + 31 ) / 32;
	try
	{
		result.resize( elts );
	}
	catch( const std::bad_alloc& )
	{
		return E_OUTOFMEMORY;
	}
	result.swap( vec );
	return S_OK;
}

HRESULT Compressor::bcml( Cgml::iTensor** rdi, const Cgml::sTensorDesc& desc, std::vector<__m256i>& data, size_t lengthBytes ) noexcept
{
	const size_t elts = desc.shape.countElements();
	const size_t cbElt = bytesPerElement( desc.dataType );
	size_t expectedBytes = elts * cbElt;
	if( expectedBytes != lengthBytes )
	{
		logError( u8"Incorrect size" );
		return E_INVALIDARG;
	}

	ComLight::CComPtr<ComLight::Object<Cgml::Tensor>> tensor;
	{
		sTensorDesc compressedDesc;
		CHECK( Bcml1::makeDesc( compressedDesc, desc ) );
		CHECK( ComLight::Object<Cgml::Tensor>::create( tensor, compressedDesc, nullptr ) );
	}

	Lock lk{ mutex };
	CHECK( uploadCompleteTensors() );

	auto wakeUp = [ this ]()
	{
		HRESULT hr = uploadCompleteTensors();
		if( FAILED( hr ) )
			InterlockedCompareExchange( &m_status, hr, S_FALSE );

		if( !pending.has_value() )
			return true;

		if( FAILED( m_status ) )
			return true;

		return false;
	};
	condVar.wait( lk, wakeUp );
	CHECK( m_status );
	if( pending.has_value() )
		return E_UNEXPECTED;

	pending.emplace();
	pending->sourceVector.swap( data );
	pending->tensor = tensor;	//< addRef there
	pending->sourceStride = desc.shape.stride;
	pending->sourceType = desc.dataType;
	SubmitThreadpoolWork( work );

	// Move ownership to the caller
	*rdi = tensor.detach();
	return S_OK;
}

HRESULT Compressor::join() noexcept
{
	CHECK( m_status );
	WaitForThreadpoolWorkCallbacks( work, FALSE );
	CHECK( m_status );
	CHECK( uploadCompleteTensors() );
	return S_OK;
}

HRESULT Compressor::workCallback()
{
	std::optional<PendingJob> pending;
	std::vector<uint32_t> compressedData;
	{
		Lock lk{ mutex };
		CHECK( m_status );
		if( !this->pending.has_value() )
			return E_UNEXPECTED;
		pending.swap( this->pending );

		if( !poolCompressed.empty() )
		{
			compressedData.swap( *poolCompressed.rbegin() );
			poolCompressed.pop_back();
		}
	}

	const HRESULT hr = compressImpl( *pending, compressedData );

	{
		Lock lk{ mutex };

		if( !pending->sourceVector.empty() )
			poolInputs.emplace_back( std::move( pending->sourceVector ) );

		if( !compressedData.empty() )
			poolCompressed.emplace_back( std::move( compressedData ) );
	}

	return hr;
}

HRESULT Compressor::compressImpl( PendingJob& job, std::vector<uint32_t>& resultBuffer )
{
	sTensorDesc descCompressed = job.tensor->getDesc();

	CHECK( Bcml1::compress( job.sourceType, descCompressed, job.sourceVector, resultBuffer ) );

	{
		Lock lk{ mutex };
		// Release source buffer back to the pool
		poolInputs.emplace_back( std::move( job.sourceVector ) );
		job.sourceVector.clear();

		// Move the result buffer to the completeTensors vector
		CompleteJob& rdi = completeTensors.emplace_back();
		rdi.bufferData.swap( resultBuffer );
		rdi.tensor.attach( job.tensor.detach() );
	}

	condVar.notify_all();
	return S_OK;
}

HRESULT Compressor::uploadCompleteTensors()
{
	while( !completeTensors.empty() )
	{
		CompleteJob& top = *completeTensors.rbegin();
		CHECK( top.tensor->createImmutableRaw( device, top.bufferData ) );
		poolCompressed.emplace_back( std::move( top.bufferData ) );
		completeTensors.pop_back();
	}
	return S_OK;
}