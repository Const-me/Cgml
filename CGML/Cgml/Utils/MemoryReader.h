#pragma once
#include "../../ComLightLib/streams.h"

// Implementation of read-only stream COM interface over a continuous memory buffer
class MemoryReader : public ComLight::iReadStream
{
	const uint8_t* const buffer;
	const size_t length;
	size_t position;

	HRESULT COMLIGHTCALL read( void* lpBuffer, int nNumberOfBytesToRead, int& lpNumberOfBytesRead ) noexcept override final;
	HRESULT COMLIGHTCALL seek( int64_t offset, ComLight::eSeekOrigin origin ) noexcept override final;
	HRESULT COMLIGHTCALL getPosition( int64_t& position ) noexcept override final;
	HRESULT COMLIGHTCALL getLength( int64_t& length ) noexcept override final;

	HRESULT COMLIGHTCALL QueryInterface( REFIID riid, void** ppvObject ) noexcept override final { return E_NOTIMPL; }
	uint32_t COMLIGHTCALL AddRef() noexcept override final { return 1; }
	uint32_t COMLIGHTCALL Release() noexcept override final { return 1; }

public:
	MemoryReader( const void* rsi, size_t cb );
};