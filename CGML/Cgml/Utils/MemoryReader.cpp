#include "stdafx.h"
#include "MemoryReader.h"

MemoryReader::MemoryReader( const void* rsi, size_t cb ) :
	buffer( (const uint8_t*)rsi ),
	length( cb ),
	position( 0 )
{ }

HRESULT COMLIGHTCALL MemoryReader::read( void* lpBuffer, int nNumberOfBytesToRead, int& lpNumberOfBytesRead ) noexcept
{
	ptrdiff_t bytesLeft = (ptrdiff_t)length - position;
	if( bytesLeft <= 0 )
	{
		lpNumberOfBytesRead = 0;
		return S_OK;
	}

	ptrdiff_t bytesToCopy = std::min( bytesLeft, (ptrdiff_t)nNumberOfBytesToRead );
	__movsb( (BYTE*)lpBuffer, buffer + position, bytesToCopy );
	position += bytesToCopy;
	lpNumberOfBytesRead = (int)bytesToCopy;
	return S_OK;
}

HRESULT COMLIGHTCALL MemoryReader::seek( int64_t offset, ComLight::eSeekOrigin origin ) noexcept
{
	return E_NOTIMPL;
}

HRESULT COMLIGHTCALL MemoryReader::getPosition( int64_t& position ) noexcept
{
	position = this->position;
	return S_OK;
}

HRESULT COMLIGHTCALL MemoryReader::getLength( int64_t& length ) noexcept
{
	length = this->length;
	return S_OK;
}