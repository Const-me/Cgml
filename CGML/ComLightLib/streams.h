#pragma once
#include <vector>
#include "comLightCommon.h"

// COM interfaces to marshal streams across the interop.
namespace ComLight
{
	enum struct eSeekOrigin : uint8_t
	{
		Begin = 0,
		Current = 1,
		End = 2
	};

	namespace details
	{
		template<class E>
		inline size_t sizeofVector( const std::vector<E>& vec )
		{
			return sizeof( E ) * vec.size();
		}
	}

	// COM interface for readonly stream. You'll get these interfaces what you use [ReadStream] attribute in C#.
	struct DECLSPEC_NOVTABLE iReadStream : public IUnknown
	{
		DEFINE_INTERFACE_ID( "006af6db-734e-4595-8c94-19304b2389ac" );

	public:
		virtual HRESULT COMLIGHTCALL read( void* lpBuffer, int nNumberOfBytesToRead, int& lpNumberOfBytesRead ) = 0;
		virtual HRESULT COMLIGHTCALL seek( int64_t offset, eSeekOrigin origin ) = 0;
		virtual HRESULT COMLIGHTCALL getPosition( int64_t& position ) = 0;
		virtual HRESULT COMLIGHTCALL getLength( int64_t& length ) = 0;

		HRESULT read( void* pv, size_t cb )
		{
			ptrdiff_t cbLeft = cb;
			if( 0 == cbLeft )
				return S_OK;

			uint8_t* rdi = (uint8_t*)pv;
			while( cbLeft > 0 )
			{
				const int cbRequested = (int)std::min( cbLeft, (ptrdiff_t)( 1u << 20 ) );
				int cbRead = 0;
				CHECK( read( rdi, cbRequested, cbRead ) );
				if( 0 == cbRead )
					return E_EOF;
				cbLeft -= cbRead;
				rdi += cbRead;
			}
			return S_OK;
		}

		template<class E>
		inline HRESULT read( std::vector<E>& vec )
		{
			size_t cb = details::sizeofVector( vec );
			if( 0 != cb )
				return read( vec.data(), cb );
			return S_OK;
		}
	};

	// COM interface for readonly stream. You'll get these interfaces what you use [WriteStream] attribute in C#.
	struct DECLSPEC_NOVTABLE iWriteStream : public IUnknown
	{
		DEFINE_INTERFACE_ID( "d7c3eb39-9170-43b9-ba98-2ea1f2fed8a8" );

		virtual HRESULT COMLIGHTCALL write( const void* lpBuffer, int nNumberOfBytesToWrite ) = 0;
		virtual HRESULT COMLIGHTCALL flush() = 0;

		template<class E>
		inline HRESULT write( const std::vector<E>& vec )
		{
			const int cb = (int)details::sizeofVector( vec );
			return write( vec.data(), cb );
		}
	};
}