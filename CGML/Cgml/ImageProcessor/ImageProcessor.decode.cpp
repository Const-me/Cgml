#include "stdafx.h"
#include "ImageProcessor.h"
#include "WICTextureLoader11.h"
using namespace Cgml;

HRESULT ImageProcessor::createWicFactory()
{
	HRESULT hr = CoCreateInstance( CLSID_WICImagingFactory2, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS( &wicFactory ) );
	return hr;
}

#include <atlcom.h>
namespace
{
	// Implementation of Win32 IStream COM interface on top of ComLight iReadStream interface
	// Allows to supply ComLight streams to WIC image decoders
	class StreamAdapter:
		public CComObjectRootEx<CComMultiThreadModel>,
		public IStream
	{
	public:
		ComLight::iReadStream* m_pReadStream = nullptr;

		BEGIN_COM_MAP( StreamAdapter )
			COM_INTERFACE_ENTRY( IStream )
		END_COM_MAP()

	private:
		// ISequentialStream methods
		HRESULT __stdcall Read( void* pv, ULONG cb, ULONG* pcbRead ) noexcept override final
		{
			if( !pv || !pcbRead )
				return E_POINTER;

			int bytesRead = 0;
			HRESULT hr = m_pReadStream->read( pv, cb, bytesRead );
			if( SUCCEEDED( hr ) )
			{
				*pcbRead = bytesRead;
				return S_OK;
			}
			return hr;
		}

		HRESULT __stdcall Write( const void*, ULONG, ULONG* ) noexcept override final
		{
			return E_NOTIMPL;
		}

		// IStream methods
		HRESULT __stdcall Seek( LARGE_INTEGER dlibMove, DWORD dwOrigin, ULARGE_INTEGER* plibNewPosition ) noexcept override final
		{
			ComLight::eSeekOrigin origin;
			bool skipSeekIfZero = false;
			switch( dwOrigin )
			{
			case STREAM_SEEK_SET:
				origin = ComLight::eSeekOrigin::Begin;
				break;
			case STREAM_SEEK_CUR:
				origin = ComLight::eSeekOrigin::Current;
				skipSeekIfZero = true;
				break;
			case STREAM_SEEK_END:
				origin = ComLight::eSeekOrigin::End;
				break;
			default:
				logError( u8"IStream.Seek - invalid origin constant %i", dwOrigin );
				return E_INVALIDARG;
			}

			int64_t move = (int64_t)dlibMove.QuadPart;
			if( 0 != move || !skipSeekIfZero )
				CHECK( m_pReadStream->seek( move, origin ) );

			if( nullptr != plibNewPosition )
			{
				int64_t newPos = 0;
				CHECK( m_pReadStream->getPosition( newPos ) );
				plibNewPosition->QuadPart = (uint64_t)newPos;
			}
			return S_OK;
		}

		HRESULT __stdcall SetSize( ULARGE_INTEGER ) noexcept override final
		{
			return E_NOTIMPL;
		}

		HRESULT __stdcall CopyTo( IStream*, ULARGE_INTEGER, ULARGE_INTEGER*, ULARGE_INTEGER* ) noexcept override final
		{
			return E_NOTIMPL;
		}

		HRESULT __stdcall Commit( DWORD ) noexcept override final
		{
			return E_NOTIMPL;
		}

		HRESULT __stdcall Revert() noexcept override final
		{
			return E_NOTIMPL;
		}

		HRESULT __stdcall LockRegion( ULARGE_INTEGER, ULARGE_INTEGER, DWORD ) noexcept override final
		{
			return E_NOTIMPL;
		}

		HRESULT __stdcall UnlockRegion( ULARGE_INTEGER, ULARGE_INTEGER, DWORD ) noexcept override final
		{
			return E_NOTIMPL;
		}

		HRESULT __stdcall Stat( STATSTG* pstatstg, DWORD grfStatFlag ) noexcept override final
		{
			// Zero fill
			__stosd( (DWORD*)pstatstg, 0, sizeof( STATSTG ) / 4 );
			// Set the type
			pstatstg->type = STGTY_STREAM;
			// Query and set length of the stream in bytes
			int64_t length = 0;
			CHECK( m_pReadStream->getLength( length ) );
			pstatstg->cbSize.QuadPart = (uint64_t)length;

			// We don't know any other fields of the STATSTG structure
			return S_OK;
		}

		HRESULT __stdcall Clone( IStream** ) noexcept override final
		{
			return E_NOTIMPL;
		}
	};

	thread_local IWICImagingFactory2* ts_wicFactory = nullptr;
	struct SetThreadFactory
	{
		SetThreadFactory( IWICImagingFactory2* wf ) { ts_wicFactory = wf; }
		~SetThreadFactory() { ts_wicFactory = nullptr; }
	};
}

IWICImagingFactory2* DirectX::GetWIC() noexcept
{
	return ts_wicFactory;
}

HRESULT ImageProcessor::decodeImage( Image& rdi, ComLight::iReadStream* stream )
{
	HRESULT hr;
	CComObjectStackEx<StreamAdapter> sa;
	sa.m_pReadStream = stream;

	CComPtr<IWICBitmapDecoder> bitmapDecoder;
	hr = wicFactory->CreateDecoderFromStream( &sa, nullptr, WICDecodeMetadataCacheOnLoad, &bitmapDecoder );
	if( FAILED( hr ) )
		return hr;

	CComPtr<IWICBitmapFrameDecode> frame;
	hr = bitmapDecoder->GetFrame( 0, &frame );
	if( FAILED( hr ) )
		return hr;

	UINT width, height;
	hr = frame->GetSize( &width, &height );
	if( FAILED( hr ) )
		return hr;

	CComPtr<ID3D11ShaderResourceView> textureView;
	SetThreadFactory stf{ wicFactory };
	hr = DirectX::createTextureFromWIC( device, frame, &textureView );
	if( FAILED( hr ) )
		return hr;

	rdi.size[ 0 ] = width;
	rdi.size[ 1 ] = height;
	rdi.texture.Attach( textureView.Detach() );
	logDebug( u8"Decoded image from the stream, %i×%i pixels", width, height );
	return S_OK;
}