#include "stdafx.h"
#include "ImageProcessor.h"
#include <DirectXMath.h>
#include <D3D/ConstantBuffersPool.h>
#include <D3D/tensorInterop.h>
#include <D3D/RenderDoc/renderDoc.h>
#include <mfapi.h>
#pragma comment(lib, "Mfplat.lib")
using namespace Cgml;

ImageProcessor::ImageProcessor( ID3D11Device* dev, ID3D11DeviceContext* ctx, ConstantBuffersPool& cbp ):
	device( dev ),
	context( ctx ),
	cbPool( cbp )
{
	HRESULT hr = createShaders();
	if( FAILED( hr ) )
		throw hr;
	hr = createSampler();
	if( FAILED( hr ) )
		throw hr;

	hr = createWicFactory();
	if( FAILED( hr ) )
		throw hr;
}

HRESULT iImageProcessor::create( std::unique_ptr<iImageProcessor>& rdi, ID3D11Device* device, ID3D11DeviceContext* context, ConstantBuffersPool& cbPool )
{
	try
	{
		rdi = std::make_unique<ImageProcessor>( device, context, cbPool );
		return S_OK;
	}
	catch( HRESULT hr )
	{
		return hr;
	}
}

HRESULT ImageProcessor::createSampler()
{
	CD3D11_SAMPLER_DESC desc{ CD3D11_DEFAULT{} };
	desc.Filter = D3D11_FILTER_ANISOTROPIC;
	desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
	desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
	desc.MaxAnisotropy = 16;
	return device->CreateSamplerState( &desc, &samAnisotropic );
}

HRESULT ImageProcessor::RenderTarget::createIfNeeded( const std::array<uint32_t, 2>& requestedSize, ID3D11Device* device )
{
	const uint64_t reqested = *(const uint64_t*)requestedSize.data();
	const uint64_t current = *(const uint64_t*)size.data();
	if( reqested == current )
		return S_OK;

	rtv = nullptr;
	srv = nullptr;

	constexpr DXGI_FORMAT format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	CD3D11_TEXTURE2D_DESC textureDesc
	{
		format,
		requestedSize[ 0 ], requestedSize[ 1 ],
		1, 1, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE
	};
	CComPtr<ID3D11Texture2D> tex;
	CHECK( device->CreateTexture2D( &textureDesc, nullptr, &tex ) );

	CD3D11_RENDER_TARGET_VIEW_DESC rtvDesc{ D3D11_RTV_DIMENSION_TEXTURE2D, format };
	CHECK( device->CreateRenderTargetView( tex, &rtvDesc, &rtv ) );

	CD3D11_SHADER_RESOURCE_VIEW_DESC srvDesc{ D3D_SRV_DIMENSION_TEXTURE2D, format };
	CHECK( device->CreateShaderResourceView( tex, &srvDesc, &srv ) );

	size = requestedSize;
	return S_OK;
}

HRESULT ImageProcessor::PreviewTarget::createIfNeeded( const std::array<uint32_t, 2>& requestedSize, ID3D11Device* device )
{
	const uint64_t reqested = *(const uint64_t*)requestedSize.data();
	const uint64_t current = *(const uint64_t*)size.data();
	if( reqested == current )
		return S_OK;

	rtv = nullptr;
	vram = nullptr;
	staging = nullptr;

	constexpr DXGI_FORMAT format = DXGI_FORMAT_B8G8R8A8_UNORM;
	CD3D11_TEXTURE2D_DESC textureDesc
	{
		format,
		requestedSize[ 0 ], requestedSize[ 1 ],
		1, 1, D3D11_BIND_RENDER_TARGET
	};
	CHECK( device->CreateTexture2D( &textureDesc, nullptr, &vram ) );

	textureDesc.Usage = D3D11_USAGE_STAGING;
	textureDesc.BindFlags = 0;
	textureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	CHECK( device->CreateTexture2D( &textureDesc, nullptr, &staging ) );

	CD3D11_RENDER_TARGET_VIEW_DESC rtvDesc{ D3D11_RTV_DIMENSION_TEXTURE2D, format };
	CHECK( device->CreateRenderTargetView( vram, &rtvDesc, &rtv ) );

	size = requestedSize;
	return S_OK;
}

HRESULT ImageProcessor::PreviewTarget::copyToSystemMemory( ID3D11DeviceContext* context, uint32_t* rdi ) const
{
	context->CopyResource( staging, vram );

	D3D11_MAPPED_SUBRESOURCE mapped;
	CHECK( context->Map( staging, 0, D3D11_MAP_READ, 0, &mapped ) );

	const DWORD width = size[ 0 ] * 4;
	HRESULT hr = MFCopyImage( (BYTE*)rdi, width, (const BYTE*)mapped.pData, mapped.RowPitch, width, size[ 1 ] );

	context->Unmap( staging, 0 );
	CHECK( hr );
	return S_OK;
}

static HRESULT validateResult( iTensor* result, const sImageProcessorParams& ipp )
{
	Cgml::sTensorDesc resultDesc;
	CHECK( result->getDesc( resultDesc ) );

	switch( resultDesc.dataType )
	{
	case eDataType::FP16:
	case eDataType::FP32:
		break;
	default:
		logError( u8"iContext.loadImage: output tensor must be FP32 or FP16" );
		return E_INVALIDARG;
	}
	if( resultDesc.layout != eTensorLayout::Dense )
	{
		logError( u8"iContext.loadImage: output tensor must be dense" );
		return E_INVALIDARG;
	}

	const auto& size = resultDesc.shape.size;
	if( size[ 0 ] != ipp.outputSize[ 0 ]
		|| size[ 1 ] != ipp.outputSize[ 1 ]
		|| size[ 2 ] != 3 || size[ 3 ] != 1 )
	{
		logError( u8"iContext.loadImage: output tensor size doesn't match the requested output size" );
		return E_INVALIDARG;
	}

	switch( resultDesc.usage )
	{
	case eBufferUse::ReadWrite:
	case eBufferUse::ReadWriteDownload:
		break;
	default:
		logError( u8"iContext.loadImage: output tensor doesn't support writes from GPU" );
		return E_INVALIDARG;
	}

	return S_OK;
}

void ImageProcessor::renderFullScreenTriangle( ID3D11RenderTargetView* rtv, const std::array<uint32_t, 2>& size )
{
	context->VSSetShader( vsFullScreenTriangle, nullptr, 0 );
	context->OMSetRenderTargets( 1, &rtv, nullptr );

	CD3D11_VIEWPORT viewport
	{
		0.0f,
		0.0f,
		(float)(int)size[ 0 ],
		(float)(int)size[ 1 ]
	};
	context->RSSetViewports( 1, &viewport );
	context->IASetPrimitiveTopology( D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST );
	context->Draw( 3, 0 );

	rtv = nullptr;
	context->OMSetRenderTargets( 1, &rtv, nullptr );
}

inline void psSetResource( ID3D11DeviceContext* context, ID3D11ShaderResourceView* srv )
{
	context->PSSetShaderResources( 0, 1, &srv );
}

HRESULT ImageProcessor::generateMipMaps( Image& image )
{
	// Create the output texture
	constexpr DXGI_FORMAT format = DXGI_FORMAT_R16G16B16A16_FLOAT;
	constexpr uint32_t bindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
	CD3D11_TEXTURE2D_DESC textureDesc{ format, image.size[ 0 ], image.size[ 1 ], 1, 0, bindFlags };
	textureDesc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;

	CComPtr<ID3D11Texture2D> texture;
	CHECK( device->CreateTexture2D( &textureDesc, nullptr, &texture ) );

	// Create render target view
	CD3D11_RENDER_TARGET_VIEW_DESC rtDesc{ D3D11_RTV_DIMENSION_TEXTURE2D, format };
	CComPtr<ID3D11RenderTargetView> rtv;
	CHECK( device->CreateRenderTargetView( texture, &rtDesc, &rtv ) );

	// Dispatch passthrough pixel shader to upcast the pixels
	context->PSSetShader( psUpcastTexture, nullptr, 0 );
	psSetResource( context, image.texture );
	renderFullScreenTriangle( rtv, image.size );

	// Create shader resource view
	CD3D11_SHADER_RESOURCE_VIEW_DESC srvDesc{ D3D_SRV_DIMENSION_TEXTURE2D , format };
	CComPtr<ID3D11ShaderResourceView> srv;
	CHECK( device->CreateShaderResourceView( texture, &srvDesc, &srv ) );

	// Generate pyramid of mip levels
	context->GenerateMips( srv );

	// Replace texture view in the image structure
	image.texture = nullptr;
	image.texture.Attach( srv.Detach() );
	return S_OK;
}

namespace
{
	constexpr int insertMask( int sourceLane, int destLane, int zeroLanesMask )
	{
		assert( sourceLane >= 0 && sourceLane < 4 );
		assert( destLane >= 0 && destLane < 4 );
		assert( zeroLanesMask >= 0 && zeroLanesMask < 16 );
		return ( sourceLane << 6 ) | ( destLane << 4 ) | zeroLanesMask;
	}

	inline __m128 loadFloat3( const float* rsi )
	{
		__m128 xy = _mm_castpd_ps( _mm_load_sd( (const double*)rsi ) );
		__m128 z = _mm_load_ss( ( (const float*)rsi ) + 2 );
		return _mm_insert_ps( xy, z, insertMask( 0, 2, 0 ) );
	}

	// Content of the constant buffer in slot #0 of CopyAndNormalizeCS.hlsl compute shader
	class CopyAndNormalizeConstants
	{
		uint32_t width;
		uint32_t layerStride;
		uint64_t zzPadding;
		std::array<float, 4> mean, stdInv;

	public:
		CopyAndNormalizeConstants( const sImageProcessorParams& ipp )
		{
			width = ipp.outputSize[ 0 ];
			layerStride = ipp.outputSize[ 0 ] * ipp.outputSize[ 1 ];
			zzPadding = 0;

			// Load imageMean field, because it's not the last structure element can load a full vector
			__m128 vec = _mm_loadu_ps( ipp.imageMean.data() );
			// Reset w lane to 0.0
			vec = _mm_insert_ps( vec, vec, insertMask( 0, 0, 0b1000 ) );
			// Store
			_mm_storeu_ps( mean.data(), vec );

			// Load imageStd field; it's at the end of the structure, need to load exactly 3 floats.
			vec = loadFloat3( ipp.imageStd.data() );
			// Reset w to 1.0, avoids division by 0
			const __m128 one = DirectX::g_XMOne;
			vec = _mm_blend_ps( vec, one, 0b1000 );
			// Compute inverse of the value
			vec = _mm_div_ps( one, vec );
			// Store
			_mm_storeu_ps( stdInv.data(), vec );
		}

		operator const uint8_t* ( ) const
		{
			return (const uint8_t*)( this );
		}
	};
}

HRESULT ImageProcessor::sample( ID3D11ShaderResourceView* srv, const sImageProcessorParams& ipp )
{
	CHECK( renderTarget.createIfNeeded( ipp.outputSize, device ) );

	ID3D11SamplerState* ss = samAnisotropic;
	context->PSSetSamplers( 0, 1, &ss );
	psSetResource( context, srv );
	context->PSSetShader( psSample, nullptr, 0 );

	renderFullScreenTriangle( renderTarget, ipp.outputSize );

	psSetResource( context, nullptr );
	ss = nullptr;
	context->PSSetSamplers( 0, 1, &ss );
	return S_OK;
}

HRESULT ImageProcessor::makePreview( ID3D11ShaderResourceView* srv, const sImageProcessorParams& ipp, uint32_t* rdi )
{
	CHECK( previewTarget.createIfNeeded( ipp.outputSize, device ) );

	ID3D11SamplerState* ss = samAnisotropic;
	context->PSSetSamplers( 0, 1, &ss );
	psSetResource( context, srv );
	context->PSSetShader( psSample, nullptr, 0 );

	renderFullScreenTriangle( previewTarget, ipp.outputSize );

	psSetResource( context, nullptr );
	ss = nullptr;
	context->PSSetSamplers( 0, 1, &ss );

	CHECK( previewTarget.copyToSystemMemory( context, rdi ) );
	return S_OK;
}

HRESULT ImageProcessor::copyAndNormalize( iTensor* result, const sImageProcessorParams& ipp )
{
	CopyAndNormalizeConstants cbData{ ipp };
	CHECK( cbPool.updateAndBind( device, context, cbData, sizeof( cbData ) ) );
	context->CSSetShader( csCopyAndNormalize, nullptr, 0 );

	ID3D11ShaderResourceView* const srv = renderTarget;
	context->CSSetShaderResources( 0, 1, &srv );

	ID3D11UnorderedAccessView* const uav = Cgml::getTensorUav( result );
	context->CSSetUnorderedAccessViews( 0, 1, &uav, nullptr );

	context->Dispatch( ipp.outputSize[ 1 ], 1, 1 );
	return S_OK;
}

HRESULT ImageProcessor::loadImage( iTensor* result, const sImageProcessorParams& ipp, ComLight::iReadStream* stream, uint32_t* previewPixels ) noexcept
{
	if( result == nullptr || stream == nullptr )
		return E_POINTER;
	CHECK( validateResult( result, ipp ) );

	DirectCompute::CaptureRaii renderdoc{ device };

	Image image;
	CHECK( decodeImage( image, stream ) );

	CHECK( generateMipMaps( image ) );

	CHECK( sample( image.texture, ipp ) );

	CHECK( copyAndNormalize( result, ipp ) );

	if( nullptr != previewPixels )
		CHECK( makePreview( image.texture, ipp, previewPixels ) );

	logDebug( u8"Processed image into the tensor, %i×%i×3 elements",
		ipp.outputSize[ 0 ], ipp.outputSize[ 1 ] );
	return S_OK;
}