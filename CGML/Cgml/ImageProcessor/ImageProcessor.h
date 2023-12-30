#pragma once
#include "iImageProcessor.h"
#include <wincodec.h>

namespace Cgml
{
	class ImageProcessor: public iImageProcessor
	{
		CComPtr<ID3D11Device> device;
		CComPtr<ID3D11DeviceContext> context;
		ConstantBuffersPool& cbPool;

		HRESULT loadImage( iTensor* result, const sImageProcessorParams& ipp, ComLight::iReadStream* stream, uint32_t* previewPixels ) noexcept override final;

		CComPtr<ID3D11VertexShader> vsFullScreenTriangle;
		CComPtr<ID3D11PixelShader> psUpcastTexture;
		CComPtr<ID3D11PixelShader> psSample;
		CComPtr<ID3D11ComputeShader> csCopyAndNormalize;
		CComPtr<ID3D11SamplerState> samAnisotropic;
		CComPtr<IWICImagingFactory2> wicFactory;

		HRESULT createShaders();
		HRESULT createSampler();
		HRESULT createWicFactory();

		// FP32 RGBA texture
		class RenderTarget
		{
			std::array<uint32_t, 2> size = { 0, 0 };
			CComPtr<ID3D11RenderTargetView> rtv;
			CComPtr<ID3D11ShaderResourceView> srv;

		public:
			HRESULT createIfNeeded( const std::array<uint32_t, 2>& requestedSize, ID3D11Device* device );
			operator ID3D11RenderTargetView* ( ) const { return rtv; }
			operator ID3D11ShaderResourceView* ( ) const { return srv; }
		};
		RenderTarget renderTarget;

		// UINT8_UNORM render target for preview
		class PreviewTarget
		{
			std::array<uint32_t, 2> size = { 0, 0 };
			CComPtr<ID3D11RenderTargetView> rtv;
			CComPtr<ID3D11Texture2D> vram, staging;
		public:
			HRESULT createIfNeeded( const std::array<uint32_t, 2>& requestedSize, ID3D11Device* device );
			operator ID3D11RenderTargetView* ( ) const { return rtv; }
			HRESULT copyToSystemMemory( ID3D11DeviceContext* context, uint32_t* rdi ) const;
		};
		PreviewTarget previewTarget;

		struct Image
		{
			CComPtr<ID3D11ShaderResourceView> texture;
			std::array<uint32_t, 2> size = { 0, 0 };
		};

		// Decode source image into immutable RGBA8 texture in VRAM
		HRESULT decodeImage( Image& rdi, ComLight::iReadStream* stream );

		// Utility method to render a full-screen triangle
		void renderFullScreenTriangle( ID3D11RenderTargetView* rtv, const std::array<uint32_t, 2>& size );

		// Create another RGBA texture with FP16 pixels,
		// copy first most detailed mip level with a passthrough pixel shader,
		// and generate the pyramid of mip levels
		HRESULT generateMipMaps( Image& image );

		// Use anisotropic sampler to implement high quality image resize into the low-resolution output texture with FP32 pixels
		// The output is in renderTarget field of this class
		HRESULT sample( ID3D11ShaderResourceView* srv, const sImageProcessorParams& ipp );

		HRESULT makePreview( ID3D11ShaderResourceView* srv, const sImageProcessorParams& ipp, uint32_t* rdi );

		// Copy FP32 pixels from the texture in renderTarget field to the output tensor
		HRESULT copyAndNormalize( iTensor* result, const sImageProcessorParams& ipp );

	public:
		~ImageProcessor() override = default;
		ImageProcessor( ID3D11Device* dev, ID3D11DeviceContext* ctx, ConstantBuffersPool& cbp );
	};
}