#pragma once
#include <memory>
#include <API/iTensor.cl.h>
#include <API/sImageProcessorParams.h>
#include "../../ComLightLib/streams.h"
class ConstantBuffersPool;

namespace Cgml
{
	struct iImageProcessor
	{
		virtual ~iImageProcessor() { }

		virtual HRESULT loadImage( iTensor* result, const sImageProcessorParams& ipp, ComLight::iReadStream* stream, uint32_t* previewPixels ) = 0;

		static HRESULT create( std::unique_ptr<iImageProcessor>& rdi, ID3D11Device* device, ID3D11DeviceContext* context, ConstantBuffersPool& cbPool );
	};
}