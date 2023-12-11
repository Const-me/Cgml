#pragma once
#include "../../API/sTensorDesc.h"

namespace Bcml1
{
	constexpr uint32_t PANEL_HEIGHT = 64;

	using namespace Cgml;

	HRESULT makeDesc( sTensorDesc& rdi, const sTensorDesc& rsi );

	HRESULT compress( eDataType sourceType, const sTensorDesc& compressedDesc, const std::vector<__m256i>& sourceVector, std::vector<uint32_t>& result );

	enum struct eCpuExtensionFlags: uint8_t
	{
		AVX2 = 1,
		F16C = 2,
		BMI2 = 4,
	};
	inline eCpuExtensionFlags operator|( eCpuExtensionFlags a, eCpuExtensionFlags b )
	{
		return (eCpuExtensionFlags)( (uint8_t)a | (uint8_t)b );
	}
	inline void operator|=( eCpuExtensionFlags& a, eCpuExtensionFlags b )
	{
		a = a | b;
	}

	// Return true when the current CPU supports each and every required extension flag
	bool checkExtensionFlags( eCpuExtensionFlags required );
};