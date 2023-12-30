#pragma once
#include <array>

namespace Cgml
{
	struct sImageProcessorParams
	{
		std::array<uint32_t, 2> outputSize;
		std::array<float, 3> imageMean;
		std::array<float, 3> imageStd;
	};
}