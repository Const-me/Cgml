#pragma once
#include <stdint.h>

namespace Cgml
{
	struct sDeviceInfo
	{
		const wchar_t* name;
		uint64_t vram;
		uint16_t vendor;
		uint8_t featureLevelMajor, featureLevelMinor;
		uint8_t optionalFeatures;
	};
}