#pragma once
#include <stdint.h>

namespace Cgml
{
	enum struct eDownloadFlag : uint8_t
	{
		None = 0,
		CopyToStaging = 1,
		ReadStaging = 2,
	};
}