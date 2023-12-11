#pragma once

namespace Cgml
{
	struct ProfilerMeasure
	{
		size_t count = 0;
		// 100-nanosecond ticks
		uint64_t totalTicks = 0;
		uint64_t max = 0;

		void reset()
		{
			count = 0;
			totalTicks = 0;
			max = 0;
		}

		void add( uint64_t val )
		{
			count++;
			totalTicks += val;
			max = std::max( max, val );
		}
	};

	enum struct eProfilerMeasure : uint8_t
	{
		Block = 1,
		Shader = 2,
	};

	struct ProfilerResult
	{
		eProfilerMeasure what;
		uint16_t id;
		ProfilerMeasure result;
	};

	using pfnProfilerData = HRESULT( __stdcall* )( const ProfilerResult* data, uint32_t length, void* pv );
}