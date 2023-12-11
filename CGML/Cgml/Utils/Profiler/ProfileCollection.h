#pragma once
#include "../../API/profiler.h"
#include <atlcoll.h>

class ProfileCollection
{
	CAtlMap<uint32_t, Cgml::ProfilerMeasure> measures;

public:

	Cgml::ProfilerMeasure& block( uint16_t id );
	Cgml::ProfilerMeasure& shader( uint16_t id );

	HRESULT getData( Cgml::pfnProfilerData pfn, void* pv ) noexcept;
};