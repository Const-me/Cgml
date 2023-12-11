#include "stdafx.h"
#include "ProfileCollection.h"
#include <algorithm>

namespace
{
	using namespace Cgml;

	ProfilerResult makeResult( uint32_t key, const ProfilerMeasure& src )
	{
		ProfilerResult res;
		res.what = (eProfilerMeasure)(uint8_t)( key >> 16 );
		res.id = (uint16_t)key;
		res.result = src;
		return res;
	}

	static uint32_t makeKey( uint16_t id, eProfilerMeasure m )
	{
		uint32_t res = (uint8_t)m;
		res = res << 16;
		return res | id;
	}

	struct SortResults
	{
		inline bool operator()( const ProfilerResult& a, const ProfilerResult& b )
		{
			if( a.what != b.what )
				return (uint8_t)a.what < (uint8_t)b.what;

			if( a.result.totalTicks != b.result.totalTicks )
				return a.result.totalTicks > b.result.totalTicks;

			return a.id < b.id;
		}
	};
}

Cgml::ProfilerMeasure& ProfileCollection::block( uint16_t id )
{
	const uint32_t key = makeKey( id, eProfilerMeasure::Block );
	return measures[ key ];
}

Cgml::ProfilerMeasure& ProfileCollection::shader( uint16_t id )
{
	const uint32_t key = makeKey( id, eProfilerMeasure::Shader );
	return measures[ key ];
}

HRESULT ProfileCollection::getData( Cgml::pfnProfilerData pfn, void* pv ) noexcept
{
	std::vector<ProfilerResult> vec;
	try
	{
		vec.reserve( measures.GetCount() );
	}
	catch( const std::bad_alloc& )
	{
		return E_OUTOFMEMORY;
	}

	for( POSITION pos = measures.GetStartPosition(); nullptr != pos; )
	{
		auto pair = measures.GetNext( pos );
		vec.emplace_back( makeResult( pair->m_key, pair->m_value ) );
	}

	if( vec.empty() )
		return S_FALSE;
	std::sort( vec.begin(), vec.end(), SortResults{} );
	return pfn( vec.data(), (int)vec.size(), pv );
}