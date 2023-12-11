#include "stdafx.h"
#include "GpuProfiler.h"
#include "../../../ComLightLib/hresult.h"

inline void GpuProfiler::sProfilerData::reset()
{
	_mm_storeu_si128( ( __m128i* ) ( &callsPending ), _mm_setzero_si128() );
	timeMax = 0;
}

inline void GpuProfiler::sProfilerData::addPending( int64_t time )
{
	callsPending++;
	timePending += time;
	timeMax = std::max( timeMax, (uint64_t)time );
}

inline void GpuProfiler::sProfilerData::makeTime( uint64_t freq )
{
	dest->count += callsPending;
	dest->totalTicks += ::makeTime( timePending, freq );
	dest->max = std::max( dest->max, ::makeTime( timeMax, freq ) );

	reset();
}

HRESULT GpuProfiler::Queue::create()
{
	ID3D11Device* const dev = owner.device;

	CD3D11_QUERY_DESC desc{ D3D11_QUERY_TIMESTAMP };
	for( Entry& e : queue )
	{
		CHECK( dev->CreateQuery( &desc, &e.query ) );
		e.block = nullptr;
		e.event = eEvent::None;
		e.shader = EmptyShader;
	}
	return S_OK;
}

namespace
{
	inline void check( HRESULT hr )
	{
		if( SUCCEEDED( hr ) )
			return;
		throw hr;
	}

	static uint64_t getTimestamp( ID3D11DeviceContext* ctx, ID3D11Query* query, const DelayExecution& delay )
	{
		uint64_t res = 0;
		while( true )
		{
			const HRESULT hr = ctx->GetData( query, &res, sizeof( uint64_t ), 0 );
			check( hr );
			if( S_OK == hr )
				return res;
			delay.delay();
		}
	}

	static D3D11_QUERY_DATA_TIMESTAMP_DISJOINT waitForDisjointData( ID3D11DeviceContext* ctx, ID3D11Query* query )
	{
		ctx->End( query );

		D3D11_QUERY_DATA_TIMESTAMP_DISJOINT res;
		while( true )
		{
			const HRESULT hr = ctx->GetData( query, &res, sizeof( D3D11_QUERY_DATA_TIMESTAMP_DISJOINT ), 0 );
			check( hr );
			if( S_OK == hr )
				return res;
			Sleep( 1 );
		}
	}
}

void GpuProfiler::Queue::Entry::join( GpuProfiler& owner )
{
	assert( nullptr != block );

	uint64_t res = getTimestamp( owner.context, query, owner.delay );
#if PROFILER_COLLECT_TAGS
	block->haveTimestamp( event, shader, tag, res, owner );
#else
	block->haveTimestamp( event, shader, 0, res, owner );
#endif
	block = nullptr;
	event = eEvent::None;
	shader = EmptyShader;
}

void GpuProfiler::Queue::submit( BlockState* block, eEvent evt, uint16_t shader, uint16_t tag )
{
	// if( evt == GpuProfiler::eEvent::Shader && shader == 0 ) __debugbreak();
	assert( nullptr != block );

	Entry& e = queue[ nextEntry ];
	if( nullptr != e.block )
		e.join( owner );

	e.block = block;
	e.event = evt;
	e.shader = shader;
#if PROFILER_COLLECT_TAGS
	e.tag = tag;
#endif
	owner.context->End( e.query );
	nextEntry = ( nextEntry + 1 ) % queue.size();
}

void GpuProfiler::Queue::join()
{
	while( true )
	{
		Entry& e = queue[ nextEntry ];
		if( nullptr == e.block )
			return;
		e.join( owner );
		nextEntry = ( nextEntry + 1 ) % queue.size();
	}
}

static inline uint32_t makeTagKey( uint16_t cs, uint16_t tag )
{
	uint32_t r = cs;
	r = r << 16;
	r |= tag;
	return r;
}

void GpuProfiler::BlockState::completePrevShader( uint64_t time, GpuProfiler& profiler )
{
	if( shaderStart == -1 )
		return;
	assert( prevShader != EmptyShader );
	const int64_t elapsed = (int64_t)time - shaderStart;

	sProfilerData* dest = nullptr;
	auto* p = profiler.results.Lookup( prevShader );
	if( nullptr != p )
		dest = &p->m_value;
	else
	{
		sProfilerData& res = profiler.results[ prevShader ];
		res.dest = &profiler.collection.shader( prevShader );
		dest = &res;
	}
	dest->addPending( elapsed );

#if PROFILER_COLLECT_TAGS
	if( 0 != prevShaderTag )
	{
		const uint32_t key = makeTagKey( prevShader, prevShaderTag );
		auto* pt = profiler.resultsTagged.Lookup( key );
		if( nullptr != pt )
			dest = &pt->m_value;
		else
		{
			sProfilerData& res = profiler.resultsTagged[ key ];
			res.dest = &profiler.dest.measure( (eComputeShader)prevShader, prevShaderTag );
			dest = &res;
		}
		dest->addPending( elapsed );
	}
#endif
	prevShader = EmptyShader;
	prevShaderTag = 0;
	shaderStart = -1;
}

void GpuProfiler::BlockState::haveTimestamp( eEvent evt, uint16_t cs, uint16_t tag, uint64_t time, GpuProfiler& profiler )
{
	switch( evt )
	{
	case eEvent::BlockStart:
		assert( -1 == timeStart );
		assert( -1 == shaderStart );
		assert( cs == EmptyShader );
		timeStart = (int64_t)time;
		if( nullptr != parentBlock )
			parentBlock->completePrevShader( time, profiler );
		return;
	case eEvent::BlockEnd:
		assert( -1 != timeStart );
		assert( cs == EmptyShader );
		completePrevShader( time, profiler );
		destBlock->addPending( (int64_t)time - timeStart );
		timeStart = -1;
		return;
	case eEvent::Shader:
		assert( cs != EmptyShader );
		// if( cs == (uint16_t)0 ) __debugbreak();
		completePrevShader( time, profiler );
		prevShader = cs;
		prevShaderTag = tag;
		shaderStart = (int64_t)time;
		return;
	}
	assert( false );
}

GpuProfiler::GpuProfiler( ID3D11Device* dev, ID3D11DeviceContext* ctx, size_t queueLength, bool powerSaver ):
	device( dev ), context( ctx ), delay( powerSaver ), queries( *this, queueLength )
{ };

HRESULT GpuProfiler::create( size_t maxDepth )
{
	CD3D11_QUERY_DESC desc{ D3D11_QUERY_TIMESTAMP_DISJOINT };
	CHECK( device->CreateQuery( &desc, &disjoint ) );
	CHECK( queries.create() );
	stack.reserve( maxDepth );
	return S_OK;
}

HRESULT GpuProfiler::blockStart( uint16_t which ) noexcept
{
	try
	{
		BlockState* parentBlock;
		if( stack.empty() )
		{
			context->Begin( disjoint );
			parentBlock = nullptr;
		}
		else
			parentBlock = *stack.rbegin();

		BlockState* bs = nullptr;
		auto p = blockStates.Lookup( which );
		if( nullptr != p )
			bs = &p->m_value;
		else
		{
			BlockState& block = blockStates[ which ];
			block.destBlock = &results[ ~which ];
			block.destBlock->dest = &collection.block( which );
			bs = &block;
		}
		bs->parentBlock = parentBlock;
		queries.submit( bs, eEvent::BlockStart );
		stack.push_back( bs );
		return S_OK;
	}
	catch( HRESULT hr )
	{
		return hr;
	}
}

HRESULT GpuProfiler::blockEnd() noexcept
{
	try
	{
		assert( !stack.empty() );
		BlockState* const bs = *stack.rbegin();
		queries.submit( bs, eEvent::BlockEnd );
		stack.pop_back();

		if( !stack.empty() )
			return S_OK;

		const D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dtsd = waitForDisjointData( context, disjoint );
		queries.join();

		if( !dtsd.Disjoint )
		{
			// Fortunately, these timers appear to be relatively high resolution.
			// Specifically, on the iGPU inside Ryzen 7 5700G that frequency is 1E+8 = 100 MHz
			// On nVidia 1080Ti, that frequency is 1E+9 = 1 GHz
			const uint64_t freq = dtsd.Frequency;
			resultsMakeTime( freq );
		}
		else
		{
			// Something occurred in between the query's ID3D11DeviceContext::Begin and ID3D11DeviceContext::End calls 
			// that caused the timestamp counter to become discontinuous or disjoint, such as unplugging the AC cord on a laptop, overheating, or throttling up/down due to laptop savings events.
			// The timestamp returned by ID3D11DeviceContext::GetData for a timestamp query is only reliable if Disjoint is FALSE.
			resultsReset();
		}
		return S_OK;
	}
	catch( HRESULT hr )
	{
		return hr;
	}
}

HRESULT GpuProfiler::computeShader( uint16_t cs ) noexcept
{
	if( stack.empty() )
	{
		logError( u8"You can only dispatch compute shaders while inside a profiler block" );
		return E_FAIL;
	}

	try
	{
		BlockState* const bs = *stack.rbegin();
#if PROFILER_COLLECT_TAGS
		queries.submit( bs, eEvent::Shader, (uint16_t)cs, m_nextTag );
		m_nextTag = 0;
#else
		queries.submit( bs, eEvent::Shader, (uint16_t)cs );
#endif
		return S_OK;
	}
	catch( HRESULT hr )
	{
		return hr;
	}
}

void GpuProfiler::resultsMakeTime( uint64_t freq )
{
	for( POSITION pos = results.GetStartPosition(); nullptr != pos; )
		results.GetNextValue( pos ).makeTime( freq );
#if PROFILER_COLLECT_TAGS
	for( POSITION pos = resultsTagged.GetStartPosition(); nullptr != pos; )
		resultsTagged.GetNextValue( pos ).makeTime( freq );
#endif
}

void GpuProfiler::resultsReset()
{
	for( POSITION pos = results.GetStartPosition(); nullptr != pos; )
		results.GetNextValue( pos ).reset();
#if PROFILER_COLLECT_TAGS
	for( POSITION pos = resultsTagged.GetStartPosition(); nullptr != pos; )
		resultsTagged.GetNextValue( pos ).reset();
#endif
}

HRESULT GpuProfiler::getData( Cgml::pfnProfilerData pfn, void* pv ) noexcept
{
	if( !stack.empty() )
	{
		logError( u8"iContext.profilerGetData should be called after the measures have stopped" );
		return E_FAIL;
	}

	return collection.getData( pfn, pv );
}