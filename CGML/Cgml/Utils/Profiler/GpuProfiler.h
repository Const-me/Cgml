#pragma once
#include "ProfileCollection.h"
#include "DelayExecution.h"

class GpuProfiler
{
	ID3D11Device* const device;
	ID3D11DeviceContext* const context;
	DelayExecution delay;
	CComPtr<ID3D11Query> disjoint;

	enum struct eEvent: uint8_t
	{
		None = 0,
		BlockStart,
		BlockEnd,
		Shader
	};

	struct BlockState;
	static constexpr uint16_t EmptyShader = ~(uint16_t)0;

	// A circular buffer with in-flight queries which feeds timestamps into the iTimestampSink interface
	class Queue
	{
		// Ring buffer for individual measures
		struct Entry
		{
			CComPtr<ID3D11Query> query;
			BlockState* block;
			eEvent event;
			uint16_t shader;
#if PROFILER_COLLECT_TAGS
			uint16_t tag = 0;
#endif
			void join( GpuProfiler& owner );
		};

		GpuProfiler& owner;
		std::vector<Entry> queue;
		size_t nextEntry = 0;

	public:
		Queue( GpuProfiler& gp, size_t queueLength ) : owner( gp )
		{
			queue.resize( queueLength );
		}

		HRESULT create();

		// Begin a next query. Eventually, this will result in the BlockState.haveTimestamp callback
		void submit( BlockState* block, eEvent evt, uint16_t shader = EmptyShader, uint16_t tag = 0 );

		// Wait for all the pending queries, and call their callbacks
		void join();
	};
	Queue queries;

	struct sProfilerData;
	struct BlockState
	{
		int64_t timeStart = -1;
		sProfilerData* destBlock = nullptr;
		int64_t shaderStart = -1;
		uint16_t prevShader = EmptyShader;
		uint16_t prevShaderTag = 0;
		BlockState* parentBlock = nullptr;
		void haveTimestamp( eEvent evt, uint16_t cs, uint16_t tag, uint64_t time, GpuProfiler& profiler );
	private:
		void completePrevShader( uint64_t time, GpuProfiler& profiler );
	};
	CAtlMap<uint16_t, BlockState> blockStates;
	std::vector<BlockState*> stack;

	struct sProfilerData
	{
		// Count of accumulated measures
		size_t callsPending;
		// Total time spent running all instances of that measure, expressed in GPU ticks
		uint64_t timePending;
		// Maximum time spent, expressed in GPU ticks
		uint64_t timeMax;

		Cgml::ProfilerMeasure* dest;

		inline void makeTime( uint64_t freq );
		inline void addPending( int64_t time );
		inline void reset();

		sProfilerData()
		{
			reset();
		}
	};

	// This collection accumulates measures where we already have time, but don't yet have the disjoint data,
	// which scales ticks from custom to normal units.
	// For shaders, the hash map key is shader ID. For blocks, the hash map key is bit-flipped block ID.
	CAtlMap<uint16_t, sProfilerData> results;
#if PROFILER_COLLECT_TAGS
	CAtlMap<uint32_t, sProfilerData> resultsTagged;
#endif

	ProfileCollection collection;

	void resultsMakeTime( uint64_t freq );
	void resultsReset();

public:

	GpuProfiler( ID3D11Device* dev, ID3D11DeviceContext* ctx, size_t queueLength, bool powerSaver );

	HRESULT create( size_t maxDepth = 8 );

	HRESULT computeShader( uint16_t cs ) noexcept;

	HRESULT blockStart( uint16_t which ) noexcept;
	HRESULT blockEnd() noexcept;

	HRESULT getData( Cgml::pfnProfilerData pfn, void* pv ) noexcept;
};