namespace Mistral.Model;
using Cgml;
using System;
using System.Runtime.InteropServices;
#pragma warning disable CS0162 // Unreachable code detected, due to bfloat16 constant                                                        

/// <summary>Implements these ML algorithms, and contains a cache with GPU buffers</summary>
[StructLayout( LayoutKind.Auto )]
readonly partial struct Context
{
	const bool bfloat16 = false;
	const eDataType defaultDataType = bfloat16 ? eDataType.BF16 : eDataType.FP16;

	public readonly TemporaryTensors temp;
	readonly iContext context;
	readonly iDevice device;
	public readonly Parameters parameters;
	readonly bool isFastGpu;

#if DEBUG
	static readonly string pathPythonDumps = @"C:\Temp\2remove\Mistral\Python";
	readonly string? dumps;
#endif

	/// <summary>Create the structure</summary>
	public Context( in Device dev, TemporaryTensors temp, Parameters parameters, PerformanceParams perfParams )
	{
		this.temp = temp;
		context = dev.context;
		device = dev.device;
		this.parameters = parameters;
		isFastGpu = perfParams.isFastGpu;

#if DEBUG
		if( null != pathPythonDumps && Directory.Exists( pathPythonDumps ) )
			dumps = pathPythonDumps;
		else
			dumps = null;
#endif
	}

#if DEBUG
	public void dbgCompareTensor( iTensor tensor, string zip )
	{
		if( null == dumps )
			return;
		var data = context.downloadTensor( tensor );
		string path = Path.Combine( dumps, Path.ChangeExtension( zip, ".zip" ) );
		var test = Torch.TensorLoader.load( path );
		TensorsDiff diff = data.diff( test );
		Logger.Debug( @"""{0}"": {1}", zip, diff );
	}

	public void dbgCompareTensor( Tensor tensor, string zip ) =>
		dbgCompareTensor( tensor.native, zip );
#endif

	Tensor fp16( ref Tensor? cache, in Int128 size )
	{
		cache ??= new Tensor( device );
		const eDataType dt = bfloat16 ? eDataType.BF16 : eDataType.FP16;
		cache.createDense( size, dt );
		return cache;
	}

	Tensor fp16( ref Tensor? cache, int x, int y = 1, int z = 1, int w = 1 ) =>
		fp16( ref cache, new Int128( x, y, z, w ) );

	/// <summary>An approximate equivalent of <c>ggml_get_rows</c></summary>
	public Tensor getRows( iTensor source, iTensor rows, int start, int end, ref Tensor? cached )
	{
		sTensorDesc emb = source.getDesc();
		sTensorDesc r = rows.getDesc();

		int outputWidth = end - start;

		var cb = new ConstantBuffers.getRows
		{
			firstColumn = (uint)start,
			inputStride = (uint)r.shape.size.x,
			sourceHeight = (uint)emb.shape.size.y,
			rowLength = (uint)emb.shape.size.x,
			outputStride = (uint)outputWidth,
		};

		Tensor res;
		res = fp16( ref cached, emb.shape.size.x, outputWidth, r.shape.size.y );
		context.getRows( cb, res.native, source, rows );
		context.dispatch( outputWidth, r.shape.size.y );

		return res;
	}

	/// <summary>Dispatch 1 thread group for every row of the tensor</summary>
	void dispatchRows( in Int128 size ) =>
		context.dispatch( size.y, size.z, size.w );

	void checkRmsNormArgs( Tensor tensor, iTensor weights )
	{
		if( tensor.size.w != 1 )
			throw new NotImplementedException();
#if DEBUG
		Int128 weightSize = weights.getSize();
		if( weightSize.x != tensor.size.x )
			throw new ArgumentException();
		if( weightSize.y != 1 || weightSize.z != 1 || weightSize.w != 1 )
			throw new ArgumentException();
#endif
	}

	/// <summary>Compute RMSNorm in place</summary>
	public void rmsNorm( Tensor tensor, iTensor weights )
	{
		checkRmsNormArgs( tensor, weights );

		var cb = new ConstantBuffers.rmsNorm
		{
			inputSize = tensor.size,
			inputStrides = tensor.stride,
			epsilon = parameters.normalEpsilon,
		};
		context.rmsNorm( cb, tensor.native, weights );
		dispatchRows( tensor.size );
	}

	/// <summary>Compute RMSnorm, writing into a temporary tensor</summary>
	public Tensor rmsNorm( Tensor tensor, iTensor weights, ref Tensor? cached )
	{
		checkRmsNormArgs( tensor, weights );
		Tensor res = fp16( ref cached, tensor.size.x, tensor.size.y, tensor.size.z );

		var cb = new ConstantBuffers.rmsNorm2
		{
			inputSize = tensor.size,
			inputStrides = tensor.stride,
			epsilon = parameters.normalEpsilon,
		};
		context.rmsNorm2( cb, res.native, tensor.native, weights );
		dispatchRows( tensor.size );
		return res;
	}

	void columnProductDense( Tensor res, Tensor a, iTensor b )
	{
		var cb = new ConstantBuffers.rowMatProduct
		{
			rowLength = (uint)a.size.x,
			rowsCount = (uint)res.size.x,
			arg0Strides = new uint2( a.stride.y, a.stride.z ),
			resultStrides = new uint2( res.stride.y, res.stride.z ),
		};
		context.rowMatProduct( cb, res.native, a.native, b );

		// TODO: adjust code generator to extract constants like these from HLSL
		const int THREADS = 512;

		int groupsX = ( res.size.x + THREADS - 1 ) / THREADS;
		context.dispatch( groupsX, a.size.y, a.size.z );
	}

	void rowMatProductFixed( Tensor res, Tensor a, iTensor b )
	{
		var cb = new ConstantBuffers.rowMatProductFixed
		{
			arg0Strides = new uint2( a.stride.y, a.stride.z ),
			resultStrides = new uint2( res.stride.y, res.stride.z ),
			rowsCount = (uint)res.size.x,
			groupOffset = 0,
			arg0SizeYZ = new uint2( a.size.y, a.size.z )
		};
		context.rowMatProductFixed( cb, res.native, a.native, b );

		const int THREADS = 64;
		int groupsX = ( res.size.x + THREADS - 1 ) / THREADS;

		if( isFastGpu || ( a.size.y == 1 && a.size.z == 1 ) )
		{
			context.dispatch( groupsX, 1, 1 );
		}
		else
		{
			// Split computations into multiple dispatches
			// The magic number is arbitrary, tested on AMD iGPU inside Ryzen 7 5700G
			const int maxGroupsPerDispatch = 16;

			foreach( (int off, int len) in MistralUtils.batchSplit( groupsX, maxGroupsPerDispatch ) )
			{
				if( 0 != off )
				{
					cb.groupOffset = (uint)off;
					context.bindShader( (ushort)eShader.rowMatProductFixed, ref cb );
				}
				context.dispatch( len, 1, 1 );
			}
		}
	}

	void columnProductImpl( Tensor res, Tensor a, iTensor b, in sTensorDesc bDesc )
	{
		if( bDesc.layout == eTensorLayout.Dense )
		{
			if( a.size.x == 4096 )
				rowMatProductFixed( res, a, b );
			else
				columnProductDense( res, a, b );
		}
		else
			columnProductCompressed( res, a, b, bDesc );
	}

	/// <summary>Multiply rows of the tensor by a matrix, i.e. [ x, y, z ] * [ x, r ] = [ r, y, z ]</summary>
	public Tensor columnProduct( Tensor a, iTensor b, ref Tensor? cached )
	{
		var bDesc = b.getDesc();
		if( bDesc.size.x != a.size.x )
			throw new ArgumentException();

		Tensor res = fp16( ref cached, bDesc.size.y, a.size.y, a.size.z );
		columnProductImpl( res, a, b, bDesc );
		return res;
	}

	public Tensor columnProductStaging( Tensor a, iTensor b, ref Tensor? cached )
	{
		var bDesc = b.getDesc();
		if( bDesc.size.x != a.size.x )
			throw new ArgumentException();

		cached ??= new Tensor( device );
		Int128 size = new Int128( bDesc.size.y, a.size.y, a.size.z, a.size.w );
		sTensorDesc desc = new sTensorDesc
		{
			shape = new TensorShape( size ),
			dataType = eDataType.FP16,
			usage = eBufferUse.ReadWriteDownload,
			layout = eTensorLayout.Dense
		};
		cached.create( desc );
		columnProductImpl( cached, a, b, bDesc );
		return cached;
	}

	public void rotaryEmbedding( Tensor xq, Tensor xk, int columnOffset, float minusHalfDimMul, float theta = 10000.0f )
	{
		var cb = new ConstantBuffers.rotaryEmbedding
		{
			size = xq.size,
			stride = xq.stride,
			theta = theta,
			minusHalfDimMul = minusHalfDimMul,
			freqsOffset = columnOffset,
		};
		context.rotaryEmbedding( cb, xq.native );
		dispatchRows( xq.size );

		cb.size = xk.size;
		cb.stride = xk.stride;
		context.rotaryEmbedding( cb, xk.native );
		dispatchRows( xk.size );
	}

	/// <summary>Create a dense row-major tensor filled with zeros</summary>
	public Tensor denseZeros( in Int128 size, ref Tensor? cache, eDataType dataType = defaultDataType )
	{
		sTensorDesc desc = new sTensorDesc
		{
			shape = new TensorShape( size ),
			dataType = dataType,
			usage = eBufferUse.ReadWrite,
			layout = eTensorLayout.Dense
		};

		cache ??= new Tensor( device );
		cache.create( desc );

		context.writeDenseZeros( cache.native, desc.shape.size );
		return cache;
	}

	/// <summary>Update per-layer attention caches, writing new data there</summary>
	public void updateAttnCache( Tensor cache, Tensor t, int offset )
	{
		if( cache.size.xy != t.size.xy )
			throw new ArgumentException();
		if( cache.stride.xy != t.stride.xy )
			throw new ArgumentException();

		int slidingWindow = parameters.slidingWindow;
		int loadPosition;

		int threadGroups;
		if( t.size.z <= slidingWindow )
		{
			// Size of the input is less than the size of the cache
			// Dispatch count of threads equal to the size of the input
			threadGroups = t.size.z;
			loadPosition = 0;
		}
		else
		{
			// Input is larger than the cache
			// Overwrite complete cache with the end slice of the input
			threadGroups = slidingWindow;
			loadPosition = t.size.z - slidingWindow;
			offset += t.size.z - slidingWindow;
		}

		var cb = new ConstantBuffers.attentionCacheUpdate
		{
			inputStride = t.stride.zw,
			cacheStride = cache.stride.zw,
			firstSliceLoad = (uint)loadPosition,
			firstSliceStore = (uint)offset,
			slidingWindow = (uint)slidingWindow,
			rowLength = (uint)( cache.size.x * cache.size.y )
		};
		context.attentionCacheUpdate( cb, cache.native, t.native );
		context.dispatch( threadGroups, t.size.w );
	}

	public Tensor mulMat( iTensor a, iTensor b,
		in TensorShape aShape, in TensorShape bShape,
		ref Tensor? cached, float mul = 1.0f )
	{
		if( aShape.size.x != bShape.size.x )
			throw new ArgumentException();
		if( aShape.size.zw != bShape.size.zw )
			throw new ArgumentException();

		Tensor res = fp16( ref cached, aShape.size.y, bShape.size.y, aShape.size.z, aShape.size.w );
		var cb = new ConstantBuffers.mulMatTiled
		{
			arg0Size = aShape.size,
			arg0Strides = aShape.stride,
			arg1Strides = bShape.stride,
			resultSize = res.size,
			resultStrides = res.stride,
			finalMul = mul
		};

		context.mulMatTiled( cb, res.native, a, b );

		const int TILE_SIZE = 32;
		int x = ( res.size.x + TILE_SIZE - 1 ) / TILE_SIZE;
		int y = ( res.size.y + TILE_SIZE - 1 ) / TILE_SIZE;
		int z = res.size.z * res.size.w;
		context.dispatch( x, y, z );

		return res;
	}

	public Tensor mulMatRepeatZ( iTensor a, iTensor b,
		in TensorShape aShape, in TensorShape bShape, int arg0RepeatZ,
		ref Tensor? cached, float mul = 1.0f )
	{
		if( aShape.size.x != bShape.size.x )
			throw new ArgumentException();
		if( aShape.size.z * arg0RepeatZ != bShape.size.z )
			throw new ArgumentException();
		if( aShape.size.w != bShape.size.w )
			throw new ArgumentException();

		Tensor res = fp16( ref cached, aShape.size.y, bShape.size.y, aShape.size.z * arg0RepeatZ, aShape.size.w );
		var cb = new ConstantBuffers.mulMatTiledRepeatZ
		{
			arg0Size = aShape.size,
			arg0Strides = aShape.stride,
			arg1Strides = bShape.stride,
			resultSize = res.size,
			resultStrides = res.stride,
			finalMul = mul,
			arg0RepeatZ = (uint)arg0RepeatZ,
		};

		context.mulMatTiledRepeatZ( cb, res.native, a, b );

		const int TILE_SIZE = 32;
		int x = ( res.size.x + TILE_SIZE - 1 ) / TILE_SIZE;
		int y = ( res.size.y + TILE_SIZE - 1 ) / TILE_SIZE;
		int z = res.size.z * res.size.w;
		context.dispatch( x, y, z );

		return res;
	}

	static uint bf16( float val )
	{
		uint u = BitConverter.SingleToUInt32Bits( val );
		// This works fine even for INF input
		// INF is encoded as 0x7F800000 (positive) / 0xFF800000 (negative), mantissa bits are all zeros
		uint bias = ( u >> 16 ) & 1u;
		bias += 0x7FFF;
		u += bias;
		return u >> 16;
	}

	/// <summary>Write specified value (the default is negative infinity) into upper-right diagonal portion of the tensor</summary>
	public void applyMask( Tensor t, in ModelMask mask, float val = float.NegativeInfinity )
	{
		if( mask.size > t.size.x || mask.size != t.size.y )
			throw new ArgumentException( "Unexpected mask size" );
		if( t.stride.x != 1 )
			throw new ArgumentException();

		int maskXOffset = t.size.x - mask.size;
		var cb = new ConstantBuffers.applyMask
		{
			width = (uint)mask.size,
			strides = t.stride.yzw,
			xOffset = (uint)maskXOffset,
			diagonal = mask.diagonal,
		};
		if( bfloat16 )
			cb.maskValue = bf16( val );
		else
			cb.maskValue = BitConverter.SingleToUInt32Bits( val );

		context.applyMask( cb, t.native );
		dispatchRows( t.size );
	}

	/// <summary>torch.nn.Softmax, on the X dimension of the tensor</summary>
	/// <seealso href="https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html" />
	public void softMax( Tensor t )
	{
		if( t.stride.x != 1 )
			throw new ArgumentException();

		var cb = new ConstantBuffers.softMax
		{
			width = (uint)t.size.x,
			strides = t.stride.yzw,
		};
		context.softMax( cb, t.native );
		dispatchRows( t.size );
	}

	/// <summary>Compute the logarithm of the softmax function.</summary>
	/// <remarks>In principle, log_softmax(x) = log(softmax(x)) but using a more accurate implementation.</remarks>
	public void logSoftMax( Tensor t )
	{
		if( t.stride.x != 1 )
			throw new ArgumentException();

		var cb = new ConstantBuffers.logSoftMax
		{
			width = (uint)t.size.x,
			strides = t.stride.yzw,
		};
		context.logSoftMax( cb, t.native );
		dispatchRows( t.size );
	}

	public Tensor copyTranspose( Tensor t, in TensorShape shape, ref Tensor? cache )
	{
		cache ??= new Tensor( device );
		cache.createDense( shape.size, t.dataType );

		var cb = new ConstantBuffers.copyTranspose
		{
			inputStrides = shape.stride,
			width = (uint)cache.size.x,
			outputStrides = cache.stride.yzw,
		};
		context.copyTranspose( cb, cache.native, t.native );

		dispatchRows( shape.size );
		return cache;
	}

	/// <summary>Compute <c>a += b</c>, element-wise</summary>
	public void addInPlace( Tensor a, Tensor b )
	{
		if( a.shape != b.shape || a.shape.stride.x != 1 )
			throw new ArgumentException();

		var cb = new ConstantBuffers.addInPlace
		{
			width = (uint)a.shape.size.x,
			strides = a.shape.stride.yzw,
		};
		context.addInPlace( cb, a.native, b.native );
		dispatchRows( a.size );
	}

	/// <summary>a = silu(a) * b, in-place</summary>
	/// <seealso href="https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html" />
	public void SiLU( Tensor a, Tensor b )
	{
		if( a.shape != b.shape || a.shape.stride.x != 1 )
			throw new ArgumentException();

		var cb = new ConstantBuffers.silu
		{
			width = (uint)a.shape.size.x,
			strides = a.shape.stride.yzw,
		};
		context.silu( cb, a.native, b.native );
		dispatchRows( a.size );
	}

	void softMaxFinal( Tensor logits, float temperature )
	{
		if( logits.stride.x != 1 )
			throw new ArgumentException();

		var cb = new ConstantBuffers.softMaxFinal
		{
			width = (uint)logits.size.x,
			strides = logits.stride.yzw,
			initialMul = 1.0f / temperature,
		};
		context.softMaxFinal( cb, logits.native );
		dispatchRows( logits.size );
	}

	Tensor makeUintTensor( ref Tensor? cached, int x, int y = 1, eBufferUse usage = eBufferUse.ReadWrite )
	{
		cached ??= new Tensor( device ); ;
		Tensor res = cached;
		sTensorDesc desc = new sTensorDesc
		{
			shape = TensorShape.rowMajor( x, y ),
			dataType = eDataType.U32,
			usage = usage,
			layout = eTensorLayout.Dense,
		};
		res.create( desc );
		return res;
	}

	public Tensor sampleTopP( Tensor logits, SamplingParams samplingParams, Random rand )
	{
		softMaxFinal( logits, samplingParams.temperature );

		Int128 size = logits.size;
		Int128 stride = logits.stride;
		if( size.zw != new uint2( 1, 1 ) || stride.x != 1 )
			throw new ArgumentException();
		if( size.x > ushort.MaxValue )
		{
			// That compute shader packs FP16 probabilities and uint16_t indices into a single uint32_t values
			throw new ArgumentOutOfRangeException();
		}

		Tensor temp = makeUintTensor( ref this.temp.topPCounters, 0x8000, size.y );
		Tensor res = makeUintTensor( ref this.temp.topP, size.y, 1, eBufferUse.ReadWriteDownload );

		ConstantBuffers.sampleTopP cb = new ConstantBuffers.sampleTopP
		{
			width = (uint)size.x,
			tensorStride = (uint)stride.y,
			topP = samplingParams.topP,
			rand = rand.NextSingle(),
		};
		context.sampleTopP( cb, temp.native, res.native, logits.native );
		context.dispatch( size.y );
		return res;
	}

	/// <summary><c>torch.multinomial</c></summary>
	public Tensor sampleAll( Tensor logits, Random rand )
	{
		Int128 size = logits.size;
		Int128 stride = logits.stride;
		if( size.yzw != new uint3( 1, 1, 1 ) || stride.x != 1 )
			throw new ArgumentException();

		Tensor res = makeUintTensor( ref temp.topP, 1, 1, eBufferUse.ReadWriteDownload );
		double r = rand.NextDouble();
		ConstantBuffers.sampleAll cb = new ConstantBuffers.sampleAll
		{
			width = (uint)size.x,
			rand = (float)r,
			rand64 = (uint2)r,
		};
		context.sampleAll( cb, res.native, logits.native );
		context.dispatch( 1 );
		return res;
	}

	/// <summary>next_token = torch.argmax(logprobs[:, -1,:], dim=-1)</summary>
	public Tensor sampleMax( Tensor logits )
	{
		Int128 size = logits.size;
		Int128 stride = logits.stride;
		if( size.zw != new uint2( 1, 1 ) || stride.x != 1 )
			throw new ArgumentException();

		Tensor res = makeUintTensor( ref temp.topP, size.y, 1, eBufferUse.ReadWriteDownload );
		ConstantBuffers.sampleMax cb = new ConstantBuffers.sampleMax
		{
			width = (uint)size.x,
			tensorStride = (uint)stride.y,
		};
		context.sampleMax( cb, res.native, logits.native );
		context.dispatch( size.y );
		return res;
	}

	public void replaceColumn( iTensor tokens, int x, iTensor mask, Tensor nextTokens )
	{
		sTensorDesc tokDesc = tokens.getDesc();
		sTensorDesc maskDesc = mask.getDesc();
		if( tokDesc.dataType != eDataType.U32 || maskDesc.dataType != eDataType.U32 || nextTokens.dataType != eDataType.U32 )
			throw new ArgumentException();

		if( nextTokens.size.yzw != new uint3( 1, 1, 1 ) )
			throw new ArgumentException();
		if( x < 0 || x >= tokDesc.shape.size.x )
			throw new ArgumentOutOfRangeException();

		int height = nextTokens.size.x;
		uint3 expectedSize = new uint3( height, 1, 1 );
		if( tokDesc.shape.size.yzw != expectedSize || maskDesc.shape.size.yzw != expectedSize )
			throw new ArgumentException();

		var cb = new ConstantBuffers.replaceResultColumn
		{
			height = (uint)height,
			curPos = (uint)x,
			resultStride = (uint)tokDesc.shape.stride.y,
			maskStride = (uint)maskDesc.shape.stride.y,
		};
		context.replaceResultColumn( cb, tokens, nextTokens.native, mask );
		context.dispatch( 1 );
	}

	public ProfilerBlock profilerBlock( eProfilerBlock id ) =>
		context.profilerBlock( (ushort)id );

	/// <summary>When <c>size.y > 1</c>, move the last row to the first position,
	/// trim the tensor to `size.y = 1`, and make it dense again</summary>
	/// <remarks>When <c>size.y == 1</c>, the method does nothing, returns the input tensor</remarks>
	public Tensor trimToLastRow( Tensor t, ref Tensor? cache )
	{
		if( t.stride.x != 1 )
			throw new ArgumentException();
		if( t.size.y == 1 )
			return t;   // Already a single row, nothing to do

		// Dispatch a shader to move size.y-1 row to zero index
		var cb = new ConstantBuffers.copyLastRow
		{
			inputStrides = t.stride,
			width = (uint)t.size.x,
			lastRowIndex = (uint)( t.size.y - 1 )
		};
		context.copyLastRow( cb, t.native );
		context.dispatch( 1, t.size.z, t.size.w );

		var shape = t.shape.trim( 1, 1 );
		return copyTranspose( t, shape, ref cache );
	}

	public void dbgSaveTensor( iTensor tensor, string path )
	{
		var data = context.downloadTensor( tensor );
		data.save( path );
	}

	public Tensor unrotate( Tensor t, in RotatedTensorShape shape, ref Tensor? cache )
	{
		Tensor res = fp16( ref cache, shape.size );
		var cb = shape.unrotateConstants( res.shape );
		context.unrotate( cb, res.native, t.native );
		dispatchRows( res.size );
		return res;
	}

	public void unbindInputs() => context.unbindInputs();
}