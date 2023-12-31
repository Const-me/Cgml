namespace Mistral.Model;
using Cgml;
using System;
using System.Diagnostics;
using System.IO.Compression;
using System.Runtime.InteropServices;

sealed partial class Model: iModel
{
	readonly Transformer transformer;
	readonly Tokenizer tokenizer;
	readonly Device dev;

	public void Dispose()
	{
		transformer?.Dispose();
		tokenizer?.Dispose();
		dev.Dispose();
	}

	/// <summary>Load CGML package</summary>
	public Model( in Device dev, string path, Action<double>? pfnProgress )
	{
		this.dev = dev;

		using var zipFile = File.OpenRead( path );
		using var zip = new ZipArchive( zipFile, ZipArchiveMode.Read );

		static Tokenizer readTokenizer( in Device dev, ZipArchive zip )
		{
			ZipArchiveEntry e = zip.GetEntry( vocabEntry ) ??
				throw new ApplicationException( $"{vocabEntry} is missing" );

			using var stream = e.Open();
			return new Tokenizer( dev, stream, (int)e.Length );
		}
		tokenizer = readTokenizer( dev, zip );
		transformer = (Transformer)Transformer.serializer().read( zip, dev.device, pfnProgress );
		transformer.afterLoadFix();
		samplingParams = SamplingParams.makeDefault();
		cacheMetadata = new RotatingCacheMetadata( transformer.parameters.slidingWindow );

		printMemopyUse( false );
	}

	/// <summary>Construct from Python data</summary>
	public Model( in Device dev, ParamsJson json, Tokenizer tokenizer, Dictionary<string, iTensor> tensors )
	{
		this.dev = dev;
		this.tokenizer = tokenizer;
		transformer = new Transformer( json, tensors );
		samplingParams = SamplingParams.makeDefault();
		cacheMetadata = new RotatingCacheMetadata( transformer.parameters.slidingWindow );

		printMemopyUse( false );
	}

	public SamplingParams? samplingParams { get; set; } = null;
	Random? random = null;

	iTensor? input;
	iTensor createInputTensor( in TensorShape inputShape )
	{
		sTensorDesc desc = new sTensorDesc()
		{
			shape = inputShape,
			dataType = eDataType.U32,
			usage = eBufferUse.ReadWriteDownload,
			layout = eTensorLayout.Dense
		};
		input = dev.device.createTensor( ref desc, input );
		return input;
	}

	iTensor createInputTensor( in TensorShape inputShape, List<int[]> promptTokens )
	{
		Debug.Assert( promptTokens.Count == inputShape.size.y );
		iTensor tensor = createInputTensor( inputShape );

		int totalLen = inputShape.size.x;
		pfnWriteTensor<int> pfn = delegate ( Span<int> data )
		{
			for( int i = 0; i < promptTokens.Count; i++ )
			{
				Span<int> dest = data.Slice( 0, totalLen );
				ReadOnlySpan<int> source = promptTokens[ i ];

				int lengthToCopy = Math.Min( source.Length, totalLen );
				source = source.Slice( 0, lengthToCopy );
				source.CopyTo( dest );

				if( lengthToCopy < totalLen )
					dest.Slice( source.Length ).Fill( tokenizer.idPad );

				data = data.Slice( totalLen );
			}
		};
		dev.context.writeDynamic( tensor, inputShape, pfn );

		return tensor;
	}

	RotatingCacheMetadata cacheMetadata;

	string iModel.generate( string prompt, int maxTokens )
	{
		Context ctx = transformer.context( dev, performanceParams );
		using var rootBlock = ctx.profilerBlock( eProfilerBlock.Generate );
		transformer.prepareCaches( ctx );

		List<int[]> promptTokens = new List<int[]>( 1 )
		{
			tokenizer.encode( prompt, true, false )
		};
		int minPromptSize = promptTokens.Min( p => p.Length );
		int maxPromptSize = promptTokens.Max( p => p.Length );
		int totalLen = maxTokens + maxPromptSize;

		TensorShape inputShape = TensorShape.rowMajorMatrix( totalLen, promptTokens.Count );
		input = createInputTensor( inputShape, promptTokens );

		Tensor logprobs;
		// Pre-fill
		using( var block = ctx.profilerBlock( eProfilerBlock.PreFill ) )
		{
			cacheMetadata.begin( minPromptSize );
			logprobs = transformer.preFill( ctx, input, cacheMetadata, minPromptSize );
			cacheMetadata.end( minPromptSize );

			// ctx.dbgCompareTensor( logprobs, "15-logits" );
			ctx.logSoftMax( logprobs );
			// ctx.dbgCompareTensor( logprobs, "16-lsm" );
			logprobs = ctx.trimToLastRow( logprobs, ref ctx.temp.logProbsTrimmed );
		}

		List<int> generated = new List<int>( maxTokens );
		using( var block = ctx.profilerBlock( eProfilerBlock.MakeToken ) )
		{
			pfnReadTensor<int> pfnRead = delegate ( ReadOnlySpan<int> source )
			{
				Debug.Assert( source.Length == 1 );
				generated.Add( source[ 0 ] );
			};
			iTensor? prevToken = null;

			for( int i = 0; i < maxTokens; i++ )
			{
				if( null != prevToken )
				{
					// Fetch previous token from the staging buffer
					dev.context.download( prevToken, pfnRead, eDownloadFlag.ReadStaging );
				}

				Tensor token;
				SamplingParams? samplingParams = this.samplingParams;
				if( null == samplingParams )
					token = ctx.sampleMax( logprobs );
				else
				{
					random ??= new Random();
					token = ctx.sampleTopP( logprobs, samplingParams, random );
				}

				// Submit CopyResource command, but don't yet access the data in the staging tensor
				// By the next iteration of this loop it will definitely arrive, computeNext() method below dispatch hundreds of compute shaders
				dev.context.copyToStaging( token.native );
				prevToken = token.native;

				cacheMetadata.begin();
				logprobs = transformer.computeNext( ctx, token, cacheMetadata );
				cacheMetadata.end();

				ctx.logSoftMax( logprobs );
			}

			// Fetch final token from the staging buffer
			if( null != prevToken )
				dev.context.download( prevToken, pfnRead, eDownloadFlag.ReadStaging );
		}

		return tokenizer.decode( CollectionsMarshal.AsSpan( generated ) );
	}

	bool firstGenerate = true;

	iTokenizer iModel.tokenizer => tokenizer;

	iTensor createInputTensor( IReadOnlyList<int> tokens )
	{
		TensorShape inputShape = TensorShape.rowMajorMatrix( tokens.Count, 1 );
		iTensor tensor = createInputTensor( inputShape );

		pfnWriteTensor<int> pfn = delegate ( Span<int> data )
		{
			if( tokens is List<int> list )
			{
				ReadOnlySpan<int> source = CollectionsMarshal.AsSpan( list );
				source.CopyTo( data );
			}
			else
			{
				for( int i = 0; i < tokens.Count; i++ )
					data[ i ] = tokens[ i ];
			}
		};
		dev.context.writeDynamic( tensor, inputShape, pfn );
		return tensor;
	}

	string iModel.generate( IReadOnlyList<int> tokens, ChatClient client )
	{
		Context ctx = transformer.context( dev, performanceParams );
		using var rootBlock = ctx.profilerBlock( eProfilerBlock.Generate );
		if( firstGenerate )
		{
			firstGenerate = false;
			transformer.prepareCaches( ctx );
			cacheMetadata = new RotatingCacheMetadata( ctx.parameters.slidingWindow );
		}

		int promptSize = tokens.Count;
		int maxTokens = client.maxResponseTokens();
		input = createInputTensor( tokens );

		Tensor logprobs;
		// Pre-fill
		using( var block = ctx.profilerBlock( eProfilerBlock.PreFill ) )
		{
			cacheMetadata.begin( promptSize );
			logprobs = transformer.preFill( ctx, input, cacheMetadata, promptSize );
			cacheMetadata.end( promptSize );

			ctx.logSoftMax( logprobs );
			logprobs = ctx.trimToLastRow( logprobs, ref ctx.temp.logProbsTrimmed );
		}

		List<int> generated = new List<int>( maxTokens );
		using( var block = ctx.profilerBlock( eProfilerBlock.MakeToken ) )
		{
			pfnReadTensor<int> pfnRead = delegate ( ReadOnlySpan<int> source )
			{
				Debug.Assert( source.Length == 1 );
				generated.Add( source[ 0 ] );
			};

			string result = "";
			for( int i = 0; i < maxTokens; i++ )
			{
				Tensor token;
				SamplingParams? samplingParams = this.samplingParams;
				if( null == samplingParams )
					token = ctx.sampleMax( logprobs );
				else
				{
					random ??= new Random();
					token = ctx.sampleTopP( logprobs, samplingParams, random );
				}

				dev.context.download( token.native, pfnRead );

				int generatedCount = generated.Count;
				bool eos = generated[ generated.Count - 1 ] == tokenizer.idEOS;
				if( eos )
					generated.RemoveAt( generated.Count - 1 );

				result = tokenizer.decode( CollectionsMarshal.AsSpan( generated ) );
				if( eos )
				{
					client.complete( generatedCount );
					return result;
				}

				string? res = client.tryMakeResponse( result );
				if( null != res )
				{
					client.complete( generatedCount );
					return res;
				}

				cacheMetadata.begin();
				logprobs = transformer.computeNext( ctx, token, cacheMetadata );
				cacheMetadata.end();

				ctx.logSoftMax( logprobs );
			}
			return result;
		}
	}
}