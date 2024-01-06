namespace Mistral.Model;
using Cgml;
using Cgml.Serialize;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Serialization;

[DataContract]
sealed class Transformer: IDisposable
{
	[DataMember]
	internal readonly TransformerBlock[] layers;
	[DataMember]
	readonly iTensor tok_embeddings, norm, output;
	[DataMember]
	internal readonly Parameters parameters;
	[IgnoreDataMember]
	TemporaryTensors temp = new TemporaryTensors();
	[IgnoreDataMember]
	public eModelVersion modelVersion => parameters.modelVersion;

	/// <summary>Construct from the original Python model</summary>
	public Transformer( ParamsJson p, Dictionary<string, iTensor> tensors )
	{
		layers = new TransformerBlock[ p.n_layers ];
		parameters = new Parameters( p );

		for( int i = 0; i < layers.Length; i++ )
			layers[ i ] = new TransformerBlock( i, tensors );

		tok_embeddings = tensors[ "tok_embeddings.weight" ];
		norm = tensors[ "norm.weight" ];
		output = tensors[ "output.weight" ];

		tensors.Clear();
	}

	public void afterLoadFix() =>
		parameters.afterLoadFix();

	/// <summary>Initialize the context structure, which implements these ML algorithms and caches GPU buffers</summary>
	public Context context( in Device dev, PerformanceParams perfParams )
	{
		// That line is needed because DataContractSerializer doesn't call default constructors when de-serializing
		temp ??= new TemporaryTensors();
		return new Context( dev, temp, parameters, perfParams );
	}

	public Tensor preFill( ref Context ctx, iTensor tokens, iRotatingCacheMetadata cacheMetadata, int length )
	{
#if DEBUG
		ctx.prefix = "pre";
#endif
		// ctx.dbgCompareTensor( tokens, "01-tokens" );
		const int colStart = 0;
		int colEnd = length;
		Tensor t = ctx.getRows( tok_embeddings, tokens, colStart, colEnd, ref temp.inpL );
		ctx.dbgCompareTensor( t, "01-embeddings" );
		ModelMask mask = new ModelMask( colStart, colEnd );

		for( int i = 0; i < layers.Length; i++ )
		{
			TransformerBlock layer = layers[ i ];
#if DEBUG
			ctx.prefix = $"pre-b{i}";
#endif
			using var block = ctx.profilerBlock( eProfilerBlock.Layer );
			t = layer.forward( ctx, t, cacheMetadata, mask );
		}
#if DEBUG
		ctx.prefix = "pre";
#endif
		ctx.rmsNorm( t, norm );
		t = ctx.columnProduct( t, output, ref temp.result );
		return t;
	}

	public Tensor computeNext( ref Context ctx, Tensor tokens, iRotatingCacheMetadata cacheMetadata, int idx )
	{
#if DEBUG
		ctx.prefix = $"{idx}";
#endif
		Tensor t = ctx.getRows( tok_embeddings, tokens.native, 0, 1, ref temp.inpL );

		for( int i = 0; i < layers.Length; i++ )
		{
			TransformerBlock layer = layers[ i ];
#if DEBUG
			ctx.prefix = $"{idx}-b{i}";
#endif
			using var block = ctx.profilerBlock( eProfilerBlock.Layer );
			t = layer.forward( ctx, t, cacheMetadata, null );
		}
#if DEBUG
		ctx.prefix = $"{idx}";
#endif
		ctx.rmsNorm( t, norm );
		t = ctx.columnProduct( t, output, ref temp.result );
		return t;
	}

	public void prepareCaches( in Context ctx )
	{
		foreach( var layer in layers )
			layer.attention.prepareCacheTensors( ctx );
	}

	public long totalWeights()
	{
		long res = 0;
		res += tok_embeddings.countElements();
		res += norm.countElements();
		res += output.countElements();

		foreach( TransformerBlock layer in layers )
			res += layer.totalWeights();

		return res;
	}

	public eTensorLayout compression =>
		layers[ 0 ].attention.wk.getDesc().layout;

	public void Dispose()
	{
		temp?.Dispose();
		foreach( var layer in layers )
			layer?.Dispose();

		tok_embeddings?.Dispose();
		norm?.Dispose();
		output?.Dispose();
	}

	public Vector128<long> getMemoryUse()
	{
		Vector128<long> v = tok_embeddings.getMemoryUse();
		v = Sse2.Add( v, norm.getMemoryUse() );
		v = Sse2.Add( v, output.getMemoryUse() );
		foreach( var layer in layers )
			v = Sse2.Add( v, layer.getMemoryUse() );
		return v;
	}

	public void getVideoMemoryUsage( ref long kv, ref long temp )
	{
		if( null != this.temp )
			temp += this.temp.getVideoMemoryUsage();
		foreach( var layer in layers )
			kv += layer.attention.kvVideoMemoryUsage();
	}

	public void getMemoryUse( Dictionary<string, Vector128<long>> dict )
	{
		dict.addTensorMemory( tok_embeddings );
		dict.addTensorMemory( norm );
		dict.addTensorMemory( output );
		foreach( var layer in layers )
			layer.getMemoryUse( dict );
	}

	public readonly int maxBatchSize = 1;

	public static Serializer serializer()
	{
		string[] dictEntries = new string[]
		{
			"http://schemas.datacontract.org/2004/07/Mistral.Model",
			nameof(Transformer),
			nameof(TransformerBlock),

			nameof(layers),
			nameof(tok_embeddings),
			nameof(norm),
			nameof(output),
			nameof(parameters),

			nameof(TransformerBlock.attention),
			nameof(TransformerBlock.feedForward),
			nameof(TransformerBlock.attention_norm),
			nameof(TransformerBlock.ffn_norm),

			nameof(Attention.wk),
			nameof(Attention.wo),
			nameof(Attention.wq),
			nameof(Attention.wv),

			nameof(FeedForward.w1),
			nameof(FeedForward.w2),
			nameof(FeedForward.w3),

			nameof(Parameters.countHeads),
			nameof(Parameters.repeats),
			nameof(Parameters.minusHalfDimMul),
			nameof(Parameters.attnScoresMul),
			nameof(Parameters.normalEpsilon),
			nameof(Parameters.attnCacheSize),
		};

		return new Serializer( typeof( Transformer ), dictEntries );
	}
}