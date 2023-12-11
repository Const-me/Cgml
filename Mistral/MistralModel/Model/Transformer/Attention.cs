// When SKIP_REPEATS is defined, use special edition of matrix multiplication shader which repeats first argument along Z on the fly.
namespace Mistral.Model;
using Cgml;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Serialization;

[DataContract]
sealed class Attention: IDisposable
{
	[DataMember]
	internal readonly iTensor wk, wo, wq, wv;

	[IgnoreDataMember]
	Tensor? cacheK, cacheV;

	/// <summary>This constructor is only called during import of Python formats</summary>
	public Attention( int n, Dictionary<string, iTensor> tensors )
	{
		wk = tensors[ $"layers.{n}.attention.wk.weight" ];
		wo = tensors[ $"layers.{n}.attention.wo.weight" ];
		wq = tensors[ $"layers.{n}.attention.wq.weight" ];
		wv = tensors[ $"layers.{n}.attention.wv.weight" ];
	}

	public Tensor forward( in Context ctx, Tensor x, iRotatingCacheMetadata cacheMetadata, in ModelMask? mask )
	{
		// xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
		Tensor xq = ctx.columnProduct( x, wq, ref ctx.temp.xq );
		Tensor xk = ctx.columnProduct( x, wk, ref ctx.temp.xk );
		Tensor xv = ctx.columnProduct( x, wv, ref ctx.temp.xv );

		// ctx.dbgCompareTensor( xq, "03-xq" ); ctx.dbgCompareTensor( xk, "03-xk" ); ctx.dbgCompareTensor( xv, "03-xv" );

		// xq = xq.view(bsz, seqlen, self.n_heads, self.args.head_dim)
		xq.view( ctx.parameters.headDim, ctx.parameters.countHeads, xq.size.y, xq.size.z );
		// xk = xk.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
		xk.view( ctx.parameters.headDim, ctx.parameters.countKVHeads, xk.size.y, xk.size.z );
		// xv = xv.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
		xv.view( ctx.parameters.headDim, ctx.parameters.countKVHeads, xv.size.y, xv.size.z ); ;

		// xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
		ctx.rotaryEmbedding( xq, xk, cacheMetadata.absolute, ctx.parameters.minusHalfDimMul );
		// ctx.dbgCompareTensor( xq, "05-xq" ); ctx.dbgCompareTensor( xk, "05-xk" );

		// The cache is a rotating buffer
		(Tensor cacheK, Tensor cacheV) = getCacheTensors( ctx, xk.size.w );

		// Store into the caches
		ctx.updateAttnCache( cacheK, xk, cacheMetadata.storePosition );
		ctx.updateAttnCache( cacheV, xv, cacheMetadata.storePosition );
		// ctx.dbgCompareTensor( cacheK, "07-ck" ); ctx.dbgCompareTensor( cacheV, "07-cv" );

		Tensor key, value;

		// Load from both caches, and transpose the tensors as required for the next step
		RotatedTensorShape rts = cacheMetadata.loadShape( cacheK );
		rts.permute( 0, 2, 1, 3 );
		key = ctx.unrotate( cacheK, rts, ref ctx.temp.attnKey );

		rts = cacheMetadata.loadShape( cacheV );
		rts.permute( 2, 0, 1, 3 );
		value = ctx.unrotate( cacheV, rts, ref ctx.temp.attnVal );

		Tensor scores = ctx.mulMatRepeatZ( key.native, xq.native,
			key.shape,
			xq.shape.permute( 0, 2, 1, 3 ),
			ctx.parameters.repeats,
			ref ctx.temp.scores,
			ctx.parameters.attnScoresMul );
		// ctx.dbgCompareTensor( scores, "09-scores" );

		if( mask.HasValue )
			ctx.applyMask( scores, mask.Value );
		// ctx.dbgCompareTensor( scores, "10-scores" );

		ctx.softMax( scores );
		// ctx.dbgCompareTensor( scores, "11-scores" );

		Tensor res = ctx.mulMatRepeatZ( value.native, scores.native,
			value.shape, scores.shape,
			ctx.parameters.repeats,
			ref ctx.temp.attnTemp1 );
		// ctx.dbgCompareTensor( res, "12-out" );

		res = ctx.copyTranspose( res,
			res.shape.permute( 0, 2, 1, 3 ),
			ref ctx.temp.attnTemp2 );
		Int128 size = res.size;
		size = new Int128( size.x * size.y, size.z, size.w, 1 );
		res.view( size );

		res = ctx.columnProduct( res, wo, ref ctx.temp.attnOut );
		// ctx.dbgCompareTensor( res, "13-attn-out" );

		return res;
	}

	public void prepareCacheTensors( in Context ctx )
	{
		Int128 size = ctx.parameters.attnCacheSize;
		ctx.denseZeros( size, ref cacheK );
		ctx.denseZeros( size, ref cacheV );
	}

	/// <summary>Get the two attention cache tensors</summary>
	(Tensor, Tensor) getCacheTensors( in Context ctx, int length )
	{
		if( null != cacheK && null != cacheV && cacheK.size.w == length && cacheV.size.w == length )
			return (cacheK, cacheV);
		throw new ApplicationException( "prepareCacheTensors() was not called" );
	}

	public void Dispose()
	{
		cacheV?.Dispose();
		cacheK?.Dispose();
		wk?.Dispose();
		wo?.Dispose();
		wq?.Dispose();
		wv?.Dispose();
	}

	public Vector128<long> getMemoryUse()
	{
		Vector128<long> v = wk.getMemoryUse();
		v = Sse2.Add( v, wo.getMemoryUse() );
		v = Sse2.Add( v, wq.getMemoryUse() );
		v = Sse2.Add( v, wv.getMemoryUse() );
		// Include temporaries
		v = Sse2.Add( v, cacheK.getMemoryUse() );
		v = Sse2.Add( v, cacheV.getMemoryUse() );
		return v;
	}

	public long kvVideoMemoryUsage()
	{
		long res = 0;
		res += cacheK.getMemoryUse().GetElement( 1 );
		res += cacheV.getMemoryUse().GetElement( 1 );
		return res;
	}

	public void getMemoryUse( Dictionary<string, Vector128<long>> dict )
	{
		dict.addTensorMemory( wk );
		dict.addTensorMemory( wo );
		dict.addTensorMemory( wq );
		dict.addTensorMemory( wv );
		dict.addTensorMemory( cacheK );
		dict.addTensorMemory( cacheV );
	}

	public long totalWeights()
	{
		long res = 0;
		res += wk.countElements();
		res += wo.countElements();
		res += wq.countElements();
		res += wv.countElements();
		return res;
	}

	public void backup( iContext context, iTensor k, iTensor v )
	{
		if( null != cacheK?.native )
			context.copy( k, cacheK.native );
		else
			context.writeDenseZeros( k, k.getSize() );

		if( null != cacheV?.native )
			context.copy( v, cacheV.native );
		else
			context.writeDenseZeros( v, v.getSize() );
	}

	public void restore( iContext context, iTensor k, iTensor v )
	{
		if( null != cacheK?.native )
			context.copy( cacheK.native, k );

		if( null != cacheV?.native )
			context.copy( cacheV.native, v );
	}

	public void clear( iContext context )
	{
		if( null != cacheK?.native )
			context.writeDenseZeros( cacheK.native, cacheK.size );
		if( null != cacheV?.native )
			context.writeDenseZeros( cacheV.native, cacheV.size );
	}
}