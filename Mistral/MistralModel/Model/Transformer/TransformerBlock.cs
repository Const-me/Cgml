namespace Mistral.Model;
using Cgml;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Serialization;

[DataContract]
sealed class TransformerBlock: IDisposable
{
	[DataMember]
	internal readonly Attention attention;
	[DataMember]
	internal readonly FeedForward feedForward;
	[DataMember]
	internal readonly iTensor attention_norm, ffn_norm;

	public TransformerBlock( int layerId, Dictionary<string, iTensor> tensors )
	{
		attention = new Attention( layerId, tensors );
		feedForward = new FeedForward( layerId, tensors );
		attention_norm = tensors[ $"layers.{layerId}.attention_norm.weight" ];
		ffn_norm = tensors[ $"layers.{layerId}.ffn_norm.weight" ];
	}

	public Tensor forward( in Context ctx, Tensor x, iRotatingCacheMetadata cacheMetadata, in ModelMask? mask )
	{
		// self.attention_norm(x)
		Tensor norm = ctx.rmsNorm( x, attention_norm, ref ctx.temp.norm );
		ctx.dbgCompareTensor( norm, "02-norm" );

		// self.attention.forward(r, freqs_cis, positions, mask)
		Tensor tmp = attention.forward( ctx, norm, cacheMetadata, mask );

		ctx.addInPlace( x, tmp );

		norm = ctx.rmsNorm( x, ffn_norm, ref ctx.temp.norm );
		tmp = feedForward.forward( ctx, norm );

		ctx.addInPlace( x, tmp );
		ctx.dbgCompareTensor( x, "14-out" );
		return x;
	}

	public void Dispose()
	{
		attention?.Dispose();
		feedForward?.Dispose();
		attention_norm?.Dispose();
		ffn_norm?.Dispose();
	}

	public Vector128<long> getMemoryUse()
	{
		Vector128<long> v = attention.getMemoryUse();
		v = Sse2.Add( v, feedForward.getMemoryUse() );
		v = Sse2.Add( v, attention_norm.getMemoryUse() );
		v = Sse2.Add( v, ffn_norm.getMemoryUse() );
		return v;
	}

	public void getMemoryUse( Dictionary<string, Vector128<long>> dict )
	{
		attention.getMemoryUse( dict );
		feedForward.getMemoryUse( dict );
		dict.addTensorMemory( attention_norm );
		dict.addTensorMemory( ffn_norm );
	}

	public long totalWeights()
	{
		long res = 0;
		res += attention.totalWeights();
		res += feedForward.totalWeights();

		res += attention_norm.countElements();
		res += ffn_norm.countElements();
		return res;
	}
}