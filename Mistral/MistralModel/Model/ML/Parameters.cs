namespace Mistral.Model;
using Cgml;
using System.Runtime.Serialization;

[DataContract]
sealed record class Parameters
{
	[DataMember]
	public readonly int countHeads, repeats;
	[DataMember]
	public readonly float minusHalfDimMul, attnScoresMul, normalEpsilon;
	[DataMember]
	public readonly Int128 attnCacheSize;

	[IgnoreDataMember]
	public int headDim => attnCacheSize.x;
	[IgnoreDataMember]
	public int countKVHeads => attnCacheSize.y;
	[IgnoreDataMember]
	public int slidingWindow => attnCacheSize.z;
	[IgnoreDataMember]
	const int maxBatchSize = 1;

	public Parameters( ParamsJson p )
	{
		countHeads = p.n_heads;
		int countKVHeads = p.n_kv_heads;
		int slidingWindow = p.sliding_window;
		repeats = countHeads / countKVHeads;
		int headDim = p.dim / countHeads;
		minusHalfDimMul = (float)( -2.0 / headDim );
		attnCacheSize = new Int128( headDim, countKVHeads, slidingWindow, maxBatchSize );
		normalEpsilon = p.norm_eps;
		attnScoresMul = (float)( 1.0 / Math.Sqrt( headDim ) );
	}
}