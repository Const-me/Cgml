namespace Mistral.Model;
using Cgml;
using System.Diagnostics;
using System.Runtime.Serialization;

enum eModelVersion: int
{
	Original = 0,
	Instruct02 = 1
}

[DataContract]
sealed class Parameters
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

	// New parameters added in version 1.1; obviously, the originally published model doesn't have these numbers
	[DataMember( IsRequired = false, Name = nameof( modelVersion ) )]
	readonly int modelVersionInt;
	[IgnoreDataMember]
	public eModelVersion modelVersion => (eModelVersion)modelVersionInt;

	[DataMember( IsRequired = false )]
	public float ropeTheta { get; private set; }

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

		modelVersionInt = (int)p.modelVersion;
		ropeTheta = p.ropeTheta;
	}

	public void afterLoadFix()
	{
		switch( modelVersion )
		{
			case eModelVersion.Original:
				if( ropeTheta == default )
					ropeTheta = 10000.0f;
				break;
			case eModelVersion.Instruct02:
				Debug.Assert( ropeTheta != default );
				if( ropeTheta == default )
					ropeTheta = 1000000.0f;
				break;
		}
	}
}