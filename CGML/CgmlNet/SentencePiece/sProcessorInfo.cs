namespace SentencePiece;

/// <summary>Some constants specific to the loaded SentencePiece model</summary>
public struct sProcessorInfo
{
	/// <summary>Size of sentence pieces, which is the same as the size of vocabulary for NMT.</summary>
	public readonly int vocabSize;
	/// <summary>BOS (&lt;s&gt;) id.</summary>
	public readonly int idBOS;
	/// <summary>EOS (&lt;/s&gt;) id.</summary>
	public readonly int idEOS;
	/// <summary>PAD (&lt;pad&gt;) id.</summary>
	public readonly int idPad;
};