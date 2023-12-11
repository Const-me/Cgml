namespace Mistral.Model;
using Cgml;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Serialization;

[DataContract]
sealed class FeedForward: IDisposable
{
	[DataMember]
	internal readonly iTensor w1, w2, w3;

	public FeedForward( int n, Dictionary<string, iTensor> tensors )
	{
		w1 = tensors[ $"layers.{n}.feed_forward.w1.weight" ];
		w2 = tensors[ $"layers.{n}.feed_forward.w2.weight" ];
		w3 = tensors[ $"layers.{n}.feed_forward.w3.weight" ];
	}

	public Tensor forward( in Context ctx, Tensor x )
	{
		Tensor tmp = ctx.columnProduct( x, w1, ref ctx.temp.ff1 );
		Tensor tmp2 = ctx.columnProduct( x, w3, ref ctx.temp.ff2 );
		ctx.SiLU( tmp, tmp2 );
		ctx.unbindInputs();
		tmp = ctx.columnProduct( tmp, w2, ref ctx.temp.ff2 );
		return tmp;
	}

	public void Dispose()
	{
		w1?.Dispose();
		w2?.Dispose();
		w3?.Dispose();
	}

	public Vector128<long> getMemoryUse()
	{
		Vector128<long> v = w1.getMemoryUse();
		v = Sse2.Add( v, w2.getMemoryUse() );
		v = Sse2.Add( v, w3.getMemoryUse() );
		return v;
	}

	public void getMemoryUse( Dictionary<string, Vector128<long>> dict )
	{
		dict.addTensorMemory( w1 );
		dict.addTensorMemory( w2 );
		dict.addTensorMemory( w3 );
	}

	public long totalWeights()
	{
		long res = 0;
		res += w1.countElements();
		res += w2.countElements();
		res += w3.countElements();
		return res;
	}
}