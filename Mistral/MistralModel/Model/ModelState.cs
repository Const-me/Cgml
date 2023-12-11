namespace Mistral.Model;
using Cgml;
using System.Runtime.Serialization;

[DataContract]
sealed class ModelState: iModelState
{
	[DataMember]
	public readonly int window;

	[DataMember]
	public int absolute = 0;

	public sealed class LayerCache
	{
		[DataMember]
		public iTensor? k, v;
	}

	[DataMember]
	public LayerCache[] layers;

	public ModelState( int window, int countLayers )
	{
		this.window = window;
		layers = new LayerCache[ countLayers ];
	}

	void IDisposable.Dispose()
	{
		foreach( var layer in layers )
		{
			layer.k?.Dispose();
			layer.k = null;

			layer.v?.Dispose();
			layer.v = null;
		}
	}
}