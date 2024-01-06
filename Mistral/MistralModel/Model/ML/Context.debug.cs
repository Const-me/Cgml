namespace Mistral.Model;
using Cgml;

partial struct Context
{
#if DEBUG
	static readonly string pathPythonDumps = @"C:\Temp\2remove\Mistral02";
	readonly string? dumps;
	public string? prefix;
#endif

#if DEBUG
	public void dbgCompareTensor( iTensor tensor, string zip )
	{
		if( null == dumps )
			return;
		TensorData data = context.downloadTensor( tensor );
		if( null != prefix )
			zip = $"{prefix}-{zip}";

		string path = Path.Combine( dumps, Path.ChangeExtension( zip, ".zip" ) );
		TensorData test = Torch.TensorLoader.load( path );
		TensorsDiff diff = data.diff( test );
		Logger.Debug( @"{0}: {1}, {2}", zip, data.desc.shape.description(), diff );
	}

	public void dbgCompareTensor( Tensor tensor, string zip ) =>
		dbgCompareTensor( tensor.native, zip );
#else
	public void dbgCompareTensor( iTensor tensor, string zip ) { }
	public void dbgCompareTensor( Tensor tensor, string zip ) { }
#endif
}