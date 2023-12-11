namespace Mistral.Model;
using Cgml;
using System.Diagnostics;

sealed partial class Model: iModel
{
	iModelState iModel.stateBackup( iModelState? input )
	{
		using var rootBlock = dev.context.profilerBlock( (ushort)eProfilerBlock.BackupRestoreState );

		int layers = transformer.layers.Length;
		ModelState res;
		if( input == null )
		{
			res = new ModelState( transformer.parameters.slidingWindow, layers );
		}
		else
		{
			res = (ModelState)input;
			if( res.window != transformer.parameters.slidingWindow || res.layers.Length != layers )
				throw new ArgumentException();
		}

		res.absolute = cacheMetadata?.absolute ?? 0;
		if( 0 == res.absolute )
			return res;

		sTensorDesc desc = new sTensorDesc
		{
			shape = new TensorShape( transformer.parameters.attnCacheSize ),
			dataType = eDataType.FP16,
			usage = eBufferUse.ReadWrite,
			layout = eTensorLayout.Dense
		};

		iTensor createIfNeeded( ref iTensor? tensor, ref sTensorDesc desc )
		{
			if( null != tensor )
			{
				sTensorDesc oldDesc = tensor.getDesc();
				Debug.Assert( oldDesc == desc );
			}
			else
				tensor = dev.device.createTensor( ref desc );
			return tensor;
		}

		iContext context = dev.context;
		for( int i = 0; i < layers; i++ )
		{
			ModelState.LayerCache dest = res.layers[ i ];
			iTensor k = createIfNeeded( ref dest.k, ref desc );
			iTensor v = createIfNeeded( ref dest.v, ref desc );

			Attention source = transformer.layers[ i ].attention;
			source.backup( context, k, v );
		}
		return res;
	}

	void resetState()
	{
		cacheMetadata = new RotatingCacheMetadata( transformer.parameters.slidingWindow );
		foreach( var layer in transformer.layers )
			layer.attention.clear( dev.context );
	}

	void iModel.stateRestore( iModelState? state )
	{
		using var rootBlock = dev.context.profilerBlock( (ushort)eProfilerBlock.BackupRestoreState );
		
		if( null == state )
		{
			resetState();
			return;
		}

		ModelState sourceState = (ModelState)state;

		int layers = transformer.layers.Length;
		if( sourceState.window != transformer.parameters.slidingWindow || sourceState.layers.Length != layers )
			throw new ArgumentException();

		if( 0 == sourceState.absolute )
		{
			resetState();
			return;
		}

		iContext context = dev.context;
		for( int i = 0; i < layers; i++ )
		{
			ModelState.LayerCache sourceLayer = sourceState.layers[ i ];
			if( null == sourceLayer.k || null == sourceLayer.v )
				throw new ArgumentException();

			Attention dest = transformer.layers[ i ].attention;
			dest.restore( context, sourceLayer.k, sourceLayer.v );
		}

		cacheMetadata = new RotatingCacheMetadata( sourceState.window, sourceState.absolute );
	}
}