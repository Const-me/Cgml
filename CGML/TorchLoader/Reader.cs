namespace Torch;
using Cgml;
using Razorvine.Pickle;
using Razorvine.Pickle.Objects;

/// <summary>Factory for Torch-specific classes being de-serialized from Pickle stream</summary>
sealed class Reader: Unpickler
{
	sealed class StorageConstructor: IObjectConstructor
	{
		readonly eDataType dataType;
		public StorageConstructor( eDataType dataType )
		{
			this.dataType = dataType;
		}

		public object construct( object[] args ) =>
			new Storage( dataType, args );
	}

	static Reader()
	{
		registerConstructor( "torch", "HalfStorage", new StorageConstructor( eDataType.FP16 ) );
		registerConstructor( "torch", "FloatStorage", new StorageConstructor( eDataType.FP32 ) );
		registerConstructor( "torch", "BFloat16Storage", new StorageConstructor( eDataType.BF16 ) );
		registerConstructor( "torch._utils", "_rebuild_tensor_v2", new AnyClassConstructor( typeof( Tensor ) ) );
	}

	protected internal override object persistentLoad( object args )
	{
		if( args is object[] arr )
		{
			string name = (string)arr[ 0 ];
			StorageConstructor ctor = (StorageConstructor)arr[ 1 ];
			ReadOnlySpan<object> span = arr;
			span = span.Slice( 2 );
			return construct( name, ctor, span );
		}

		return base.persistentLoad( args );
	}

	static object construct( string name, StorageConstructor ctor, ReadOnlySpan<object> args )
	{
		switch( name )
		{
			case "storage":
				return ctor.construct( args.ToArray() );
		}
		throw new NotImplementedException();
	}
}