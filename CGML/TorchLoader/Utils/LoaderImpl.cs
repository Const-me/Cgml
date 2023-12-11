namespace Torch;
using Cgml;
using System.IO.Compression;
using eMergeTactic = LoadTraits.eMergeTactic;

sealed class LoaderImpl: iWeightsLoader
{
	readonly iDevice device;
	readonly LoadTraits traits;
	public Dictionary<string, iTensor> tensors { get; }

	public LoaderImpl( iDevice device, LoadTraits traits )
	{
		tensors = new Dictionary<string, iTensor>();
		this.device = device;
		this.traits = traits;
	}

	public void Dispose()
	{
		foreach( var tensor in tensors.Values )
			tensor?.Dispose();
		tensors.Clear();
	}

	readonly struct PendingTensor: IComparable<PendingTensor>
	{
		public string key { get; init; }
		public Tensor tensor { get; init; }
		public override string ToString() => $"\"{key}\": {tensor}";

		public int payloadBytes =>
			tensor.shape.countElements() * tensor.storage.dataType.elementSize();

		public int CompareTo( PendingTensor other ) =>
			tensor.offset.CompareTo( other.tensor.offset );
	}

	// Tensors from the same binary file in the ZIP, in the order they were present in the unpickled metadata
	// Key = zip entry name
	sealed class LoadMap: Dictionary<string, List<PendingTensor>> { }

	static LoadMap makeLoadMap( Dictionary<string, Tensor> loaded )
	{
		var res = new LoadMap();
		foreach( var kvp in loaded )
		{
			string entry = kvp.Value.storage.payload;

			if( !res.TryGetValue( entry, out List<PendingTensor>? list ) )
			{
				list = new List<PendingTensor>();
				res.Add( entry, list );
			}

			PendingTensor pt = new PendingTensor
			{
				key = kvp.Key,
				tensor = kvp.Value
			};
			list.Add( pt );
		}

		foreach( List<PendingTensor> v in res.Values )
			v.Sort();

		return res;
	}

	void loadTensors( ZipArchiveEntry entry, LoadMap map )
	{
		string name = entry.Name;
		if( !map.TryGetValue( name, out var list ) )
			throw new ArgumentException( $"ZIP entry \"{entry.FullName}\" doesn't correspond to any tensor" );
		map.Remove( name );

		int cb = list.Sum( pt => pt.payloadBytes );
		if( cb != entry.Length )
			throw new ArgumentException( "Unexpected entry length" );

		using var stream = entry.Open();
		foreach( PendingTensor pt in list )
			loadTensor( stream, pt.tensor, pt.key, pt.payloadBytes );
	}

	static IEnumerable<(ZipArchive, string)> metadataSource( ZipArchives zip, string[] subdirs )
	{
		for( int i = 0; i < zip.length; i++ )
		{
			string subdir = subdirs[ i ];
			yield return (zip[ i ], $"{subdir}/data.pkl");
		}
	}

	public void loadSingle( string path )
	{
		using Stream zipStream = File.OpenRead( path );
		using ZipArchive zip = new ZipArchive( zipStream, ZipArchiveMode.Read );

		string subdir = Path.GetFileNameWithoutExtension( path );
		Dictionary<string, Tensor> metadata = MetadataLoader.load( zip, $"{subdir}/data.pkl" );
		subdir = $"{subdir}/data/";

		var ordered = makeLoadMap( metadata );

		foreach( ZipArchiveEntry entry in zip.Entries )
		{
			string fullName = entry.FullName;
			fullName = fullName.Replace( '\\', '/' );
			if( !fullName.StartsWith( subdir ) )
				continue;

			loadTensors( entry, ordered );
		}

		if( ordered.Count > 0 )
			throw new ApplicationException( "Some tensors were mentioned in the metadata, but the weights were not found in the ZIP" );

		device.waitForWeightsCompressor();
	}

	void verifyMerge( Dictionary<string, Tensor>[] metadata, LoadMap[] maps )
	{
		if( !metadata.Select( d => d.Count ).same() )
			throw new ArgumentException( "Tensors count is different" );
		if( !maps.Select( d => d.Count ).same() )
			throw new ArgumentException( "Payload files count is different" );

		Tensor[] arr = new Tensor[ metadata.Length ];
		foreach( string k in metadata[ 0 ].Keys )
		{
			for( int i = 0; i < arr.Length; i++ )
				arr[ i ] = metadata[ i ][ k ];

			if( !arr.Select( t => t.storage.dataType ).same() )
				throw new ArgumentException( "Data types are different" );

			if( !arr.Select( t => t.shape.size ).same() )
				throw new ArgumentException( "Chunks have different sizes" );
		}
	}

	void iWeightsLoader.loadMultipart( string[] arr )
	{
		if( 1 == arr.Length )
		{
			loadSingle( arr[ 0 ] );
			return;
		}

		using Streams zipStream = new Streams( arr );
		using ZipArchives zip = new ZipArchives( zipStream );

		string[] subdirs = new string[ arr.Length ];
		for( int i = 0; i < subdirs.Length; i++ )
			subdirs[ i ] = $"consolidated.{i:00}";

		Dictionary<string, Tensor>[] metadata =
			MetadataLoader.load( metadataSource( zip, subdirs ) )
			.ToArray();
		LoadMap[] maps = metadata.Select( makeLoadMap ).ToArray();

		verifyMerge( metadata, maps );

		for( int i = 0; i < subdirs.Length; i++ )
			subdirs[ i ] = $"{subdirs[ i ]}/data/";

		ZipArchiveEntry[] entries = new ZipArchiveEntry[ subdirs.Length ];
		List<PendingTensor>[] tensors = new List<PendingTensor>[ subdirs.Length ];
		Span<int> tensorBytes = stackalloc int[ subdirs.Length ];
		Span<TensorShape> shapes = stackalloc TensorShape[ subdirs.Length ];
		Tensor[] tensorParts = new Tensor[ subdirs.Length ];

		using LargeBuffer buffer = new LargeBuffer();

		foreach( ZipArchiveEntry entry in zip[ 0 ].Entries )
		{
			string fullName = entry.FullName;
			fullName = fullName.Replace( '\\', '/' );
			if( !fullName.StartsWith( subdirs[ 0 ] ) )
				continue;

			for( int i = 0; i < tensors.Length; i++ )
				tensors[ i ] = maps[ i ][ entry.Name ];

			entries[ 0 ] = entry;
			for( int i = 1; i < tensors.Length; i++ )
			{
				if( !tensors[ 0 ].Select( p => p.key ).SequenceEqual( tensors[ i ].Select( p => p.key ) ) )
					throw new ArgumentException();

				string name = subdirs[ i ] + entry.Name;
				var e = zip[ i ].GetEntry( name );
				if( null == e )
					throw new ArgumentException();
				entries[ i ] = e;
			}

			for( int i = 0; i < tensors.Length; i++ )
			{
				int cb = tensors[ i ].Sum( pt => pt.payloadBytes );
				if( cb != entries[ i ].Length )
					throw new ArgumentException( "Unexpected entry length" );
			}

			using var streams = new Streams( entries );

			for( int i = 0; i < tensors[ 0 ].Count; i++ )
			{
				int cbTotal = 0, cbMax = 0;
				for( int j = 0; j < tensorBytes.Length; j++ )
				{
					var src = tensors[ j ][ i ];
					tensorParts[ j ] = src.tensor;
					int cb = src.payloadBytes;
					tensorBytes[ j ] = cb;
					cbTotal += cb;
					cbMax = Math.Max( cbMax, cb );
					shapes[ j ] = src.tensor.shape;
				}

				Span<byte> span;
				string key = tensors[ 0 ][ i ].key;
				eMergeTactic merge = traits.tensorMergeTactic( key, shapes );

				if( merge == eMergeTactic.Ignore )
				{
					span = buffer.resize( cbMax );
					streams.copyFirstTensor( span, tensorBytes );
					continue;
				}

				switch( merge )
				{
					case eMergeTactic.ConcatData:
						span = buffer.resize( cbTotal );
						streams.concatTensors( span, tensorBytes );
						break;
					case eMergeTactic.ConcatRows:
						{
							span = buffer.resize( cbTotal );
							int cbElement = tensors[ 0 ][ i ].tensor.storage.dataType.elementSize();
							for( int j = 0; j < tensorBytes.Length; j++ )
								tensorBytes[ j ] = shapes[ j ].size.x * cbElement;
							streams.concatRows( span, shapes[ 0 ].countRows(), tensorBytes );
						}
						break;
					case eMergeTactic.UseFirst:
						span = buffer.resize( cbMax );
						streams.copyFirstTensor( span, tensorBytes );
						span = span.Slice( 0, tensorBytes[ 0 ] );
						break;
					default:
						throw new NotImplementedException();
				}

				loadMergedTensor( tensorParts, key, span, merge );
			}
		}

		device.waitForWeightsCompressor();
	}

	void loadTensor( Stream stream, Tensor tensor, string key, int byteWidth )
	{
		if( tensors.ContainsKey( key ) )
			throw new ApplicationException( "Already loaded" );
		sTensorDesc desc;
		desc.shape = tensor.shape;
		desc.dataType = tensor.storage.dataType;
		desc.usage = eBufferUse.Immutable;
		desc.layout = traits.tensorVramLayout( key );
		eLoadTransform tform = traits.tensorLoadTransform();
		tensors[ key ] = device.loadImmutableTensor( ref desc, stream, byteWidth, tform );
	}

	void loadMergedTensor( Tensor[] tensors, string key, ReadOnlySpan<byte> payload, eMergeTactic merge )
	{
		if( this.tensors.ContainsKey( key ) )
			throw new ApplicationException( "Already loaded" );

		sTensorDesc desc;
		desc.dataType = tensors[ 0 ].storage.dataType;
		desc.usage = eBufferUse.Immutable;
		desc.layout = traits.tensorVramLayout( key );

		if( merge == eMergeTactic.ConcatData )
		{
			if( tensors[ 0 ].shape.isVector )
			{
				int length = tensors.Sum( t => t.shape.size.x );
				desc.shape = TensorShape.rowMajor( length );
			}
			else if( tensors[ 0 ].shape.isMatrix )
			{
				if( !tensors.Select( t => t.shape.size.x ).same() )
					throw new ArgumentException();

				int height = tensors.Sum( t => t.shape.size.y );
				desc.shape = TensorShape.rowMajor( tensors[ 0 ].shape.size.x, height );
			}
			else
				throw new NotImplementedException();
		}
		else if( merge == eMergeTactic.ConcatRows )
		{
			if( !tensors.Select( t => t.shape.size.yzw ).same() )
				throw new ArgumentException();

			int x = tensors.Sum( t => t.shape.size.x );
			Int128 size = tensors[ 0 ].shape.size;
			size = new Int128( x, size.y, size.z, size.w );
			desc.shape = new TensorShape( size );
		}
		else if( merge == eMergeTactic.UseFirst )
			desc.shape = tensors[ 0 ].shape;
		else throw new NotImplementedException();

		var tform = traits.tensorLoadTransform();
		if( tform != eLoadTransform.None )
			throw new NotImplementedException();	// Need to adjust that API as well

		this.tensors[ key ] = device.uploadImmutableTensor( desc, payload );
	}
}