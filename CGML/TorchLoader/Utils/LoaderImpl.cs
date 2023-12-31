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

	/// <summary>True when all tensors on the list are duplicates:
	/// same shape, same data type, and the length matches the length of the ZIP entry</summary>
	bool isDuplicateTensors( long entryLength, List<PendingTensor> list )
	{
		PendingTensor first = list[ 0 ];
		if( first.payloadBytes != entryLength )
			return false;

		for( int i = 1; i < list.Count; i++ )
		{
			PendingTensor pt = list[ i ];
			if( pt.payloadBytes != entryLength )
				return false;
			if( pt.tensor.storage.dataType != first.tensor.storage.dataType )
				return false;
			if( pt.tensor.shape != first.tensor.shape )
				return false;
		}
		return true;
	}

	/// <summary>When all tensors on the list are duplicates, load one of them,
	/// set multiple output dictionary entries into the same object, and log warning about it.</summary>
	bool tryLoadDupes( Stream stream, long entryLength, List<PendingTensor> list )
	{
		if( isDuplicateTensors( entryLength, list ) )
		{
			PendingTensor pt = list[ 0 ];
			loadTensor( stream, pt.tensor, pt.key, pt.payloadBytes );
			iTensor tensor = tensors[ pt.key ];
			for( int i = 1; i < list.Count; i++ )
				tensors.Add( list[ i ].key, tensor );

			// Log a warning message
			string str = string.Join( ", ", list.Select( pt => pt.key ) );
			Logger.Warning( $"Detected duplicate tensors: {str}" );
			return true;
		}
		return false;
	}

	static int tensorSliceEnd( in PendingTensor pt )
	{
		int elts = pt.tensor.offset + pt.tensor.shape.countElements();
		return elts * pt.tensor.storage.dataType.elementSize();
	}

	/// <summary>Advance the stream forward by the specified count of bytes</summary>
	static void skipBytes( Stream stream, int count )
	{
		if( count < 0 )
			throw new ArgumentOutOfRangeException();
		if( 0 == count )
			return;

		if( stream.CanSeek )
		{
			stream.Seek( count, SeekOrigin.Current );
			return;
		}

		// Seek anyway, by reading and then discarding bytes from the stream
		const int maxBufferSize = 1024 * 16;
		byte[] buffer = new byte[ Math.Min( maxBufferSize, count ) ]; // You can adjust the buffer size as needed
		while( count > 0 )
		{
			int requested = Math.Min( buffer.Length, count );
			int received = stream.Read( buffer, 0, requested );
			if( 0 == received )
				throw new EndOfStreamException();
			count -= received;
		}
	}

	/// <summary>Load tensor[s] from a ZIP entry which has paddings around or between the payloads of the tensors</summary>
	/// <remarks>Implemented for <c>model.norm.weight</c> tensor inside <c>Mistral-7B-Instruct-v0.2/pytorch_model-00003-of-00003.bin</c> ZIP file.<br/>
	/// That tensor has 8kb of data in 500MB ZIP entry, without other tensors in the entry.</remarks>
	void loadPadded( Stream stream, List<PendingTensor> list, ZipArchiveEntry entry, string zipName )
	{
		eDataType dt = list[ 0 ].tensor.storage.dataType;
		int cbElement = dt.elementSize();
		for( int i = 1; i < list.Count; i++ )
		{
			PendingTensor curr = list[ i ];
			if( curr.tensor.storage.dataType != dt )
				throw new ArgumentException( "A single ZIP entry contains tensors of different data types. This is not supported" );

			PendingTensor prev = list[ i - 1 ];
			int prevEnd = tensorSliceEnd( prev );
			int currBegin = cbElement * curr.tensor.offset;
			if( prevEnd > currBegin )
				throw new ArgumentException( "Overlapped tensors in a single ZIP entry" );
		}

		long entryLength = entry.Length;
		{
			PendingTensor last = list[ list.Count - 1 ];
			int lastEnd = tensorSliceEnd( last );
			if( lastEnd > entryLength )
				throw new ArgumentException( $"ZIP entry is {entryLength} bytes, metadata says the last tensor in that entry ends at offset {lastEnd}" );
		}

		// Things are good so far, load these tensors
		int off = 0;
		foreach( PendingTensor pt in list )
		{
			int begin = cbElement * pt.tensor.offset;
			if( begin < off )
				throw new ApplicationException();

			if( begin > off )
			{
				skipBytes( stream, begin - off );
				off = begin;
			}

			int payloadBytes = pt.payloadBytes;
			loadTensor( stream, pt.tensor, pt.key, payloadBytes );
			off += payloadBytes;
		}

		string wasted = Cgml.MiscUtils.printMemoryUse( entryLength - list.Sum( pt => pt.payloadBytes ) );
		string tensors = string.Join( ", ", list.Select( pt => pt.key ) );
		Logger.Warning( $"{wasted} unused data in {zipName}/{entry.FullName}: {tensors}" );
	}

	void loadTensors( ZipArchiveEntry entry, LoadMap map, string zipName )
	{
		string name = entry.Name;
		if( !map.TryGetValue( name, out var list ) )
			throw new ArgumentException( $"ZIP entry \"{entry.FullName}\" doesn't correspond to any tensor" );
		map.Remove( name );

		int cb = list.Sum( pt => pt.payloadBytes );
		using Stream stream = entry.Open();
		if( cb == entry.Length )
		{
			foreach( PendingTensor pt in list )
				loadTensor( stream, pt.tensor, pt.key, pt.payloadBytes );
			return;
		}

		if( tryLoadDupes( stream, entry.Length, list ) )
			return;

		loadPadded( stream, list, entry, zipName );
	}

	static IEnumerable<(ZipArchive, string)> metadataSource( ZipArchives zip, string[] subdirs )
	{
		for( int i = 0; i < zip.length; i++ )
		{
			string subdir = subdirs[ i ];
			yield return (zip[ i ], $"{subdir}/data.pkl");
		}
	}

	void loadSingle( string path, bool waitForCompressor )
	{
		using Stream zipStream = File.OpenRead( path );
		using ZipArchive zip = new ZipArchive( zipStream, ZipArchiveMode.Read );

		string subdir = Path.GetFileNameWithoutExtension( path );
		Dictionary<string, Tensor> metadata = MetadataLoader.load( zip, $"{subdir}/data.pkl" );
		subdir = $"{subdir}/data/";

		LoadMap ordered = makeLoadMap( metadata );

		foreach( ZipArchiveEntry entry in zip.Entries )
		{
			string fullName = entry.FullName;
			fullName = fullName.Replace( '\\', '/' );
			if( !fullName.StartsWith( subdir ) )
				continue;

			loadTensors( entry, ordered, Path.GetFileName( path ) );
		}

		if( ordered.Count > 0 )
			throw new ApplicationException( "Some tensors were mentioned in the metadata, but the weights were not found in the ZIP" );

		if( waitForCompressor )
			device.waitForWeightsCompressor();
	}
	void iWeightsLoader.loadSingle( string path ) =>
		loadSingle( path, true );

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
			loadSingle( arr[ 0 ], true );
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
		eLoadTransform tform = traits.tensorLoadTransform( tensor.storage.dataType, key );
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

		var tform = traits.tensorLoadTransform( desc.dataType, key );
		if( tform != eLoadTransform.None )
			throw new NotImplementedException();    // Need to adjust that API as well

		this.tensors[ key ] = device.uploadImmutableTensor( desc, payload );
	}

	void iWeightsLoader.loadTransformer( TransformerIndex index )
	{
		try
		{
			foreach( string path in index.listDataFiles() )
				loadSingle( path, false );
		}
		catch
		{
			foreach( iTensor i in tensors.Values )
				i.Dispose();
			tensors.Clear();
			throw;
		}
		finally
		{
			device.waitForWeightsCompressor();
		}
	}
}