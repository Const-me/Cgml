namespace PackShaders;
using Cgml;

/// <summary>Name of the HLSL, with dot-separated suffixes parsed into optional GPU features</summary>
readonly struct sParsedName
{
	public readonly string name;
	public readonly eOptionalFeatures features;

	static readonly Dictionary<string, eOptionalFeatures> dictSuffixes = new Dictionary<string, eOptionalFeatures>( StringComparer.InvariantCultureIgnoreCase )
	{
		{ "fp0", eOptionalFeatures.None },
		{ "fp1", eOptionalFeatures.FP64Basic },
		{ "fp2", eOptionalFeatures.FP64Advanced },
	};

	public sParsedName( string name )
	{
		features = eOptionalFeatures.None;

		int idx = name.IndexOf( '.' );
		if( idx < 0 )
		{
			this.name = name;
			return;
		}

		this.name = name.Substring( 0, idx );
		string[] suffixes = name.Substring( idx + 1 ).Split( '.' );
		foreach( string suffix in suffixes )
		{
			if( dictSuffixes.TryGetValue( suffix, out var bit ) )
			{
				features |= bit;
				continue;
			}
			throw new ArgumentException( $"PSBS07: unrecognized feature suffix \".{suffix}\"" );
		}
	}
}

sealed class GroupedShaders
{
	public readonly IReadOnlyList<sShaderBinary> binaries;

	public readonly string[] names;
	readonly Dictionary<eOptionalFeatures, ushort>[] groups;

	public GroupedShaders( IReadOnlyList<sShaderBinary> binaries )
	{
		this.binaries = binaries;

		var dict = new Dictionary<string, Dictionary<eOptionalFeatures, ushort>>( StringComparer.InvariantCultureIgnoreCase );

		for( int i = 0; i < binaries.Count; i++ )
		{
			sShaderBinary bin = binaries[ i ];
			sParsedName pn = new sParsedName( bin.name );

			Dictionary<eOptionalFeatures, ushort>? inner;
			if( !dict.TryGetValue( pn.name, out inner ) )
			{
				inner = new Dictionary<eOptionalFeatures, ushort>();
				dict.Add( pn.name, inner );
			}
			if( inner.TryAdd( pn.features, (ushort)i ) )
				continue;
			throw new ArgumentException( $"PSBS08: the project contains multiple shaders with name={pn.name}, features {pn.features}" );
		}

		names = dict.Keys.ToArray();
		groups = dict.Values.ToArray();

		// Verify all shaders have a binary with eOptionalFeatures.None
		for( int i = 0; i < groups.Length; i++ )
		{
			if( groups[ i ].ContainsKey( eOptionalFeatures.None ) )
				continue;
			throw new ArgumentException( $"PSBS09: the shader {names[ i ]} doesn't have a compatible version" );
		}
	}

	public ushort[] basic()
	{
		ushort[] arr = new ushort[ groups.Length ];
		for( int i = 0; i < groups.Length; i++ )
			arr[ i ] = groups[ i ][ eOptionalFeatures.None ];
		return arr;
	}

	ushort[]? makePatchTable( eOptionalFeatures feature )
	{
		List<ushort>? list = null;
		for( int i = 0; i < groups.Length; i++ )
		{
			Dictionary<eOptionalFeatures, ushort> dict = groups[ i ];
			if( !dict.TryGetValue( feature, out ushort bin ) )
				continue;
			list ??= new List<ushort>();
			list.Add( (ushort)i );
			list.Add( bin );
		}
		return list?.ToArray();
	}

	public ushort[]? fp1() => makePatchTable( eOptionalFeatures.FP64Basic );
	public ushort[]? fp2() => makePatchTable( eOptionalFeatures.FP64Advanced );

	public IEnumerable<(string,string)> list()
	{
		for( int i = 0; i < groups.Length; i++ )
		{
			string name = names[ i ];
			ushort bin = groups[ i ][ eOptionalFeatures.None ];
			string source = binaries[ bin ].sourcePath;
			yield return (name, source);
		}
	}
}