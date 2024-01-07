namespace PackShaders;
using System.Text;
using System.Text.RegularExpressions;

/// <summary>Function to parse HLSL files, and generate C# language projection.</summary>
static class SourceParser
{
	// These regular expressions are horrible.
	// Ideally, should be using proper AST parsing, or maybe DXBC reflection, to extract API from the shaders.
	// However, these alternatives were much harder to implement.

	// Match lines like "Buffer<float> source: register( t0 );"
	// Capture `source` and `0` values
	static readonly Regex reSrv = new Regex( @"\s(\S+)\s*:\s*register\s*\(\s*t([\d+])\s*\)\s*;", RegexOptions.IgnoreCase );

	// Match lines like "RWBuffer<float> result: register( u0 );"
	// Capture `result` and `0` values
	static readonly Regex reUav = new Regex( @"\s(\S+)\s*:\s*register\s*\(\s*u([\d+])\s*\)\s*;", RegexOptions.IgnoreCase );

	// Match lines like "cbuffer Constants: register( b0 )"
	// Capture `Constants` value
	static readonly Regex reCbuffer = new Regex( @"^\s*cbuffer\s(\S+)\s*:\s*register\s*\(\s*b0\s*\)\s*", RegexOptions.IgnoreCase );

	static readonly Regex reField = new Regex( @"^\s*(\S+)\s+(\S+)\s*:\s*packoffset\s*\(\s*c([\d]+)(.[xyzw])?\s*\)\s*;", RegexOptions.IgnoreCase );

	static bool isComment( this string line )
	{
		line = line.Trim();
		return line.StartsWith( "//" );
	}

	static string group( this Match m, int i )
	{
		return m.Groups[ i + 1 ].Value;
	}

	static string? groupOpt( this Match m, int i )
	{
		if( m.Groups.Count <= i )
			return null;
		return m.Groups[ i + 1 ].Value;
	}

	static string? extractComment( string[] lines, int iNext )
	{
		if( iNext == 0 )
			return null;
		string line = lines[ iNext - 1 ];
		if( !line.isComment() )
			return null;
		line = line.Trim();
		line = line.Substring( 2 ).TrimStart();
		return line;
	}

	static (string, byte) makeCsType( string hlsl ) => hlsl switch
	{
		"uint" => ("uint", 1),
		"int" => ("int", 1),
		"float" => ("float", 1),
		"uint4" => ("Int128", 4),
		"int4" => ("Int128", 4),
		"uint2" => ("uint2", 2),
		"uint3" => ("uint3", 3),
		_ => throw new NotImplementedException( $"HLSL type \"{hlsl}\" is not currently supported" )
	};

	static ConstantBufferField makeField( string[] lines, int iCurrent, Match m )
	{
		(string csType, byte sz) = makeCsType( m.group( 0 ) );
		string name = m.group( 1 );
		byte idxVec = byte.Parse( m.group( 2 ) );
		byte idxLane = m.groupOpt( 3 ) switch
		{
			null => 0,
			"" => 0,
			".x" => 0,
			".y" => 1,
			".z" => 2,
			".w" => 3,
			_ => throw new ArgumentException()
		};

		string? comment = extractComment( lines, iCurrent );

		return new ConstantBufferField
		{
			name = name,
			idxVector = idxVec,
			idxOffset = idxLane,
			size = sz,
			comment = comment,
			csType = csType
		};
	}

	static ResourceBinding parseCbuffer( string[] lines, ref int iCurrent )
	{
		string? comment = extractComment( lines, iCurrent );
		if( !lines[ ++iCurrent ].Trim().StartsWith( "{" ) )
			throw new ArgumentException();

		List<ConstantBufferField> fields = new List<ConstantBufferField>();
		while( true )
		{
			string line = lines[ ++iCurrent ];
			if( string.IsNullOrWhiteSpace( line ) )
				continue;
			if( line.isComment() )
				continue;
			if( line.Trim().StartsWith( "}" ) )
				break;

			Match m = reField.Match( line );
			if( !m.Success )
				throw new ArgumentException();

			fields.Add( makeField( lines, iCurrent, m ) );
		}

		return new ResourceBinding
		{
			kind = eResourceKind.CBuffer,
			slot = 0,
			name = "",
			comment = comment,
			extraData = fields.ToArray()
		};
	}

	static ShaderReflection parseBindings( string sourcePath )
	{
		string[] lines = File.ReadAllLines( sourcePath, Encoding.ASCII );

		ResourceBinding makeTensor( int i, Match m, eResourceKind kind )
		{
			string name = m.group( 0 );
			byte idx = byte.Parse( m.group( 1 ) );
			return new ResourceBinding()
			{
				kind = kind,
				slot = idx,
				name = name,
				comment = extractComment( lines, i )
			};
		}

		List<ResourceBinding> result = new List<ResourceBinding>();

		bool foundConstants = false;
		for( int i = 0; i < lines.Length; i++ )
		{
			string line = lines[ i ];
			if( string.IsNullOrWhiteSpace( line ) || line.isComment() )
				continue;

			Match m = reUav.Match( line );
			if( m.Success )
			{
				result.Add( makeTensor( i, m, eResourceKind.UAV ) );
				continue;
			}

			m = reSrv.Match( line );
			if( m.Success )
			{
				result.Add( makeTensor( i, m, eResourceKind.SRV ) );
				continue;
			}

			m = reCbuffer.Match( line );
			if( m.Success )
			{
				if( foundConstants )
					throw new ApplicationException( "Code generator doesn't support multiple constant buffers" );
				foundConstants = true;
				result.Add( parseCbuffer( lines, ref i ) );
				continue;
			}
		}

		string? comment = extractComment( lines, 1 );
		if( !foundConstants && comment != ShaderReflection.IgnoreMarker )
			throw new ApplicationException( "Code generator requires a constant buffer in the shader" );

		return new ShaderReflection
		{
			bindings = result.ToArray(),
			comment = comment,
		};
	}

	/// <summary>Parse HLSL files, and generate two C# source files:<br/>
	/// • <c>ConstantBuffers.cs</c> with constant buffers.<br/>
	/// • <c>ContextOps.cs</c> with extension methods to dispatch these shaders.</summary>
	public static void generateBindings( GroupedShaders shaders )
	{
		using CbufferWriter structsWriter = new CbufferWriter();
		using OpsWriter opsWriter = new OpsWriter();

		foreach( (string name, string hlslPath) in shaders.list() )
		{
			ShaderReflection hlsl = parseBindings( hlslPath );
			if( hlsl.codegenIgnore )
				continue;

			ResourceBinding[] arrCb = hlsl.bindings.Where( x => x.kind == eResourceKind.CBuffer ).ToArray();
			if( arrCb.Length != 1 )
				throw new ArgumentException();

			structsWriter.add( name, arrCb[ 0 ] );
			opsWriter.add( name, hlsl );
		}
	}
}