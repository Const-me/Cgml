using Cgml;

namespace PackShaders;

static class Program
{
	static Inputs? m_inputs;
	public static Inputs inputs => m_inputs ?? throw new ArgumentNullException();

	static ShaderPackage makePackage( GroupedShaders shaders )
	{
		MemoryStream ms = new MemoryStream();

		List<int> offsets = new List<int>( shaders.binaries.Count + 1 );
		foreach( var bin in shaders.binaries )
		{
			offsets.Add( (int)ms.Length );
			ms.Write( bin.dxbc );
		}
		offsets.Add( (int)ms.Length );

		ShaderPackage res = new ShaderPackage();
		res.blob = ms.ToArray();
		res.binaries = offsets.ToArray();
		res.shaders = shaders.basic();
		res.fp1 = shaders.fp1();
		res.fp2 = shaders.fp2();
		return res;
	}

	static void mainImpl()
	{
		// Load all shaders compiled by the C++ project
		var sources = ProjectParser.parse( inputs.projectPath );
		List<sShaderBinary> binaries = sShaderBinary.load( sources ).ToList();
		// Sort them by HLSL name, case-insensitive
		binaries.Sort();

		// Hash DXBC and HLSL content. DXBC is in memory by now, HLSL files are on disk. Modern SSDs are fast.
		Guid inputHash = InputHash.compute( binaries );
		// When no changes were detected, quit ASAP. This tool runs after each build.
		if( InputHash.isCurrent( inputHash ) )
			return;

		// Parse names into features, and group the shaders
		GroupedShaders grouped = new GroupedShaders( binaries );

		// Produce serialized package
		ShaderPackage package = makePackage( grouped );
		string bin = $"Shaders{inputs.config}.bin";
		Directory.CreateDirectory( inputs.result );
		bin = Path.Combine( inputs.result, bin );
		using( var f = File.Create( bin ) )
			package.write( f );

		// Generate C# code and data structures to dispatch them
		SourceParser.generateBindings( grouped );
		// Generate C# enum for shader IDs
		ShaderNames.write( Path.Combine( inputs.result, "eShader.cs" ), grouped.names );
		// Update that hash stored on disk
		InputHash.store( inputHash );

		// Print success message
		const double mulKb = ( 1.0 / 1024.0 );
		Console.WriteLine( "Compressed {0} compute shaders, {1:F1} kb -> {2:F1} kb",
			binaries.Count,
			mulKb * package.blob.Length,
			mulKb * new FileInfo( bin ).Length );
	}

	static int Main( string[] args )
	{
		try
		{
			// Inputs.dbgSetupMistralShaders();
			m_inputs = new Inputs();
			mainImpl();
			return 0;
		}
		catch( Exception e )
		{
			Console.Error.WriteLine( e.Message );
			return e.HResult;
		}
	}
}