namespace PackShaders;
using System.Runtime.CompilerServices;

/// <summary>Input parameters for the tool, extracted from these custom environment variables</summary>
sealed class Inputs
{
	public readonly string projectPath;
	public readonly string projectDir;
	public readonly string config;
	public readonly string temp;
	public readonly string result;
	public readonly string ns;

	static string env( string name )
	{
		string? res = Environment.GetEnvironmentVariable( name );
		if( null == res )
			throw new ApplicationException( $"PSBS01: The required environment variable is missing: %{name}%" );
		return unquote( res );
	}

	/// <summary>If the input string starts and ends with a quote, extract the middle slice</summary>
	static string unquote( string s )
	{
		if( s.Length < 2 )
			return s;
		char c0 = s[ 0 ];
		if( c0 != s[ s.Length - 1 ] )
			return s;
		if( c0 == '"' || c0 == '\'' )
			return s.Substring( 1, s.Length - 2 );
		return s;
	}

	/// <summary>Populate parameters from environment variables of the current process</summary>
	public Inputs()
	{
		projectPath = env( "SHADERS_PROJECT" );
		if( !File.Exists( projectPath ) )
			throw new ApplicationException( $"PSBS02: The input C++ project is not found, expected there: \"{projectPath}\"" );
		projectDir = Path.GetDirectoryName( projectPath ) ?? throw new ApplicationException( "PSBS03" );

		config = env( "SHADERS_CONFIG" );

		temp = env( "SHADERS_TEMP" );
		if( !Path.IsPathRooted( temp ) )
			temp = Path.Combine( projectDir, temp );
		if( !Directory.Exists( temp ) )
			throw new ApplicationException( $"PSBS04: the project output directory doesn't exist, expected there: \"{temp}\"" );

		result = env( "SHADERS_RESULT" );

		ns = env( "SHADERS_NAMESPACE" );
	}

	/// <summary>Set correct environment variables to be able to debug this tool without MSBuild</summary>
	public static void dbgSetupMistralShaders( [CallerFilePath] string? path = null )
	{
		string? dir = Path.GetDirectoryName( path );
		dir = Path.GetDirectoryName( dir );
		dir = Path.GetDirectoryName( dir );
		string root = dir ?? throw new ApplicationException();
#if DEBUG
		const string config = "Debug";
#else
		const string config = "Release";
#endif
		static void setEnv( string name, string val )
		{
			Environment.SetEnvironmentVariable( name, val, EnvironmentVariableTarget.Process );
		}

		setEnv( "SHADERS_PROJECT", Path.Combine( root, "Mistral", "MistralShaders", "MistralShaders.vcxproj" ) );
		setEnv( "SHADERS_CONFIG", config );
		setEnv( "SHADERS_TEMP", $"x64\\{config}" );
		setEnv( "SHADERS_RESULT", Path.Combine( root, "Mistral", "MistralModel", "Model", "Generated" ) );
		setEnv( "SHADERS_NAMESPACE", "Mistral.Model" );
	}
}