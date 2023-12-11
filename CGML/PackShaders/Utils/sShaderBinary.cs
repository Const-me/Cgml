namespace PackShaders;
using System.Runtime.InteropServices;

[StructLayout( LayoutKind.Auto )]
record struct sShaderBinary: IComparable<sShaderBinary>
{
	/// <summary>File name without extension</summary>
	public string name;

	/// <summary>Full path to the HLSL source file</summary>
	public readonly string sourcePath;

	/// <summary>Compiled shader</summary>
	public readonly byte[] dxbc;

	sShaderBinary( string hlsl )
	{
		sourcePath = Path.Combine( Program.inputs.projectDir, hlsl );
		if( !File.Exists( sourcePath ) )
			throw new ApplicationException( $"PSBS05: HLSL file is missing, expected there: \"{sourcePath}\"" );

		name = Path.GetFileNameWithoutExtension( hlsl );

		string dxbc = Path.ChangeExtension( name, ".cso" );
		dxbc = Path.Combine( Program.inputs.temp, dxbc );
		if( !File.Exists( dxbc ) )
			throw new ApplicationException( $"PSBS06: DXBC file is missing, expected there: \"{dxbc}\"" );

		this.dxbc = File.ReadAllBytes( dxbc );
	}

	/// <summary>Produce a sequence of shader binaries from a sequence of source files</summary>
	public static IEnumerable<sShaderBinary> load( IEnumerable<string> sources )
	{
		foreach( string hlsl in sources )
			yield return new sShaderBinary( hlsl );
	}

	/// <summary>A string for debugger</summary>
	public override string ToString() =>
		$"{name}.hlsl, {dxbc.Length} bytes";

	/// <summary>Compare by names, case insensitively</summary>
	int IComparable<sShaderBinary>.CompareTo( sShaderBinary other ) =>
		string.Compare( name, other.name, StringComparison.InvariantCultureIgnoreCase );
}