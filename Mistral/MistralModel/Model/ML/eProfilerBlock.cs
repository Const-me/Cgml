namespace Mistral.Model;
using Cgml;

enum eProfilerBlock: ushort
{
	Generate = 1,
	PreFill = 2,
	MakeToken = 3,
	Layer = 4,
	BackupRestoreState = 5,
}

static class FormatExt
{
	static string blockName( ushort id )
	{
		eProfilerBlock e = (eProfilerBlock)id;
		if( Enum.IsDefined( e ) )
			return e.ToString();
		throw new ArgumentException();
	}

	static string shaderName( ushort id )
	{
		eShader e = (eShader)id;
		if( Enum.IsDefined( e ) )
			return e.ToString();
		throw new ArgumentException();
	}

	public static IEnumerable<string> formatted( this ProfilerResult[]? arr )
	{
		return arr.formatted( blockName, shaderName );
	}
}