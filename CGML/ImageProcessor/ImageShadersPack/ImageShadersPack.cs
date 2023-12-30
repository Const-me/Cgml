namespace ImageShadersPack;
using System.Runtime.CompilerServices;

static class Program
{
	static string makeCgmlRoot( [CallerFilePath] string? thisSourceFile = null )
	{
		string dir = Path.GetDirectoryName( thisSourceFile ) ?? throw new ApplicationException();
		dir = Path.GetDirectoryName( dir ) ?? throw new ApplicationException();
		dir = Path.GetDirectoryName( dir ) ?? throw new ApplicationException();
		return dir;
	}

#if DEBUG
	const string configuration = "Debug";
#else
	const string configuration = "Release";
#endif

	static string makeSourcePath( string dir ) =>
		Path.Combine( dir, "ImageProcessor", "ImageShaders", "x64", configuration );

	static string resultDir( string dir ) =>
		Path.Combine( dir, "Cgml", "ImageProcessor", "Generated" );
	static string resultName() =>
		$"shaders-{configuration}.inl";

	static void Main( string[] args )
	{
		string root = makeCgmlRoot();
		string source = makeSourcePath( root );
		BinaryPackage package = new BinaryPackage( source );

		string dir = resultDir( root );
		string name = resultName();
		package.generate( dir, name );
	}
}