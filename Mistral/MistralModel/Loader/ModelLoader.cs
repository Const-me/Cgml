namespace Mistral;
using Cgml;
using System.Reflection;

/// <summary>Public functions to load and save Mistral models</summary>
/// <remarks>This library implements custom proprietary <c>*.cgml</c> format for these models.<br/>
/// It can also import the original PyTorch format.</remarks>
public static partial class ModelLoader
{
	/// <summary>Enumerate graphics adapters on this computer, and return their names.</summary>
	public static string[] listGraphicAdapters() =>
		Library.listGraphicAdapters();

	static iModel loadImpl( Func<Device, iModel> fn, sDeviceParams deviceParams )
	{
		Device dev = Library.createDevice( deviceParams );
		try
		{
			sDeviceInfo info = dev.device.getDeviceInfo();
			const double mulGb = 1.0 / ( 1 << 30 );
			Logger.Info( "Created DirectCompute device \"{0}\", {1:F1} GB VRAM, feature level {2}.{3}",
				info.name, mulGb * info.vram, info.featureLevelMajor, info.featureLevelMinor );
			createShaders( dev.context, info );
			return fn( dev );
		}
		catch
		{
			dev.Dispose();
			throw;
		}
	}

	static void createShaders( iContext ctx, in sDeviceInfo devInfo )
	{
		const string logicalName = @"Model\Generated\Shaders.bin";
		Stream? stm = Assembly.GetExecutingAssembly().GetManifestResourceStream( logicalName );
		if( null == stm )
			throw new ApplicationException( $"Embedded resource missing: {logicalName}" );
		ShaderFactory.createShaders( ctx, devInfo, stm );
	}

	/// <summary>Load model in CGML format</summary>
	public static iModel load( string path, sDeviceParams deviceParams, Action<double>? pfnProgress = null )
	{
		if( !File.Exists( path ) )
			throw new FileNotFoundException();

		Func<Device, iModel> pfn = delegate ( Device dev )
		{
			Logger.Debug( "Loading CGML model.." );
			return new Model.Model( dev, path, pfnProgress );
		};
		return loadImpl( pfn, deviceParams );
	}

	/// <summary>Save model in CGML format</summary>
	public static void save( iModel model, string tokenizer, string cgml, Action<double>? pfnProgress = null )
	{
		Model.Model m = (Model.Model)model;
		m.save( cgml, tokenizer, pfnProgress );
	}
}