namespace Cgml;
using System.Diagnostics;

/// <summary>Factory function to create a set of compute shaders from serialized package</summary>
public static class ShaderFactory
{
	/// <summary>Apply patch table to the array of shader binaries</summary>
	static void applyPatch( ushort[] arr, ushort[]? patch )
	{
		if( null == patch )
			return;
		Debug.Assert( 0 == ( patch.Length % 2 ) );
		for( int i = 0; i < patch.Length; i += 2 )
			arr[ patch[ i ] ] = patch[ i + 1 ];
	}

	/// <summary>Produce array of shader binaries matching optional features of the user's GPU </summary>
	static ushort[] makePatchedShaders( ShaderPackage package, in sDeviceInfo deviceInfo )
	{
		if( deviceInfo.optionalFeatures == eOptionalFeatures.None )
			return package.shaders;    // GPU doesn't support any optional features
		if( null == package.fp1 && null == package.fp2 )
			return package.shaders;    // The package doesn't define any patches for them

		// Make a copy of that array
		ushort[] shaders = package.shaders.ToArray();
		// Apply patches to the array, based on these optional features
		if( deviceInfo.optionalFeatures.HasFlag( eOptionalFeatures.FP64Basic ) )
			applyPatch( shaders, package.fp1 );
		if( deviceInfo.optionalFeatures.HasFlag( eOptionalFeatures.FP64Advanced ) )
			applyPatch( shaders, package.fp2 );

		return shaders;
	}

	/// <summary>Deserialize a set of packaged shaders from the stream, and upload them to GPU</summary>
	public static int createShaders( iContext context, in sDeviceInfo deviceInfo, Stream packageStream )
	{
		ShaderPackage package = ShaderPackage.read( packageStream );

		ushort[] shaders = makePatchedShaders( package, deviceInfo );
		int len = shaders.Length;
		ShaderBinarySlice[] slices = new ShaderBinarySlice[ len ];

		for( int i = 0; i < len; i++ )
		{
			int bin = shaders[ i ];
			ref ShaderBinarySlice slice = ref slices[ i ];
			slice.begin = package.binaries[ bin ];
			slice.end = package.binaries[ bin + 1 ];
		}

		context.createComputeShaders( len, ref slices[ 0 ], package.blob, package.blob.Length );

		Logger.Debug( "Created {0} compute shaders", len );
		return len;
	}
}