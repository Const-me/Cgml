namespace PackShaders;
using System.Buffers;
using System.Diagnostics;
using System.Security.Cryptography;

static class InputHash
{
	/// <summary>Add array of bytes to MD5</summary>
	static void addBytes( this MD5 md5, byte[] arr ) =>
		md5.TransformBlock( arr, 0, arr.Length, null, 0 );

#if !NET7_0_OR_GREATER
	/// <summary>Compatibility for .NET &lt; 7.0</summary>
	public static void ReadExactly( this Stream stream, Span<byte> buffer )
	{
		while( !buffer.IsEmpty )
		{
			int received = stream.Read( buffer );
			if( received <= 0 )
				throw new EndOfStreamException();
			Debug.Assert( received <= buffer.Length );
			buffer = buffer.Slice( received );
		}
	}
#endif

	/// <summary>Add content of the file to the MD5</summary>
	static void addFile( this MD5 md5, string path )
	{
		using var stream = File.OpenRead( path );
		// This method is used to hash HLSL files, and they are rather small, 10kb max.
		// That's why instead of streaming we load complete files into rented arrays
		int length = (int)stream.Length;
		byte[] buffer = ArrayPool<byte>.Shared.Rent( length );
		try
		{
			stream.ReadExactly( buffer.AsSpan( 0, length ) );
			md5.TransformBlock( buffer, 0, length, null, 0 );
		}
		finally
		{
			ArrayPool<byte>.Shared.Return( buffer );
		}
	}

	/// <summary>Finalize the MD5 hash, and convert into <see cref="Guid" /></summary>
	static Guid hash( this MD5 md5 )
	{
		md5.TransformFinalBlock( Array.Empty<byte>(), 0, 0 );
		byte[] hash = md5.Hash ?? throw new ApplicationException();
		return new Guid( hash );
	}

	/// <summary>Compute MD5 hash of all shaders; the array is expected to be sorted already.</summary>
	/// <remarks>The hash includes both <c>*.hlsl</c> source codes, and compiled binaries</remarks>
	public static Guid compute( IEnumerable<sShaderBinary> binaries )
	{
		using MD5 md5 = MD5.Create();
		foreach( sShaderBinary bin in binaries )
		{
			// We want to include both HLSL and DXBC there
			// DXBC may change without HLSL with project settings, or changes in *.hlsli include files
			// HLSL may change without DXBC with comments, and we translate some of the comments from HLSL to C#
			md5.addBytes( bin.dxbc );
			md5.addFile( bin.sourcePath );
		}
		return md5.hash();
	}

	static string hashFilePath =>
		Path.Combine( Program.inputs.temp, "packedShadersHash.bin" );

	/// <summary><c>true</c> when the hash matches the stored value</summary>
	public static bool isCurrent( in Guid inputHash )
	{
		string path = hashFilePath;
		if( !File.Exists( path ) )
			return false;

		byte[] arr = File.ReadAllBytes( path );
		if( arr.Length != 16 )
			return false;

		Guid storedHash = new Guid( arr );
		return storedHash == inputHash;
	}

	/// <summary>Store new hash to disk</summary>
	public static void store( in Guid inputHash ) =>
		File.WriteAllBytes( hashFilePath, inputHash.ToByteArray() );
}