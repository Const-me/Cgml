﻿using K4os.Compression.LZ4;

namespace ImageShadersPack;

sealed class BinaryPackage
{
	// compression speed drops rapidly when not using FAST mode, while decompression speed stays the same
	// Actually, it is usually faster for high compression levels as there is less data to process
	// https://github.com/MiloszKrajewski/K4os.Compression.LZ4#compression-levels
	const LZ4Level compressionLevel = LZ4Level.L12_MAX;

	readonly string[] names = new string[]
	{
		"FullScreenTriangleVS",
		"UpcastTexturePS",
		"SamplePS",
		"CopyAndNormalizeCS",
	};

	readonly byte[] compressed;
	readonly int uncompressedLength;
	readonly int[] offsets;

	public BinaryPackage( string dirBinaries )
	{
		List<int> offsets = new List<int>();
		MemoryStream ms = new MemoryStream();

		int off = 0;
		foreach( string i in names )
		{
			offsets.Add( off );
			string name = $"{i}.cso";
			string path = Path.Combine( dirBinaries, name );
			byte[] bytes = File.ReadAllBytes( path );
			ms.Write( bytes, 0, bytes.Length );
			off += bytes.Length;
		}
		offsets.Add( off );

		this.offsets = offsets.ToArray();

		byte[] dxbc = ms.ToArray();
		uncompressedLength = dxbc.Length;
		int maxLength = LZ4Codec.MaximumOutputSize( dxbc.Length );
		byte[] output = new byte[ maxLength ];
		int cb = LZ4Codec.Encode( dxbc, output, compressionLevel );
		if( cb <= 0 )
			throw new ApplicationException( $"LZ4Codec.Encode failed with status {cb}" );
		Array.Resize( ref output, cb );
		compressed = output;
		Console.WriteLine( "Compressed {0} image processing shaders, {1:F1} kb -> {2:F1} kb",
			names.Length, uncompressedLength / 1024.0, compressed.Length / 1024.0 );
	}

	public void generate( string dir, string name )
	{
		Directory.CreateDirectory( dir );
		string path = Path.Combine( dir, name );
		using StreamWriter stream = File.CreateText( path );

		stream.Write( @"// This source file is generated by a tool

// This array contains concatenated and compressed DXBC binaries for the compiled shaders
static const std::array<uint8_t, {0}> s_shadersCompressed =
{{", compressed.Length );

		for( int i = 0; i < compressed.Length; i++ )
		{
			if( 0 == i % 16 )
				stream.Write( "\r\n\t" );
			else
				stream.Write( ' ' );
			stream.Write( "0x{0:X02},", compressed[ i ] );
		}

		stream.Write( @"
}};

// Uncompressed length of the above array
constexpr int shadersUncompressedLength = {0};

// [ start, length ] of individual shaders in the uncompressed blob", uncompressedLength );

		for( int i = 0; i < names.Length; i++ )
		{
			string shader = names[ i ];
			int start = offsets[ i ];
			int length = offsets[ i + 1 ] - start;

			stream.Write( @"
inline std::pair<uint32_t,uint32_t> span{0}() {{ return {{ {1}, {2} }}; }}", shader, start, length );
		}
	}
}