namespace Cgml;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>Miscellaneous utilities</summary>
public static class MiscUtils
{
	/// <summary>Size in bytes used by tensor elements</summary>
	public static int elementSize( this eDataType dt ) => dt switch
	{
		eDataType.FP16 => 2,
		eDataType.FP32 => 4,
		eDataType.U32 => 4,
		eDataType.BF16 => 2,
		_ => throw new ArgumentException()
	};

	readonly struct PrintedSize
	{
		public readonly double size;
		public readonly string unit;

		public PrintedSize( long bytes )
		{
			Debug.Assert( bytes >= 0 );
			if( bytes < 1 << 10 )
			{
				size = bytes;
				unit = "bytes";
			}
			else if( bytes < 1 << 20 )
			{
				const double mul = 1.0 / ( 1 << 10 );
				size = mul * bytes;
				unit = "kb";
			}
			else if( bytes < 1 << 30 )
			{
				const double mul = 1.0 / ( 1 << 20 );
				size = mul * bytes;
				unit = "MB";
			}
			else
			{
				const double mul = 1.0 / ( 1 << 30 );
				size = mul * bytes;
				unit = "GB";
			}
		}
	}

	/// <summary>A string with printed memory use</summary>
	public static string printMemoryUse( Vector128<long> vec )
	{
		PrintedSize ram = new PrintedSize( vec.ToScalar() );
		PrintedSize vram = new PrintedSize( vec.GetElement( 1 ) );
		return $"{ram.size:F1} {ram.unit} RAM, {vram.size:F1} {vram.unit} VRAM";
	}

	/// <summary>A string with printed memory use</summary>
	public static string printMemoryUse( long bytes )
	{
		PrintedSize ps = new PrintedSize( bytes );
		return $"{ps.size:F1} {ps.unit}";
	}

	/// <summary>Accumulate memory use of the tensor in the dictionary</summary>
	public static void addTensorMemory( this Dictionary<string, Vector128<long>> dict,
		iTensor? tensor,
		[CallerArgumentExpression( "tensor" )] string? tensorName = null )
	{
		string key = tensorName ?? string.Empty;
		Vector128<long> v = tensor.getMemoryUse();
		if( dict.TryGetValue( key, out var acc ) )
		{
			acc = Sse2.Add( acc, v );
			dict[ key ] = acc;
		}
		else
			dict[ key ] = v;
	}

	/// <summary>Accumulate memory use of the tensor in the dictionary</summary>
	public static void addTensorMemory( this Dictionary<string, Vector128<long>> dict,
	Tensor? tensor,
	[CallerArgumentExpression( "tensor" )] string? tensorName = null ) =>
		dict.addTensorMemory( tensor?.native, tensorName );

	static IEnumerable<(long, string)> vram( this Dictionary<string, Vector128<long>> dict )
	{
		foreach( var kvp in dict )
		{
			long vramBytes = kvp.Value.GetElement( 1 );
			if( 0 == vramBytes )
				continue;
			yield return (vramBytes, kvp.Key);
		}
	}

	/// <summary>Summarize VRAM usage statistics</summary>
	public static IEnumerable<string> printVramStats( this Dictionary<string, Vector128<long>> dict )
	{
		foreach( var p in dict.vram().OrderByDescending( v => v.Item1 ) )
		{
			PrintedSize vram = new PrintedSize( p.Item1 );
			yield return $"{p.Item2}\t{vram.size:F1} {vram.unit} VRAM";
		}
	}
}