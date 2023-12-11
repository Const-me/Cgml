#pragma warning disable CS0649 // Field is never assigned to
namespace Cgml;
using System.Diagnostics;
using System.Runtime.InteropServices;

/// <summary>Type of the profiler measure</summary>
public enum eProfilerMeasure: byte
{
	/// <summary>Block</summary>
	Block = 1,
	/// <summary>Compute shader</summary>
	Shader = 2,
}

/// <summary>Individual measure of the profiler</summary>
public readonly struct ProfilerMeasure
{
	/// <summary>Count of measures</summary>
	public readonly long count;
	readonly long m_time;
	readonly long m_maximum;

	/// <summary>Total time taken</summary>
	public TimeSpan total => TimeSpan.FromTicks( m_time );

	/// <summary>Average time per measure</summary>
	public TimeSpan average
	{
		get
		{
			Debug.Assert( count > 0 );
			if( 1 == count )
				return total;
			else
				return TimeSpan.FromTicks( m_time / count );
		}
	}

	/// <summary>Maximum time measured</summary>
	public TimeSpan maximum => TimeSpan.FromTicks( m_maximum );
}

/// <summary>Result entry for profiler</summary>
public readonly struct ProfilerResult
{
	/// <summary>What's being measured</summary>
	public readonly eProfilerMeasure what;
	/// <summary>For blocks, the integer passed to <see cref="iContext.profilerBlockStart" /><br/>
	/// For shaders, 0-based ID passed to <see cref="iContext.bindShader" />
	/// </summary>
	public readonly ushort id;

	/// <summary>Result of the measure</summary>
	public readonly ProfilerMeasure result;
}

/// <summary>Function pointer to receive profiler data from C++</summary>
[UnmanagedFunctionPointer( CallingConvention.StdCall )]
public delegate int pfnProfilerDataUnsafe(
	[In, MarshalAs( UnmanagedType.LPArray, SizeParamIndex = 1 )] ProfilerResult[] arr,
	int length, IntPtr pv );

/// <summary>RAII helper to implement profiler blocks</summary>
public ref struct ProfilerBlock
{
	readonly iContext context;
	internal ProfilerBlock( iContext context, ushort id )
	{
		this.context = context;
		context.profilerBlockStart( id );
	}

	/// <summary>End the block</summary>
	public void Dispose()
	{
		context.profilerBlockEnd();
	}
}

/// <summary>Utility structure for human-readable time</summary>
public readonly struct PrintedTime
{
	/// <summary>Number</summary>
	public readonly double value;

	/// <summary>Unit</summary>
	public readonly string unit;

	PrintedTime( in TimeSpan ts )
	{
		if( ts.Ticks >= TimeSpan.TicksPerSecond )
		{
			value = ts.TotalSeconds;
			unit = "seconds";
		}
		else if( ts.Ticks >= TimeSpan.TicksPerMillisecond )
		{
			value = ts.TotalMilliseconds;
			unit = "milliseconds";
		}
		else
		{
			value = ts.Ticks / 10;
			unit = "microseconds";
		}
	}

	/// <summary>Convert from TimeSpan</summary>
	public static implicit operator PrintedTime( TimeSpan ts ) =>
		new PrintedTime( ts );

	/// <summary>Convert to string</summary>
	public override string ToString() =>
		$"{value} {unit}";
}

/// <summary>Utility class to convert profiler result to text</summary>
public static class ProfilerExt
{
	static string format( in ProfilerMeasure res, string name )
	{
		PrintedTime total = res.total;
		if( 1 == res.count )
			return $"{name}\t{total.value} {total.unit}";
		else
		{
			PrintedTime avg = res.average;
			PrintedTime max = res.maximum;
			return $"{name}\t{total.value} {total.unit}, {res.count} calls, {avg.value} {avg.unit} average, {max.value} {max.unit} maximum";
		}
	}

	/// <summary>Generate human-readable text from profiler results</summary>
	public static IEnumerable<string> formatted( this ProfilerResult[]? arr, Func<ushort, string> block, Func<ushort, string> shader )
	{
		if( null == arr )
		{
			yield return "No profiler data";
			yield break;
		}

		foreach( var group in arr.GroupBy( x => x.what ) )
		{
			Func<ushort, string> pfnName;
			switch( group.Key )
			{
				case eProfilerMeasure.Block:
					yield return "\tBlocks";
					pfnName = block;
					break;
				case eProfilerMeasure.Shader:
					yield return "\tCompute Shaders";
					pfnName = shader;
					break;
				default:
					throw new ApplicationException();
			}

			foreach( ProfilerResult res in group )
			{
				string name = pfnName( res.id );
				yield return format( res.result, name );
			}
		}
	}
}