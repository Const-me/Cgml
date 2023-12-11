namespace Mistral;
using Cgml;
using Mistral.Model;

/// <summary>GPU profiler output data</summary>
public sealed class ProfilerData
{
	/// <summary>A single entry from the profiler</summary>
	public readonly struct Entry
	{
		/// <summary>Name of the entry</summary>
		public readonly string name;

		/// <summary>Time counter</summary>
		public readonly ProfilerMeasure result;

		internal Entry( string name, in ProfilerMeasure result )
		{
			this.name = name;
			this.result = result;
		}
	}

	/// <summary>Blocs</summary>
	public Entry[] blocks;

	/// <summary>Compute shader, sorted descending by total time</summary>
	public Entry[] shaders;

	static Entry makeBlock( ProfilerResult res )
	{
		eProfilerBlock b = (eProfilerBlock)res.id;
		return new Entry( b.ToString(), res.result );
	}

	static Entry makeShader( ProfilerResult res )
	{
		eShader s = (eShader)res.id;
		return new Entry( s.ToString(), res.result );
	}

	internal ProfilerData( ProfilerResult[] arr )
	{
		blocks = arr.Where( i => i.what == eProfilerMeasure.Block )
			.Select( makeBlock ).ToArray();

		shaders = arr.Where( i => i.what == eProfilerMeasure.Shader )
			.Select( makeShader ).ToArray();
	}
}