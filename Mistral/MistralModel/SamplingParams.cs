namespace Mistral;

/// <summary>Parameters for controlling the sampling behavior during the inference</summary>
/// <seealso cref="iModel.samplingParams" />
public record class SamplingParams
{
	/// <summary>Temperature parameter controls the randomness of the sampling. Higher values increase randomness</summary>
	public float temperature { get; }

	/// <summary>Top-P parameter controls the diversity of the sampling by excluding low-probability tokens</summary>
	public float topP { get; }

	/// <summary>Create from two numbers</summary>
	public SamplingParams( float temperature, float topP )
	{
		if( temperature <= 0 || topP <= 0 )
			throw new ArgumentOutOfRangeException();

		this.temperature = temperature;
		this.topP = topP;
	}

	/// <summary>Default parameters, the values are from <c>main.py</c> source file</summary>
	public static SamplingParams makeDefault() =>
		new SamplingParams( 0.7f, 0.8f );
}