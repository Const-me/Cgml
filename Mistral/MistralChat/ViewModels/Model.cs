namespace MistralChat.ViewModels;
using Mistral;
using System;

sealed class Model: IDisposable
{
	public readonly iModel model;
	string desc;
	readonly string? tokenizer;
	Model( iModel model, string desc, string? tokenizer = null )
	{
		this.model = model;
		this.desc = desc;
		this.tokenizer = tokenizer;

		SamplingParams? sp = Preferences.tryLoadSampling();
		if( null != sp )
			model.samplingParams = sp;

		model.performanceParams.isFastGpu = Preferences.gpuHighPerformance;
	}

	public override string ToString() => desc;

	public void Dispose() =>
		model.Dispose();

	public static Model load( string cgml, Action<double> pfnProgress )
	{
		iModel model = ModelLoader.load( cgml, Preferences.deviceParameters, pfnProgress );
		string desc = "CGML model: " + model.description();
		return new Model( model, desc );
	}

	public static Model importTorch( in TorchSource source )
	{
		iModel model = ModelLoader.importTorch( source, Preferences.deviceParameters );
		string desc = "Imported model: " + model.description();
		return new Model( model, desc, source.tokenizer );
	}

	public bool canSave => null != tokenizer;

	public void Save( string cgml, Action<double> pfnProgress )
	{
		if( null == tokenizer )
			throw new ApplicationException( "CGML models don’t support saving" );

		ModelLoader.save( model, tokenizer, cgml, pfnProgress );
		desc = "CGML model: " + model.description();
	}
}