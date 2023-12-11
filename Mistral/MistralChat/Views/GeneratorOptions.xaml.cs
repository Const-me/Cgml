#pragma warning disable CS8618 // Non-nullable field 'model' must contain a non-null value when exiting constructor.
namespace MistralChat;
using Mistral;
using System.Globalization;
using System.Windows;
using System.Windows.Controls;

/// <summary>Dialog box to adjust some basic generator settings</summary>
public partial class GeneratorOptions: Window
{
	/// <summary>This constructor is only called by UI designer</summary>
	public GeneratorOptions()
	{
		InitializeComponent();
	}

	readonly iModel model;

	/// <summary>This constructor is for real</summary>
	internal GeneratorOptions( iModel model )
	{
		InitializeComponent();

		this.model = model;
		SamplingParams sp = model.samplingParams ?? SamplingParams.makeDefault();
		temperature.Text = printFloat( sp.temperature );
		topP.Text = printFloat( sp.topP );

		queueLengthInitial = Preferences.gpuQueueDepth;
		queueLengthSlider.Value = queueLengthInitial;
		setQueueLength( queueLengthInitial );

		powerSaverInitial = Preferences.gpuPowerSaver;
		gpuPowerSaver.IsChecked = powerSaverInitial;

		isFastGpu.IsChecked = model.performanceParams.isFastGpu;
	}

	static string printFloat( float x ) => x.ToString( CultureInfo.InvariantCulture );

	static float? parseFloat( string text )
	{
		if( float.TryParse( text, NumberStyles.Float, CultureInfo.InvariantCulture, out float result ) )
			return result;
		return null;
	}

	float? parseFloat( TextBox source, float min, float max, string field )
	{
		float? val = parseFloat( source.Text );
		if( !val.HasValue )
		{
			string message = $"Unable to parse the number for {field}";
			MessageBox.Show( this, message, "Incorrect Input", MessageBoxButton.OK, MessageBoxImage.Warning );
			return null;
		}

		if( val.Value < min || val.Value > max )
		{
			string message = $"The value for {field} is out of range.\nThe range is [ {min}, {max} ]";
			MessageBox.Show( this, message, "Incorrect Input", MessageBoxButton.OK, MessageBoxImage.Warning );
			return null;
		}

		return val;
	}

	void ok_click( object sender, RoutedEventArgs e )
	{
		float? temperature = parseFloat( this.temperature, 1E-3f, 1E+2f, "temperature" );
		float? topP = parseFloat( this.topP, 1E-3f, 1E+2f, "top P" );
		if( !( temperature.HasValue && topP.HasValue ) )
			return;

		SamplingParams sp = new SamplingParams( temperature.Value, topP.Value );
		// Apply to the model
		model.samplingParams = sp;
		// Save to registry
		Preferences.saveSampling( sp );

		bool gpuPrefsChanged = false;
		if( queueLengthInitial != queueLengthCurrent )
		{
			// User has changed the queue depth preference, save to registry
			Preferences.gpuQueueDepth = queueLengthCurrent;
			gpuPrefsChanged = true;
		}

		bool powerSave = gpuPowerSaver.IsChecked ?? false;
		if( powerSaverInitial != powerSave )
		{
			// User changed the power saving checkbox, save to registry
			Preferences.gpuPowerSaver = powerSave;
			gpuPrefsChanged = true;
		}

		if( gpuPrefsChanged )
		{
			// Let user know they gonna need to reload the model
			MessageBox.Show( this,
				"The new GPU preferences will be applied\nnext time you open or import a Mistral model.",
				"GPU Queue Depth",
				MessageBoxButton.OK, MessageBoxImage.Information );
		}

		bool fastGpu = isFastGpu.IsChecked ?? false;
		model.performanceParams.isFastGpu = fastGpu;
		Preferences.gpuHighPerformance = fastGpu;

		// Dismiss the dialog
		DialogResult = true;
	}

	readonly int queueLengthInitial;
	int queueLengthCurrent;

	void setQueueLength( int i )
	{
		queueLengthCurrent = i;
		queueLengthLabel.Text = $"GPU queue depth: {i}";
	}

	void slider_ValueChanged( object sender, RoutedPropertyChangedEventArgs<double> e )
	{
		int i = (int)Math.Round( e.NewValue );
		setQueueLength( i );
	}

	readonly bool powerSaverInitial;
}