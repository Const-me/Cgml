namespace MistralChat.ViewModels;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using System.Windows;

/// <summary>View model class for the chat panel</summary>
sealed class ChatVM: PropertyChangedBase, iChatVM
{
	readonly MainWindowVM owner;

	const string defaultPrompt = Generator.prompt;

	const string noModelWatermark = "Please load or import the language model";
	const string defaultWatermark = "Type a message here, Ctrl+Enter to submit";
	const string pendingWatermark = "Generating a response, please wait…";

	public ChatVM( MainWindowVM owner )
	{
		this.owner = owner;
		history = new ObservableCollection<iChatMessageVM>();
		watermark = noModelWatermark;
	}

	public ObservableCollection<iChatMessageVM> history { get; }
	// readonly Generator generator = new GeneratorComplete();
	readonly Generator generator = new GeneratorIncremental();

	public async Task generate( string text )
	{
		if( null == owner.model )
			return;

		generatorStats = noStats;
		NotifyOfPropertyChange( nameof( generatorStats ) );

		canRegenerate = false;
		NotifyOfPropertyChange( nameof( canRegenerate ) );

		history.Add( new ChatMessageVM( text, true ) );
		using var dm = owner.disableMenu();
		setWatermark( pendingWatermark );

		generator.disableRandomness = owner.disableRandomness;

		history.Add( generator.pendingMessage() );
		canCancel = true;
		NotifyOfPropertyChange( nameof( canCancel ) );
		try
		{
			if( owner.mode == eApplicationMode.Chat )
				text = await generator.generateChat( owner.model.model, initialPrompt, history );
			else
				text = await generator.generateText( owner.model.model, text );
		}
		finally
		{
			history.RemoveAt( history.Count - 1 );
			canCancel = false;
			NotifyOfPropertyChange( nameof( canCancel ) );
		}
		history.Add( new ChatMessageVM( text, false ) );
		setWatermark( defaultWatermark );

		generatorStats = generator.stats?.ToString() ?? noStats;
		NotifyOfPropertyChange( nameof( generatorStats ) );

		canRegenerate = true;
		NotifyOfPropertyChange( nameof( canRegenerate ) );

		owner.updateMemoryLabel();
	}

	public string initialPrompt { get; set; } = defaultPrompt;

	public string watermark { get; private set; }

	void setWatermark( string message )
	{
		watermark = message;
		NotifyOfPropertyChange( nameof( watermark ) );
	}

	public void setModelAvailable( bool available )
	{
		string message = available ? defaultWatermark : noModelWatermark;
		setWatermark( message );
	}

	const string noStats = "n/a";
	public string generatorStats { get; private set; } = noStats;

	public bool canRegenerate { get; private set; }

	public bool isDeterministic => owner.disableRandomness;

	public bool canCancel { get; private set; } = false;
	public void cancel()
	{
		generator.cancel();
		canCancel = false;
		NotifyOfPropertyChange( nameof( canCancel ) );
	}

	public Visibility visInitialPrompt =>
		( owner.mode == eApplicationMode.Chat ) ? Visibility.Visible : Visibility.Collapsed;

	public void switchedMode()
	{
		history.Clear();
		canRegenerate = false;
		generatorStats = noStats;
		Refresh();
	}
}