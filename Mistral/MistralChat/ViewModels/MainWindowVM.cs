namespace MistralChat.ViewModels;

using Mistral;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Windows;

/// <summary>View model class for the main window</summary>
public class MainWindowVM: PropertyChangedBase
{
	const string strNoModel = "No model is loaded";
	public string modelInfo { get; private set; } = strNoModel;
	public string modelTempInfo { get; private set; } = string.Empty;

	public Visibility visProgress { get; private set; } = Visibility.Collapsed;
	public bool guiEnabled { get; private set; } = true;
	public bool chatEnabled { get; private set; } = false;
	public string progressMessage { get; private set; } = "";

	public bool progressIndeterminate { get; private set; } = true;
	public double progressValue { get; private set; } = 0;

	public MainWindowVM()
	{
		if( MiscUtils.isDesignTime )
		{
			visProgress = Visibility.Visible;
			progressMessage = "Loading, please wait…";
		}
		chat = new ChatVM( this );
		recentModels.clicked = openRecentModel;
	}

	[Flags]
	enum eProgressFlags: byte
	{
		None = 0,
		ReportProgress = 1,
		ThrowExceptions = 2,
	}

	async Task withProgress( Action act, eProgressFlags flags, string what, string? failMessage )
	{
		visProgress = Visibility.Visible;
		guiEnabled = false;
		progressMessage = what;
		progressValue = 0;
		progressIndeterminate = !flags.HasFlag( eProgressFlags.ReportProgress );
		Refresh();

		Exception? ex = null;
		try
		{
			await Task.Run( act );
		}
		catch( Exception e )
		{
			if( flags.HasFlag( eProgressFlags.ThrowExceptions ) )
				throw;
			ex = e;
		}
		finally
		{
			visProgress = Visibility.Collapsed;
			guiEnabled = true;
			progressMessage = "";
			Refresh();
		}

		if( null == ex )
			return;

		var wnd = Application.Current.MainWindow;
		string msg = $"{failMessage}\n{ex.Message}";
		MessageBox.Show( wnd, msg, "Operation Failed", MessageBoxButton.OK, MessageBoxImage.Warning );
	}

	PropertyChangedEventArgs? progressUpdateMessage;

	void updateProgress( double p )
	{
		progressValue = p;
		progressUpdateMessage ??= new PropertyChangedEventArgs( nameof( progressValue ) );
		Notify( progressUpdateMessage );
	}

	public bool hasModel => null != model;
	public bool canSaveModel => true == model?.canSave;

	internal Model? model { get; private set; }

	void resetModel()
	{
		if( null != model )
		{
			model.Dispose();
			model = null;
			modelInfo = strNoModel;
			NotifyOfPropertyChange( nameof( modelInfo ) );
			modelTempInfo = string.Empty;
			NotifyOfPropertyChange( nameof( modelTempInfo ) );
			chat.setModelAvailable( false );
		}
		if( chatEnabled )
		{
			chatEnabled = false;
			NotifyOfPropertyChange( nameof( chatEnabled ) );
		}
	}

	void loadModelImpl( string cgml )
	{
		resetModel();

		model = Model.load( cgml, updateProgress );
		modelInfo = model.ToString();
		NotifyOfPropertyChange( nameof( modelInfo ) );
		chatEnabled = true;
		NotifyOfPropertyChange( nameof( chatEnabled ) );
		chat.setModelAvailable( true );

		recentModels.add( cgml );
	}

	void importModelImpl( TorchSource source )
	{
		resetModel();

		model = Model.importTorch( source );
		modelInfo = model.ToString();
		NotifyOfPropertyChange( nameof( modelInfo ) );
		chatEnabled = true;
		NotifyOfPropertyChange( nameof( chatEnabled ) );
		chat.setModelAvailable( true );
	}

	const string loadModelError = "Unable to load the model";
	public async void loadModel( string cgml )
	{
		Action act = () => loadModelImpl( cgml );
		await withProgress( act, eProgressFlags.ReportProgress, "Loading model, please wait…", loadModelError );
	}

	public async void importModel( TorchSource source )
	{
		Action act = () => importModelImpl( source );
		await withProgress( act, eProgressFlags.None, "Importing model, please wait…", "Unable to import the model" );
	}

	public async void saveModel( string cgml )
	{
		Action act = () =>
		{
			if( null == model )
				throw new ApplicationException( "No current model" );
			model.Save( cgml, updateProgress );
			modelInfo = model.ToString();
			NotifyOfPropertyChange( nameof( modelInfo ) );

			recentModels.add( cgml );
		};
		await withProgress( act, eProgressFlags.ReportProgress, "Saving the model, please wait…", "Unable to save the model" );
	}

	readonly ChatVM chat;
	public iChatVM chatVm => chat;

	internal readonly struct RestoreMenu: IDisposable
	{
		readonly MainWindowVM vm;

		public RestoreMenu( MainWindowVM vm )
		{
			this.vm = vm;
		}

		public void Dispose()
		{
			vm.guiEnabled = true;
			vm.NotifyOfPropertyChange( nameof( guiEnabled ) );
		}
	}

	internal RestoreMenu disableMenu()
	{
		Debug.Assert( guiEnabled );
		guiEnabled = false;
		NotifyOfPropertyChange( nameof( guiEnabled ) );
		return new RestoreMenu( this );
	}

	public RecentModelsVM recentModels { get; } = new RecentModelsVM();

	async void openRecentModel( string cgml )
	{
		Action act = () => loadModelImpl( cgml );
		try
		{
			const eProgressFlags flags = eProgressFlags.ReportProgress | eProgressFlags.ThrowExceptions;
			await withProgress( act, flags, "Loading model, please wait…", null );
		}
		catch( Exception ex )
		{
			string msg = $"{loadModelError}\n{ex.Message}\n\nRemove the entry from recent files?";
			var wnd = Application.Current.MainWindow;
			var res = MessageBox.Show( wnd, msg, "Operation Failed", MessageBoxButton.YesNo, MessageBoxImage.Warning );
			if( res != MessageBoxResult.Yes )
				return;

			recentModels.remove( cgml );
		}
	}

	bool m_autoLoadModels = Preferences.loadLastModel;
	public bool autoLoadModels
	{
		get => m_autoLoadModels;
		set => Preferences.loadLastModel = m_autoLoadModels = value;
	}

	bool m_disableRandomness = Preferences.disableRandomness;
	public bool disableRandomness
	{
		get => m_disableRandomness;
		set => Preferences.disableRandomness = m_disableRandomness = value;
	}

	internal void windowLoaded()
	{
		if( autoLoadModels && recentModels.items.Count > 0 )
			openRecentModel( recentModels.items[ 0 ].path );
	}

	internal eApplicationMode mode { get; private set; } = eApplicationMode.Chat;

	public string windowTitle => mode switch
	{
		eApplicationMode.Chat => "Offline Chat",
		eApplicationMode.TextGenerator => "Text Generator",
		_ => throw new NotImplementedException()
	};

	public string menuToggleMode => mode switch
	{
		eApplicationMode.Chat => "Text Generator",
		eApplicationMode.TextGenerator => "Offline Chat",
		_ => throw new NotImplementedException()
	};

	public void toggleMode()
	{
		mode = mode switch
		{
			eApplicationMode.Chat => eApplicationMode.TextGenerator,
			eApplicationMode.TextGenerator => eApplicationMode.Chat,
			_ => throw new NotImplementedException()
		};
		chat.switchedMode();
		Refresh();
	}

	internal void updateMemoryLabel()
	{
		string str = string.Empty;
		if( model != null )
		{
			model.model.getVideoMemoryUsage( out long kvBytes, out long tempBytes );
			string kv = Cgml.MiscUtils.printMemoryUse( kvBytes );
			string temp = Cgml.MiscUtils.printMemoryUse( tempBytes );
			str = $"KV {kv}, temps {temp}";
		}

		modelTempInfo = str;
		NotifyOfPropertyChange( nameof( modelTempInfo ) );
	}

	internal void generationOptions( Window parent )
	{
		if( null == model )
		{
			MessageBox.Show( parent, "Please load a model first", "Generation Options",
				MessageBoxButton.OK, MessageBoxImage.Information );
			return;
		}

		if( !( chatEnabled && guiEnabled ) )
		{
			MessageBox.Show( parent, "Can’t adjust options while the model is busy", "Generation Options",
				MessageBoxButton.OK, MessageBoxImage.Information );
			return;
		}

		GeneratorOptions popup = new GeneratorOptions( model.model )
		{
			Owner = parent
		};
		popup.ShowDialog();
	}

	public GpuSelectionVM gpuSelection { get; } = new GpuSelectionVM();
}