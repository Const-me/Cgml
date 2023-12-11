namespace MistralChat;
using Microsoft.Win32;
using Mistral;
using MistralChat.ViewModels;
using System.Windows;

/// <summary>Interaction logic for MainWindow.xaml</summary>
public partial class MainWindow: Window
{
	readonly MainWindowVM vm;
	public MainWindow()
	{
		InitializeComponent();
		vm = new MainWindowVM();
		DataContext = vm;
	}

	void exit_click( object sender, RoutedEventArgs e ) => Close();

	void open_click( object sender, RoutedEventArgs e )
	{
		OpenFileDialog? ofd = new OpenFileDialog()
		{
			Title = "Open Model File",
			Filter = "CGML models (*.cgml)|*.cgml"
		};
		if( true != ofd.ShowDialog( this ) )
			return;

		string cgml = ofd.FileName;
		ofd = null;
		vm.loadModel( cgml );
	}

	void save_click( object sender, RoutedEventArgs e )
	{
		if( !vm.hasModel )
		{
			MessageBox.Show( this, "Please import a model first",
				"No model to save", MessageBoxButton.OK, MessageBoxImage.Information );
			return;
		}

		if( !vm.canSaveModel )
		{
			MessageBox.Show( this, "The current model is already in CGML format.\nUse Windows Explorer to copy the file instead.",
				"Saving is useless", MessageBoxButton.OK, MessageBoxImage.Information );
			return;
		}

		SaveFileDialog? sfd = new SaveFileDialog()
		{
			Title = "Save Model",
			Filter = "CGML models (*.cgml)|*.cgml",
			CreatePrompt = false,
			OverwritePrompt = true,
		};

		if( true != sfd.ShowDialog( this ) )
			return;

		string cgml = sfd.FileName;
		sfd = null;
		vm.saveModel( cgml );
	}

	void import_click( object sender, RoutedEventArgs e )
	{
		ImportDialog? dialog = new ImportDialog() { Owner = this };
		if( true != dialog.ShowDialog() || !dialog.result.HasValue )
			return;

		TorchSource source = dialog.result.Value;
		dialog = null;
		vm.importModel( source );
	}

	void window_Loaded( object sender, RoutedEventArgs e ) =>
		vm.windowLoaded();

	void profiler_Click( object sender, RoutedEventArgs e )
	{
		iModel? model = vm.model?.model;
		if( null == model )
		{
			MessageBox.Show( this, "Please import a model first",
				"No model", MessageBoxButton.OK, MessageBoxImage.Information );
			return;
		}

		ProfilerData? obj = model.profilerData();
		if( null == obj )
		{
			MessageBox.Show( this, "Please generate some mesages first",
				"No model", MessageBoxButton.OK, MessageBoxImage.Information );
			return;
		}

		ProfilerOutputWindow wnd = new ProfilerOutputWindow( obj );
		wnd.ShowDialog();
	}

	void toggleMode_Click( object sender, RoutedEventArgs e ) =>
		vm.toggleMode();

	void generationOptions_click( object sender, RoutedEventArgs e ) =>
		vm.generationOptions( this );

	void about_Click( object sender, RoutedEventArgs e )
	{
		var dialog = new AboutDialog()
		{
			Owner = this
		};
		dialog.ShowDialog();
	}
}