namespace MistralChat.ViewModels;
using System.Windows;
using System.Windows.Input;

/// <summary>View model for the Graphics adapter list, in the Options section of the main menu</summary>
public sealed class GpuSelectionVM
{
	/// <summary>One item of that list</summary>
	public sealed class Item: PropertyChangedBase, ICommand
	{
		/// <summary>Parent view model who owns this item</summary>
		readonly GpuSelectionVM owner;
		/// <summary>Text to render on the menu</summary>
		readonly string label;
		/// <summary>GPU ID to pass to the backend</summary>
		internal readonly string? gpu;

		internal Item( GpuSelectionVM owner, string label, string? gpu )
		{
			this.owner = owner;
			this.label = label;
			this.gpu = gpu;
		}
		public override string ToString() => label;

		/// <summary>True when this item is checked in that menu</summary>
		public bool isChecked { get; private set; }

		void ICommand.Execute( object? parameter ) =>
			owner.itemClicked( this );

		bool ICommand.CanExecute( object? parameter ) => true;
		event EventHandler? ICommand.CanExecuteChanged
		{
			add { }
			remove { }
		}

		internal void setChecked( bool val )
		{
			if( isChecked == val )
				return;
			isChecked = val;
			NotifyOfPropertyChange( nameof( isChecked ) );
		}
	}

	/// <summary>Available items to select</summary>
	public Item[] items { get; }

	/// <summary>Read current preference from registry, find and select the corresponding item</summary>
	void selectCurrent()
	{
		string? pref = Preferences.gpu;
		if( null == pref )
		{
			items[ 0 ].setChecked( true );
			return;
		}

		if( !int.TryParse( pref, out int idx ) )
		{
			for( int i = 1; i < items.Length; i++ )
			{
				if( items[ i ].gpu == pref )
				{
					items[ i ].setChecked( true );
					return;
				}
			}
		}
		else
		{
			if( idx < items.Length - 1 )
			{
				items[ idx + 1 ].setChecked( true );
				return;
			}
		}

		items[ 0 ].setChecked( true );
	}

	internal void itemClicked( Item selectedItem )
	{
		// Update the GUI
		foreach( Item i in items )
			i.setChecked( i == selectedItem );

		// Update the registry
		Preferences.gpu = selectedItem.gpu;

		// Report success to user
		MessageBox.Show( Application.Current.MainWindow,
			"The new preference will be applied\nnext time you open or import a Mistral model.",
			"Selected Graphics Adapter",
			MessageBoxButton.OK, MessageBoxImage.Information );
	}

	internal GpuSelectionVM()
	{
		// Enumerate GPUs with DXGI API deep inside the C++ backend
		// https://learn.microsoft.com/en-us/windows/win32/api/dxgi/nf-dxgi-idxgifactory1-enumadapters1
		string[] list = Mistral.ModelLoader.listGraphicAdapters();

		// Generate array of these items
		// Note we need some extra care to handle the case when user has multiple GPUs of the same model
		Dictionary<string, int> counters = new Dictionary<string, int>( list.Length );
		for( int i = 0; i < list.Length; i++ )
		{
			string str = list[ i ];
			if( counters.TryGetValue( str, out int counter ) )
			{
				counter++;
				counters[ str ] = counter;
			}
			else
				counters[ str ] = 1;
		}

		items = new Item[ list.Length + 1 ];
		items[ 0 ] = new Item( this, "Default adapter, recommended", null );
		for( int i = 0; i < list.Length; i++ )
		{
			string name = list[ i ];
			string key;
			if( counters[ name ] == 1 )
				key = name;
			else
				key = i.ToString();
			items[ i + 1 ] = new Item( this, name, key );
		}

		// Read current preference from registry, find and select the corresponding item
		selectCurrent();
	}
}