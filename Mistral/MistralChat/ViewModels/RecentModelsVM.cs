namespace MistralChat.ViewModels;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Windows;
using System.Windows.Input;

public sealed class RecentModelsVM
{
	public sealed class Item: ICommand
	{
		readonly string label;
		internal readonly string path;
		readonly RecentModelsVM owner;

		internal Item( RecentModelsVM owner, string path )
		{
			this.owner = owner;
			label = path;
			this.path = path;
		}
		public override string ToString() => label;

		void ICommand.Execute( object? parameter ) =>
			owner.clicked?.Invoke( path );
		bool ICommand.CanExecute( object? parameter ) => true;
		event EventHandler? ICommand.CanExecuteChanged
		{
			add { }
			remove { }
		}
	}

	public RecentModelsVM()
	{
		items = new ObservableCollection<Item>();

		string[]? arr = Preferences.recentModels;
		if( null != arr )
			setItems( arr );
	}

	public ObservableCollection<Item> items { get; } = new ObservableCollection<Item>();

	void setItems( string[] arr )
	{
		items.Clear();
		foreach( var path in arr )
			items.Add( new Item( this, path ) );
	}

	void addImpl( string path )
	{
		List<string> result = new List<string>( items.Count + 1 );
		result.Add( path );

		foreach( var i in items )
		{
			if( i.path.Equals( path, StringComparison.OrdinalIgnoreCase ) )
				continue;
			result.Add( i.path );
		}

		// Save to registry
		string[] arr = result.ToArray();
		Preferences.recentModels = arr;
		// Update the GUI
		setItems( arr );
	}

	/// <summary>Add entry to the recent files list<br/>
	/// If it was already there, move to the top position in the list.</summary>
	/// <remarks>May be called form any thread</remarks>
	internal void add( string path )
	{
		var app = Application.Current;
		if( false == app?.CheckAccess() )
			app.Dispatcher.BeginInvoke( () => addImpl( path ) );
		else
			addImpl( path );
	}

	/// <summary>Remove entry from the recent file list</summary>
	/// <remarks>Must be called from the main thread</remarks>
	internal void remove( string path )
	{
		List<string> result = new List<string>( items.Count );
		foreach( var i in items )
		{
			if( i.path.Equals( path, StringComparison.OrdinalIgnoreCase ) )
				continue;
			result.Add( i.path );
		}

		string[] arr = result.ToArray();
		Preferences.recentModels = arr;
		setItems( arr );
	}

	public Action<string>? clicked;
}