namespace MistralChat.ViewModels;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;

/// <summary>A base class that implements the infrastructure for property change notification,<br />
/// and automatically performs UI thread marshaling.</summary>
public class PropertyChangedBase: INotifyPropertyChanged
{
	public event PropertyChangedEventHandler? PropertyChanged;

	public void NotifyOfPropertyChange( [CallerMemberName] string? propertyName = null )
	{
		if( PropertyChanged != null )
		{
			var app = Application.Current;
			if( false == app?.CheckAccess() )
				app.Dispatcher.BeginInvoke( () => OnPropertyChanged( new PropertyChangedEventArgs( propertyName ) ) );
			else
				OnPropertyChanged( new PropertyChangedEventArgs( propertyName ) );
		}
	}

	public void Notify( PropertyChangedEventArgs pcea )
	{
		if( PropertyChanged != null )
		{
			var app = Application.Current;
			if( false == app?.CheckAccess() )
				app.Dispatcher.BeginInvoke( () => OnPropertyChanged( pcea ) );
			else
				OnPropertyChanged( pcea );
		}
	}

	/// <summary>Raises the <see cref="PropertyChanged" /> event directly.</summary>
	/// <param name="e">The <see cref="PropertyChangedEventArgs"/> instance containing the event data.</param>
	[EditorBrowsable( EditorBrowsableState.Never )]
	protected void OnPropertyChanged( PropertyChangedEventArgs e )
	{
		PropertyChanged?.Invoke( this, e );
	}

	public void Refresh() =>
		NotifyOfPropertyChange( null );
}