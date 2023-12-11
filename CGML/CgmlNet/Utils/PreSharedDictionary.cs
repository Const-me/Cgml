namespace Cgml;
using System.Diagnostics.CodeAnalysis;
using System.Xml;

/// <summary>Pre-shared dictionary for Microsoft binary XML serializer</summary>
/// <seealso href="https://learn.microsoft.com/en-us/openspecs/windows_protocols/mc-nbfx/94c66ea1-e79a-4364-af88-1fa7fef2cc33" />
sealed class PreSharedDictionary: IXmlDictionary
{
	public PreSharedDictionary( params string[] entries )
	{
		vals = new XmlDictionaryString[ entries.Length ];
		dict = new Dictionary<string, XmlDictionaryString>( entries.Length );
		for( int i = 0; i < entries.Length; i++ )
		{
			string str = entries[ i ];
			XmlDictionaryString e = new XmlDictionaryString( this, str, i );
			dict.Add( str, e );
			vals[ i ] = e;
		}
	}

	readonly XmlDictionaryString[] vals;
	readonly Dictionary<string, XmlDictionaryString> dict;

	bool IXmlDictionary.TryLookup( int key, [NotNullWhen( true )] out XmlDictionaryString? result )
	{
		if( key >= 0 && key < vals.Length )
		{
			result = vals[ key ];
			return true;
		}
		result = null;
		return false;
	}

	bool IXmlDictionary.TryLookup( string value, [NotNullWhen( true )] out XmlDictionaryString? result ) =>
		dict.TryGetValue( value, out result );

	bool IXmlDictionary.TryLookup( XmlDictionaryString value, [NotNullWhen( true )] out XmlDictionaryString? result ) =>
		dict.TryGetValue( value.Value, out result );
}