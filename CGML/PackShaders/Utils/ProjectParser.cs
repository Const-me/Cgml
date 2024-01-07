namespace PackShaders;
using System.Xml.Linq;
using System.Xml;
using System.Xml.XPath;

/// <summary>Parser for <c>.vcxproj</c> XML files</summary>
/// <remarks>The implementation is very simple because relies on XML DOM and XPath from the standard library</remarks>
static class ProjectParser
{
	/// <summary>Find all <c>&lt;FxCompile Include="..." /&gt;</c> XML elements, generate a sequence of the <c>...</c> strings</summary>
	public static IEnumerable<string> parse( string path )
	{
		XDocument doc = XDocument.Load( path );

		XmlNamespaceManager namespaceManager = new XmlNamespaceManager( new NameTable() );
		namespaceManager.AddNamespace( "ns", "http://schemas.microsoft.com/developer/msbuild/2003" );

		const string xpathExpression = "//ns:FxCompile[@Include]";
		foreach( XElement elt in doc.XPathSelectElements( xpathExpression, namespaceManager ) )
		{
			string name = ( elt.Attribute( "Include" )?.Value ) ?? throw new ApplicationException();
			yield return name;
		}
	}
}