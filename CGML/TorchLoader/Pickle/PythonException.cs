/* part of Pickle, by Irmen de Jong (irmen@razorvine.net) */

using System.Collections;
using System.Diagnostics.CodeAnalysis;
using System.Text;

namespace Razorvine.Pickle
{
	/// <summary>Exception thrown that represents a certain Python exception.</summary>
	[SuppressMessage("ReSharper", "InconsistentNaming")]
	public sealed class PythonException : Exception
	{
		public string _pyroTraceback {get; set;}
		public string PythonExceptionType {get; set;}

		public PythonException(string message) : base(message)
		{
		}

		// special constructor for UnicodeDecodeError
		// ReSharper disable once UnusedMember.Global
		public PythonException(string encoding, byte[] data, int i1, int i2, string message)
			:base("UnicodeDecodeError: "+encoding+": "+message)
		{
		}

		/// <summary>
		/// for the unpickler to restore state
		/// </summary>
		// ReSharper disable once UnusedMember.Global
		public void __setstate__(Hashtable values) {
			if(!values.ContainsKey("_pyroTraceback"))
				return;
			object tb=values["_pyroTraceback"];
			// if the traceback is a list of strings, create one string from it
			var tbcoll = tb as ICollection;
			if(tbcoll != null) {
				StringBuilder sb=new StringBuilder();
				foreach(object line in tbcoll) {
					sb.Append(line);
				}	
				_pyroTraceback=sb.ToString();
			} else {
				_pyroTraceback=(string)tb;
			}
			//Console.WriteLine("pythonexception state set to:{0}",_pyroTraceback);
		}
	}
}