/* part of Pickle, by Irmen de Jong (irmen@razorvine.net) */

namespace Razorvine.Pickle
{
	/// <summary>
	/// Exception thrown when the unpickler encountered an unknown or unimplemented opcode.
	/// </summary>
	public sealed class InvalidOpcodeException : PickleException
	{
		public InvalidOpcodeException(string message) : base(message)
		{
		}
	}
}