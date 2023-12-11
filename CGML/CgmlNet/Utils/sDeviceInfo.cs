#pragma warning disable CS0649  // Field is never assigned to
namespace Cgml;
using System.Runtime.InteropServices;

/// <summary>Vendor ID magic numbers</summary>
/// <remarks>They come from PCI-SIG database</remarks>
/// <seealso href="https://pcisig.com/membership/member-companies" />
public enum eGpuVendor: ushort
{
	/// <summary>AMD</summary>
	AMD = 0x1002,
	/// <summary>NVidia</summary>
	NVidia = 0x10de,
	/// <summary>Intel</summary>
	Intel = 0x8086,
	/// <summary>VMWare</summary>
	VMWare = 0x15ad,
};

/// <summary>Optional GPU features</summary>
[Flags]
public enum eOptionalFeatures: byte
{
	/// <summary>None of them</summary>
	None = 0,
	/// <summary>Basic FP64 support</summary>
	FP64Basic = 1,
	/// <summary>Advanced FP64 support: division and FMA instructions</summary>
	FP64Advanced = 2,
}

/// <summary>Information about the D3D device</summary>
public readonly struct sDeviceInfo
{
	readonly IntPtr m_name;
	/// <summary>Name of the device</summary>
	public string name => Marshal.PtrToStringUni( m_name ) ?? "n/a";

	/// <summary>The number of bytes of dedicated video memory that are not shared with the CPU</summary>
	public readonly ulong vram;

	/// <summary>Vendor ID</summary>
	public readonly eGpuVendor vendor;

	/// <summary>D3D feature level</summary>
	public readonly byte featureLevelMajor, featureLevelMinor;

	/// <summary>Optional features</summary>
	public readonly eOptionalFeatures optionalFeatures;
}