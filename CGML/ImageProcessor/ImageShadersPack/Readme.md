This C# command-line tool consumes shader binaries compiled by ImageShaders project,
and produces a C++ file with `std::array` of bytes with these binaries.

To reduce the size of `Cgml.dll`, these shader binaries are compressed with [LZ4](https://github.com/lz4/lz4).