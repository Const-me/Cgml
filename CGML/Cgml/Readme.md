This project builds a C++ DLL for use by CgmlNet

The DLL includes two large components:

1. Wrappers for the small relevant subset of D3D11.<br/>
That code only implements tensors I/O and low-level context operations.
Individual compute shaders and their semantics are supplied by higher-level code, outside of this project.

2. SentencePiece library written by Google.
Because their code fails to compile with C++/20 language version we have in this project, the implementation is in in the separate static library project.