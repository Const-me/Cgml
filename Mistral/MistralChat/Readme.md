This program is a GUI application which implements chat with Mistral ML model,
which runs offline on your personal Windows PC.

The GUI is based on [WPF](https://learn.microsoft.com/en-us/dotnet/desktop/wpf/overview/?view=netdesktop-6.0) framework.<br/>
For this reason, to run the app you gonna need .NET Desktop Runtime, as opposed to just the .NET Runtime.

# Quick Start Guide

Download .NET 6.0 Desktop Runtime for Windows OS and x64 platform [from microsoft.com](https://dotnet.microsoft.com/en-us/download/dotnet/6.0).<br/>
Install that thing.

Download `Mistral-7B-instruct-bcml1.cgml` file with BitTorrent, from the following magnet link:<br/>
`magnet:?xt=urn:btih:E1419810A5CB8419958B02170AB044DD7354F39C&dn=Mistral-7B-instruct&tr=udp%3A%2F%2Fbt2.archive.org%3A6969%2Fannounce`<br />
If the link won’t work, this folder also contains the corresponding `Mistral-7B-instruct.torrent` file.

Note it’s a large file, 4.55 GB.<br/>
I have only rented the [seedbox](https://seedboxes.cc/) for 3 months, the link will probably expire in March 2024.

Download `MistralChat.zip` from Releases page of this repository.<br/>
Extract the ZIP, and run MistralChat.exe.

In the main menu of the window, press “File / Open Model…” command.<br/>
Select the `Mistral-7B-instruct-bcml1.cgml` file.<br/>
Press OK button.

Chat with the AI.

## Performance

If you have a half-decent discrete GPU, click Options / Generation Options menu item,
increase queue depth to 32, check “Fast graphics card” checkbox, press OK, and reload the model.<br/>
The defaults are low because otherwise the app gonna fail on slow GPUs, with DXGI timeout errors.

On my desktop with nVidia 1080 Ti, I'm getting over 20 tokens/second when using BCML1-compressed weights.<br/>
On my laptop with integrated AMD GPU I'm getting about 2 tokens/second.

It seems the performance bottleneck is memory bandwidth.<br/>
VRAM in nVidia 1080 Ti delivers up to 484 GB/second, dual-channel DDR4 in the laptop up to 50 GB/second.

I have tested on another desktop PC with nVidia GeForce 1650 GPU, which only has 4 GB VRAM.<br/>
The software works, but because the model doesn’t fit in 4GB, it only managed to generate 1 token/second.<br/>
Apparently, to run this program at optimal speed the GPU needs to have at least 6 GB VRAM.

## Converting Original Model

Another way to obtain a model in the supported format is converting the original Python-targeted weights.

Download [mistral-7B-instruct-v0.1b.tar](https://files.mistral-7b-v0-1.mistral.ai/mistral-7B-instruct-v0.1b.tar) file.<br/>
Note it’s a large file, 13.4 GB.<br/>
Extract the `*.tar`

In the main menu of the window, press “File / Import Python…” command.<br/>
Click the "…" button and select the folder with the model extracted from the `.tar` file.<br/>
Press OK button.

If you go this way, and you have at least 14GB VRAM in your GPU, you can skip the lossy compression of the model.
To use uncompressed weights, select “Dense FP16” item in the combobox on the import dialog.

I haven’t tested the uncompressed FP16 implementations much.
It works on my desktop PC with 11GB VRAM because modern Windows uses system RAM to evict/restore GPU resources
which didn’t fit in the dedicated VRAM.
However, it’s rather slow this way.