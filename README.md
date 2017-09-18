
# SharpNeuron

SharpNeuron is a .NET Core Neural Network library based on [NeuronDotNet](https://sourceforge.net/projects/neurondotnet/), which seems to have been abandoned.
The vast majority of the code comes directly from NeuronDotNet.

SharpNeuron consists of purely managed code that is written for .NET Core (`.NETStandard1.6` profile)
and will run on any .NET Core supported platform, including Windows, macOS and Linux.

## About this project

The core of SharpNeuron (NeuronDotNet) has been left mostly intact, though specific changes have been made to make it more customizable.
One of the major goals of this project is to provide high-level APIs for specialized purposes to improve usability and simplicity.

In particular, the `Architect` API of SharpNeruon is based on [synaptic.js](https://github.com/cazala/synaptic).

## Using SharpNeuron

Simply clone the source and add the `SharpNeuron` project to your solution. This will allow you to customize it in case you would like to, and allows you to avoid using any precompiled libraries.

SharpNeuron has no platform-specific limitations, and should work on any .NET platform.

## License

SharpNeuron is licensed under the GNU GPLv3.

### Copyright

The original NeuronDotNet project: (c) 2008 Vijeth D

All new SharpNeuron code: (c) 2016 0xFireball
