# GEHealthCare
Homework assignment for GE Healthcare job application. Dense neural network implementation, training and testing UWP app for MNIST dataset.

If you find bugs in the app or cannot make it work, do not hesitate to contact be!

## Notes and Remarks

### Prerequisites
 - There are three OpenCV NuGet packages used, the build includes them, but they might be installed manually, see packages.config file.
 - If you want to use the GPU version, you have to install CUDA.
   - **The default is to use CUDA**
   - I have CUDA 10.2 locally, and the only way to create the package was to include the dll in the repo.
   - To enable/disable the CUDA version, you comment/uncomment the #define CUDA macro in pch.h
 - After you built the GENN project, place all four (train and test, images and labels) MNIST files into the AppX folder. This has to be done for both Release and Debug build
 - This is only tested with x64.

### The UI
 - This is an UWP app, I did not know (I might have asked) whether it is considered to be WPF/Windows Forms, but at least the UI is xaml and there was a sample project provided.
 - The other solution would be to create a C# app and link the native DLL through P/Invoke, but this seemed easier.
 
### About the algorithm
 - The network structure is hard-coded as it was not part of the specification to make it user-modifiable, this one reaches 80% precision relatively quickly.
 - The GPU training is significantly slower, it could be speeded up with minibatch training, but on this scale CPU is better.
 
 
 
 
 
