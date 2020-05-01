# GEHealthCare
Homework assignment for GE Healthcare job application. 
UWP app for for fully connected neural network training and testing on the MNIST dataset.

**If the app is bugged or the solution is not building, do not hesitate to contact me!**

## Notes and Remarks

### Prerequisites
 - There are three OpenCV NuGet packages used, the build includes them, but they might be installed manually, see packages.config file.
 - If you want to use the GPU version, you have to install CUDA.
   - **The default is to use CUDA**
   - I have CUDA 10.2 locally, and the only way to create the package without manual copying was to include the dll in the repo.
   - To enable/disable the CUDA version, you comment/uncomment the `#define CUDA` macro in pch.h
 - After you built the GENN project, **place all four (train and test, images and labels) MNIST files into the AppX folder.** This has to be done for both Release and Debug build
 - This is only tested with x64.

### The UI
 - I chose UWP, I did not know (I might have asked) whether it is considered to be WPF/Windows Forms, but at least the UI is xaml and there was a sample project provided in VS2019.
 - The other solution would be to create a C# app and link the native DLL through P/Invoke, but I had no experience with P/Invoke and marshaling looked a bit scary.
 
### About the algorithm
 - The network structure is hard-coded as it was not part of the specification to make it user-modifiable, this one reaches 80% precision relatively quickly.
 - The GPU training is significantly slower, it could be speeded up with minibatch training, but with this many neurons CPU might still outperform GPU.
 
 
 
 
 
