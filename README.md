# MNIST
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

 
 
 
 
 
