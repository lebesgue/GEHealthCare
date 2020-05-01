//
// pch.h
//

#define CUDA

#ifdef CUDA
#include "DenseLayer.cuh"
#define pf cuda
#else
#include "DenseLayer.h"
#define pf cpu
#endif

#pragma once

#include <collection.h>
#include <ppltasks.h>
#include <string>
#include <iostream>
#include <thread>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>

#include <windows.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "App.xaml.h"
