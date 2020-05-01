#pragma once

#include <vector>
#include <string>
#include <fstream>

#include "Matrix.h"

void read_Mnist(std::string filename, std::vector<cpu::Matrix> &imgs);
void read_Mnist_Label(std::string filename, std::vector<int> &vec);
