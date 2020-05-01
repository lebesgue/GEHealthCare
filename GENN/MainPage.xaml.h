//
// MainPage.xaml.h
// Declaration of the MainPage class.
//

#pragma once

#include "MainPage.g.h"
#include "NN.h"
#include "reader.h"

#include <Robuffer.h>

using namespace Windows::UI::Xaml::Media::Imaging;
using namespace Windows::Storage::Streams;
using namespace Windows::System::Threading;
using namespace Microsoft::WRL;

namespace GENN
{
	/// <summary>
	/// An empty page that can be used on its own or navigated to within a Frame.
	/// </summary>
	public ref class MainPage sealed
	{
	public:
		MainPage();

	private:

		std::vector<cpu::Matrix> images;
		std::vector<int> labels;

		std::vector<int> testOrder;
		std::vector<cpu::Matrix> testImages;
		std::vector<int> testLabels;

		nn::Network network;
		std::thread trainThread;
		std::thread updateThread;

		cv::Mat dashboard;

		ThreadPoolTimer^ uiUpdateTimer;
		ThreadPoolTimer^ dashboardUpdateTimer;

		void updateLayout(ThreadPoolTimer^ timer);
		void updateImages(ThreadPoolTimer^ timer);
		void loadMNIST(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
		void drawImages(std::vector<cpu::Matrix>& images, std::vector<int>& predictions);
		void updateDashboard();
		void startTraining(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
		void pauseTraining(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
		void testNN(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
	};
}
