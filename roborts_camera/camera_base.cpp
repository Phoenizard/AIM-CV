#include "camera_base.h"
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

using namespace roborts_camera;

void CameraBase::StartReadCamera(cv::Mat &img)
{
    rs2::colorizer color_map;
    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;

    pipe.start();
    std::cout << "stage\n";

     rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
    //rs2::frame depth = data.get_depth_frame().apply_filter(color_map);
    rs2::frame depth = data.get_color_frame().apply_filter(color_map);

    // Query frame size (width and height)
    const int w = depth.as<rs2::video_frame>().get_width();
    const int h = depth.as<rs2::video_frame>().get_height();

    // Create OpenCV matrix of size (w,h) from the colorized depth data
    cv::Mat image(cv::Size(w, h), CV_8UC3, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
}


