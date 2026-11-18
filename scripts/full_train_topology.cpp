// Count exterior contours and holes in an original-resolution foreground crop.
#include <opencv2/imgproc.hpp>
#include <vector>
extern "C" int topology(unsigned char* data, int h, int w, int* holes) {
    cv::Mat mask(h, w, CV_8UC1, data);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    int components = 0;
    *holes = 0;
    for (const auto& node : hierarchy) {
        if (node[3] < 0) ++components;
        else ++*holes;
    }
    return components;
}
