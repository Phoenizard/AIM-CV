获取深度信息（src+目标点坐标）
```c++
static inline double getPointDepth(const cv::Mat & depthImage, const cv::Point & pos)
{
    return (double) depthImage.at<uint16_t>(pos.y, pos.x) / 1000;
}
```


返回位姿的平移矩阵