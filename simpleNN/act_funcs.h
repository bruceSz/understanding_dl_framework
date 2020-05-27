#include <opencv2/core/core.hpp>
#include <iostream>

namespace simpleNN 
{
    cv::Mat sigmoid(cv::Mat &x);

    cv::Mat tanh(cv::Mat &x);

    cv::Mat relu(cv::Mat &x);

    void calcLoss(cv::Mat &output, cv::Mat & target, 
        cv::Mat &output_error, float &loss) ;
} // namespace simpleNN