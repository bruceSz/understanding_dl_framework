#include "act_funcs.h"
namespace simpleNN
{
void calcLoss(cv::Mat &output, cv::Mat & target, 
        cv::Mat &output_error, float &loss) {
            if (target.empty()) {
                std::cout << "no target mat";
                return;
            }

            output_error = target - output;
            cv::Mat err_square;

            pow(output_error, 2., err_square);
            cv::Scalar err_sqr_sum = sum(err_square);
            //? total number is output.rows?
            loss = err_sqr_sum[0] /(float)(output.rows);
            

        }
}