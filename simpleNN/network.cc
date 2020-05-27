#include "net.h"

namespace simpleNN
{

void Network::initNet(std::vector<int> layer_neuron_num_) 
{
    // init each layer size.
    layer_neuron_num = layer_neuron_num_;
    layer.resize(layer_neuron_num.size());
    for(int i=0; i<layer.size(); i++) {
        layer[i].create(layer_neuron_num[i], 1, CV_32FC1);
        
    }

    std::cout << "Layer created." << std::endl;

    weights.resize(layer.size() - 1);
    bias.resize(layer.size() -1);
    for(int i=0;i<(layer.size() -1);i++) {
        weights[i].create(layer[i+1].rows, layer[i].rows,CV_32FC1);
        bias[i].create(layer[i+1].rows,1,CV_32FC1);
    }

    std::cout << " Network created." << std::endl;
}

void Network::initWeight(cv::Mat &dst, int type, double a  , double b) {
    if (type==0) {
        randn(dst, a, b);
    } else {
        randu(dst, a, b);
    }
}

void Network::initWeights(int type, double a, double b) {
    for(int i=0;i<weights.size();i++) {
        initWeight(weights[i], 0,0,0.1);
    }
}


void Network::initBias(cv::Scalar& bias_) {
    for(i=0;i < bias.size(); i++) {
        bias[i]  = bias_;
    }
}


void Network::forward() {

    // traverse all layers using for loop,
    // which is very slow.
    // It is clear that forward is to update x'1, x'2,
    // different layer of values computed from last layer 
    // multiplied by weights(and bias).
    for(int i=0;i<layer_neuron_num.siz() -1;i++) {
        cv::Mat product = weights[i] * layer[i] + bias[i];
        layer[i+1] = act(product, act_type);
    }
}

cv::Mat Network::act(cv::Mat& x, std::string func_type) {
    cv::Mat fx;
    if (func_type == "sigmoid") {
        fx = sigmoid(x);
    } else if (func_type == "tanh") {
        fx = tanh(x);
    } else if (func_type == "relu")  {
        fx = relu(x);
    }
    return fx;
}

void Network::backward() {
    calcLoss(layer[layer.size() -1], target, output_error, loss);
    deltaError();
    updateWeights();
}

void Network::deltaError() {
    delta_err.resize(layer.size()-1);
    for(int i=delta_err.size() -1; i>=0;i--) {
        delta_err[i].create(layer[i+1].size(), layer[i+1].type());
        cv::Mat dx = derivateFunc(layer[i+1], act_type);
        if (i== delta_err.size() -1) {
            delta_err[i] = dx.mul(output_error);
        } else {
            //TODO ? 
        }

    }
}

}// namespace SimpleNN