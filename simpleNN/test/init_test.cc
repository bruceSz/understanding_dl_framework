#include "../network.h"

using namespace std;
int main(int argc, char** argv) {
    vector<int> lay_n = {784, 100,10};
    Network net;
    net.initNet(lay_n);
    net.initWeights(0,0.,0.01);
    net.initBias(Scalar(0.05));
    return 0;
}