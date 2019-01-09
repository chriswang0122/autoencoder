#include <iostream>
#include <vector>
#include <random>
#include <armadillo>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "mnist.hpp"
#include "network.hpp"
#include "activation.hpp"
#include "convert.hpp"


int main()
{
    // load mnist data
    arma::mat train_data = read_mnist("mnist/train-images.idx3-ubyte") / 255.0;
    arma::mat test_data = read_mnist("mnist/t10k-images-idx3-ubyte") / 255.0;

    Network<activation::Sigmoid> net({784, 256, 128, 256, 784});

    double learning_rate = 0.001;
    int epoch = 25;
    int minibatch_size = 100;
    net.train(train_data, learning_rate, epoch, minibatch_size);
    net.save();

    // save result
    auto result = net.test(test_data);
    result.save("result");

    // concatenate data and result
    arma::mat img, rec;
    for (int i = 0; i < 20; i++) {
        arma::mat t1 = test_data.row(i), t2 = result.row(i);
        t1.reshape(28, 28), t2.reshape(28, 28);
        img = arma::join_cols(img, t1);
        rec = arma::join_cols(rec, t2);
    }
    arma::mat tmp = arma::join_rows(img, rec);

    // show picture
    auto pic = to_cvmat(tmp);
    cv::imwrite("compare.png", pic * 255);
    cv::imshow("compare", pic);
    cv::waitKey(0);

    return 0;
}
