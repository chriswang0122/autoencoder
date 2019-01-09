namespace activation {
class Sigmoid
{
public:
    arma::mat operator()(arma::mat x, bool derived=false) {
        if (derived)
            return x % (1.0 - x);
        else {
            x.for_each([](arma::mat::elem_type &val){
                val = 1.0 / (1.0 + std::exp(-val));
            });
            return x;
        }
    }
};

class Tanh
{
public:
    arma::mat operator()(arma::mat x, bool derived=false) {
        if (derived)
            return 1.0 - (x % x);
        else {
            x.for_each([](arma::mat::elem_type &val){
                val = std::tanh(val);
            });
            return x;
        }
    }
};

class Relu
{
public:
    arma::mat operator()(arma::mat x, bool derived=false) {
        if (derived) {
            x.for_each([](arma::mat::elem_type &val){
                val = val > 0.0 ? 1.0 : 0.0;
            });
        }
        else {
            x.for_each([](arma::mat::elem_type &val){
                val = std::max(val, 0.0);
            });
        }
        return x;
    }
};
}
