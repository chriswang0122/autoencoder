template<class A>
class Network
{
public:
    // nested class
    class Layer
    {
    public:
        int size;
        arma::mat value, weight, delta;
        // adam optimizer parameters
        arma::mat m, v;

        // input layer
        Layer(int size) : size(size) {}

        Layer(int size, Layer *last)
            : size(size), 
              weight(last->size, size),
              m(last->size, size, arma::fill::zeros),
              v(last->size, size, arma::fill::zeros) {
            weight.for_each([](arma::mat::elem_type &val){
                val = initializer();
            });
        }

    private:
        // random generator
        static double initializer() {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::normal_distribution<double> dis(0.0, 1.0);
            return dis(gen);
        }
    };

    Network(std::vector<int> sizes) {
        layers.reserve(sizes.size());
        layers.push_back(Layer(sizes[0]));
        for (int i = 1; i < sizes.size(); i++)
            layers.push_back(Layer(sizes[i], &layers[i - 1]));
    }

    void train(arma::mat data, double learning_rate, 
               int epoch, int minibatch_size = 1) {
        double cost;
        for (int i = 0; i < epoch; i++) {
            std::cout << "Epoch " << i << " cost: ";
            // minibatch
            for (int j = 0; j < data.n_rows; j += minibatch_size) {
                auto range = std::min(j + minibatch_size,
                                      static_cast<int>(data.n_rows)) - 1;
                arma::mat x = data.rows(j, range);
                arma::mat y = data.rows(j, range);
                backpropagation(x, y, learning_rate);
                cost = compute(x, y);
            }
            std::cout << cost << std::endl;
        }
    }

    arma::mat test(arma::mat data) {
        forward(data);
        return layers.back().value;
    }

    void save() {
        for (int i = 1; i < layers.size(); i++)
            layers[i].weight.save("weight" + std::to_string(i));
    }

private:
    std::vector<Layer> layers;
    A activation;

    // compute cost(loss)
    double compute(arma::mat input, arma::mat target) {
        double cost = arma::norm(layers.back().value - target, 2) * 0.5 /
                      static_cast<double>(input.n_rows);
        return cost;
    }

    // mini-batch gradient descent
    /*void backpropagation(arma::mat input, arma::mat target,
                           double learning_rate) {
        // propagation
        forward(input), backward(target);

        // weight update
        for (auto itr = layers.begin() + 1; itr != layers.end(); itr++)
            itr->weight -= learning_rate / static_cast<double>(input.n_rows) *
                           ((itr - 1)->value.t() * itr->delta);
    }*/

    // adam optimizer
    void backpropagation(arma::mat input, arma::mat target,
                         double learning_rate) {
        // adam optimizer hyperparameters
        static double beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, t = 0.0;

        // forward passs
        forward(input);

        // backward pass
        backward(target);

        // weight update
        t++;
        for (auto itr = layers.begin() + 1; itr != layers.end(); itr++) {
            // calculate gradient
            auto g = ((itr - 1)->value.t() * itr->delta) /
                     static_cast<double>(input.n_rows);
            // calculate adam optimizer parameters
            itr->m = beta_1 * itr->m + (1 - beta_1) * g;
            itr->v = beta_2 * itr->v + (1 - beta_2) * (g % g);
            auto bias_m = itr->m / (1.0 - std::pow(beta_1, t));
            auto bias_v = itr->v / (1.0 - std::pow(beta_2, t));
            // update
            itr->weight -= learning_rate * bias_m /
                           (arma::sqrt(bias_v) + epsilon);
        }
    }

    void forward(arma::mat input) {
        layers.front().value = input;
        for (auto itr = layers.begin() + 1; itr != layers.end(); itr++)
            itr->value = activation((itr - 1)->value * itr->weight);
    }

    void backward(arma::mat target) {
        layers.back().delta = (layers.back().value - target) %
                              activation(layers.back().value, true);
        for (auto itr = layers.rbegin() + 1; itr != layers.rend() - 1; itr++)
            itr->delta = ((itr - 1)->delta * (itr - 1)->weight.t()) %
                         activation(itr->value, true);
    }
};
