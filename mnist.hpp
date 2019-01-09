#include <fstream>


int ReverseInt(int i)
{
    int ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return (ch1 << 24) + (ch2 << 16) + (ch3 << 8) + ch4;
}

arma::mat read_mnist(std::string filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        int n_element = n_rows * n_cols;
        arma::mat data(number_of_images, n_element);
        for (int i = 0; i < number_of_images; i++) {
            for (int j = 0; j < n_element; j++) {
                unsigned char tmp = 0;
                file.read((char*) &tmp, sizeof(tmp));
                data(i, j) = static_cast<double>(tmp);
            }
        }
        file.close();

        return data;
    }
    else
        std::cout << "Unable to open file." << std::endl;
}
