// convert an Armadillo matrix to OpenCV matrix.
template <class T>
cv::Mat_<T> to_cvmat(const arma::Mat<T> &arma_mat)
{
  return cv::Mat_<T>{int(arma_mat.n_cols), int(arma_mat.n_rows),
                     const_cast<T*>(arma_mat.memptr())};
}