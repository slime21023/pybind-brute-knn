#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "./knn.h"

namespace py = pybind11;

struct EuclideanKnn
{
    EuclideanKnn(const py::array_t<double> points)
    {
        py::buffer_info buf = points.request();

        int rows = buf.shape[0];
        int cols = buf.shape[1];

        vector<Point> ps;
        for (int idx = 0; idx < rows; idx++)
        {
            double *row = static_cast<double *>(buf.ptr) + idx * cols;
            Point p(row, row + cols);

            ps.push_back(p);
        }

        nn = new L2KNN(cols, ps);
    };

    vector<Point> kneighbors(const Point &q, const int k)
    {
        return nn->kneighbors(q, k);
    };

    L2KNN *nn;
};    

PYBIND11_MODULE(pybind_brute_knn, m)
{
    py::class_<EuclideanKnn>(m, "EuclideanKnn")
        .def(py::init<const py::array_t<double> &>())
        .def("kneighbors", &EuclideanKnn::kneighbors);
}