#ifndef KNN_HPP
#define KNN_HPP

#include <algorithm>
#include <cmath>
#include <list>
#include <utility>
#include <vector>

using namespace std;
typedef vector<double> Point;

class L2KNN
{
private:
    int dim;
    vector<Point> points;

public:
    L2KNN(int n_dim, vector<Point> ps);
    vector<Point> kneighbors(const Point &q, const int k);
};

L2KNN::L2KNN(int n_dim, vector<Point> ps)
{
    dim = n_dim;
    points = ps;
}

double distance(const Point &a, const Point &b, int dim)
{
    double sum = 0.0;
    for (int i = 0; i < dim; i++)
    {
        sum += pow(a[i] - b[i], 2);
    }
    return sqrt(sum);
}

vector<Point> L2KNN::kneighbors(const Point &q, const int k)
{
    list<pair<double, Point>> finding;

    auto insert = [&](Point p) {
        double d = distance(q, p, dim);
        auto it = finding.begin();
        for (; it != finding.end(); it++) {
            if(d < it->first ){
                finding.insert(it, pair<double, Point>(d, p));
                break;
            }
        }
        if (it == finding.end() && finding.size() < k) {
            finding.push_back(pair<double, Point>(d, p));
        }
    };

    for (auto &p : points)
    {
        insert(p);
    }

    vector<Point> result;
    for (auto it = finding.begin(); it != finding.end(); it++ ) {
        result.push_back(it->second);
        if(result.size() >= k) break;
    }
    return result;
}

#endif