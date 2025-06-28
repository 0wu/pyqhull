#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <thread>
#include <future>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>

namespace py = pybind11;

extern "C" {
#include "libqhull_r/libqhull_r.h"
}

class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

public:
    ThreadPool(size_t threads) : stop(false) {
        for(size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for(;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    auto enqueue(F&& f) -> std::future<typename std::result_of<F()>::type> {
        using return_type = typename std::result_of<F()>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if(stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers) worker.join();
    }
};

std::vector<bool> compute_convex_hull_mask(double* points, int n_points) {
    std::vector<bool> mask(n_points, false);

    if (n_points < 4) {
        for (int i = 0; i < n_points; ++i) {
            mask[i] = true;
        }
        return mask;
    }

    // Initialize qhull context
    qhT qh_qh;
    qhT *qh = &qh_qh;

    // Initialize qhull
    qh_zero(qh, nullptr);

    // Run qhull using reentrant API
    char flags[] = "qhull Qt";
    int exitcode = qh_new_qhull(qh, 3, n_points, points, 0, flags, nullptr, nullptr);

    if (exitcode) {
        // Raise Python error with exitcode
        PyErr_SetString(PyExc_RuntimeError, ("qh_new_qhull failed with exit code " + std::to_string(exitcode)).c_str());
        throw py::error_already_set();
    }

    // Mark vertices that are part of the convex hull
    vertexT *vertex;
    FORALLvertices {
        int vertex_id = qh_pointid(qh, vertex->point);
        if (vertex_id >= 0 && vertex_id < n_points) {
            mask[vertex_id] = true;
        }
    }

    // Clean up qhull
    qh_freeqhull(qh, !qh_ALL);
    int curlong, totlong;
    qh_memfreeshort(qh, &curlong, &totlong);

    return mask;
}


// Given the set of points and a convex hull mask, find the hyperplane parameters of the convex hull
std::vector<std::vector<double>> compute_hyperplanes_from_mask(double* points, int n_points, const std::vector<bool>& mask) {
    std::vector<std::vector<double>> hyperplanes;

    // Collect only the points on the convex hull
    std::vector<double> hull_points;
    int hull_count = 0;
    for (int i = 0; i < n_points; ++i) {
        if (mask[i]) {
            hull_points.insert(hull_points.end(), points + i * 3, points + (i + 1) * 3);
            ++hull_count;
        }
    }
    if (hull_count < 4) {
        // Not enough points for a 3D convex hull
        return hyperplanes;
    }

    // Initialize qhull context
    qhT qh_qh;
    qhT *qh = &qh_qh;
    qh_zero(qh, nullptr);

    char flags[] = "qhull Qt";
    int exitcode = qh_new_qhull(qh, 3, hull_count, hull_points.data(), 0, flags, nullptr, nullptr);

    if (exitcode) {
        PyErr_SetString(PyExc_RuntimeError, ("qh_new_qhull failed with exit code " + std::to_string(exitcode)).c_str());
        throw py::error_already_set();
    }

    // Extract hyperplane parameters from each facet
    facetT *facet;
    FORALLfacets {
        if (!facet->upperdelaunay) {
            std::vector<double> plane(4);
            for (int i = 0; i < 3; ++i) {
                plane[i] = facet->normal[i];
            }
            plane[3] = facet->offset;
            hyperplanes.push_back(plane);
        }
    }

    // Clean up qhull
    qh_freeqhull(qh, !qh_ALL);
    int curlong, totlong;
    qh_memfreeshort(qh, &curlong, &totlong);

    return hyperplanes;
}




// Global threadpool size variable and persistent pool pointer
static int g_threadpool_size = -1;
static ThreadPool* g_threadpool = nullptr;
static std::mutex g_threadpool_mutex;

void set_threadpool_size(int num_threads) {
    std::lock_guard<std::mutex> lock(g_threadpool_mutex);
    g_threadpool_size = num_threads;
    if (g_threadpool) {
        delete g_threadpool;
        g_threadpool = nullptr;
    }
    if (g_threadpool_size > 0) {
        g_threadpool = new ThreadPool(g_threadpool_size);
    }
}

pybind11::array_t<bool> convex_hull_batch(pybind11::array_t<double> points) {
    auto buf = points.request();

    if (buf.ndim != 3) {
        throw std::runtime_error("Input must be 3D array (batch, n_points, 3)");
    }
    if (buf.shape[2] != 3) {
        throw std::runtime_error("Input must have 3 coordinates per point");
    }

    int batch_size = buf.shape[0];
    int n_points = buf.shape[1];

    // Create output array
    auto result = pybind11::array_t<bool>({batch_size, n_points});
    auto result_buf = result.request();

    // Get data pointers
    double* points_data = static_cast<double*>(buf.ptr);
    bool* result_data = static_cast<bool*>(result_buf.ptr);

    ThreadPool* pool = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_threadpool_mutex);
        pool = g_threadpool;
        if (!pool) {
            throw std::runtime_error("Thread pool not initialized. Call set_threadpool_size() first.");
        }
    }

    // Submit tasks for each batch
    std::vector<std::future<void>> futures;
    for (int b = 0; b < batch_size; ++b) {
        futures.push_back(pool->enqueue([=]() {
            double* batch_points = points_data + b * n_points * 3;
            std::vector<bool> mask = compute_convex_hull_mask(batch_points, n_points);
            for (int i = 0; i < n_points; ++i) {
                result_data[b * n_points + i] = mask[i];
            }
        }));
    }

    for (auto& future : futures) {
        future.wait();
    }

    return result;
}

// Python binding: given points and mask, return list of hyperplane parameters
py::list convex_hull_hyperplanes_from_mask(py::array_t<double> points, py::array_t<bool> mask) {
    auto points_buf = points.request();
    auto mask_buf = mask.request();

    if (points_buf.ndim != 3 || points_buf.shape[2] != 3) {
        throw std::runtime_error("Points must be a 3D array with shape (n_envs, n_points, 3)");
    }
    if (mask_buf.ndim != 2 || mask_buf.shape[1] != points_buf.shape[1] || mask_buf.shape[0] != points_buf.shape[0]) {
        throw std::runtime_error("Mask must be a 2D array with shape (n_envs, n_points)");
    }

    int n_envs = points_buf.shape[0];
    int n_points = points_buf.shape[1];
    double* points_data = static_cast<double*>(points_buf.ptr);
    bool* mask_data = static_cast<bool*>(mask_buf.ptr);

    py::list result;
    for (int env = 0; env < n_envs; ++env) {
        double* env_points = points_data + env * n_points * 3;
        bool* env_mask = mask_data + env * n_points;
        std::vector<bool> mask_vec(env_mask, env_mask + n_points);
        std::vector<std::vector<double>> planes = compute_hyperplanes_from_mask(env_points, n_points, mask_vec);
        py::list env_planes;
        for (const auto& plane : planes) {
            env_planes.append(py::cast(plane));
        }
        result.append(env_planes);
    }
    return result;
}

double point_to_hyperplanes_min_distance(const std::vector<std::vector<double>>& hyperplanes, const std::vector<double>& ref_point) {
    if (ref_point.size() != 3) {
        throw std::runtime_error("Reference point must have 3 coordinates");
    }
    double min_dist = std::numeric_limits<double>::max();
    for (const auto& plane : hyperplanes) {
        if (plane.size() != 4) continue;
        double num = std::abs(plane[0] * ref_point[0] + plane[1] * ref_point[1] + plane[2] * ref_point[2] + plane[3]);
        double denom = std::sqrt(plane[0]*plane[0] + plane[1]*plane[1] + plane[2]*plane[2]);
        if (denom == 0.0) continue;
        double dist = num / denom;
        if (dist < min_dist) min_dist = dist;
    }
    return min_dist;
}

py::array_t<double> min_distance_to_hyperplanes(py::array_t<double> points, py::array_t<double> ref_points) {
    //The min_distance_to_hyperplanes function computes, for each environment (batch), the signed minimum distance from a reference point to the convex hull defined by a set of 3D points.
    // points: shape (n_envs, n_points, 3)
    // ref_points: shape (n_envs, 3)
    auto points_buf = points.request();
    auto ref_points_buf = ref_points.request();

    if (points_buf.ndim != 3 || points_buf.shape[2] != 3) {
        throw std::runtime_error("Points must be a 3D array with shape (n_envs, n_points, 3)");
    }
    if (ref_points_buf.ndim != 2 || ref_points_buf.shape[1] != 3 || ref_points_buf.shape[0] != points_buf.shape[0]) {
        throw std::runtime_error("ref_points must be a 2D array with shape (n_envs, 3)");
    }

    int n_envs = points_buf.shape[0];
    int n_points = points_buf.shape[1];
    double* points_data = static_cast<double*>(points_buf.ptr);
    double* ref_points_data = static_cast<double*>(ref_points_buf.ptr);

    // Output array
    auto result = py::array_t<double>(n_envs);
    auto result_buf = result.request();
    double* result_data = static_cast<double*>(result_buf.ptr);

    ThreadPool* pool = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_threadpool_mutex);
        pool = g_threadpool;
        if (!pool) {
            throw std::runtime_error("Thread pool not initialized. Call set_threadpool_size() first.");
        }
    }

    std::vector<std::future<void>> futures;
    for (int env = 0; env < n_envs; ++env) {
        futures.push_back(pool->enqueue([=]() {
            double* env_points = points_data + env * n_points * 3;
            double* env_ref_point = ref_points_data + env * 3;
            std::vector<bool> mask = compute_convex_hull_mask(env_points, n_points);
            std::vector<std::vector<double>> planes = compute_hyperplanes_from_mask(env_points, n_points, mask);
            std::vector<double> ref_point_vec(env_ref_point, env_ref_point + 3);

            // Compute signed distances to all planes
            double min_abs_dist = std::numeric_limits<double>::max();
            double sign = 1.0;
            bool inside = true;
            for (const auto& plane : planes) {
                if (plane.size() != 4) continue;
                double d = plane[0] * ref_point_vec[0] + plane[1] * ref_point_vec[1] + plane[2] * ref_point_vec[2] + plane[3];
                double denom = std::sqrt(plane[0]*plane[0] + plane[1]*plane[1] + plane[2]*plane[2]);
                if (denom == 0.0) continue;
                double dist = d / denom;
                if (dist > 1e-12) inside = false; // outside at least one plane
                if (std::abs(dist) < std::abs(min_abs_dist)) {
                    min_abs_dist = dist;
                    sign = (dist >= 0) ? 1.0 : -1.0;
                }
            }
            // If outside, return negative distance
            result_data[env] = inside ? std::abs(min_abs_dist) : -std::abs(min_abs_dist);
        }));
    }

    for (auto& future : futures) {
        future.wait();
    }

    return result;
}


/*
 * Compute the Minkowski sum of n sets of vectors.
 * Each contact (arg[c]) is a matrix of shape (n_fric_edge, vec_dim).
 * The result is a matrix where each row is a sum of one vector from each contact.
 *
 * Args:
 *   arg: std::vector<std::vector<std::vector<double>>> of shape (numContacts, n_fric_edge, vec_dim)
 * Returns:
 *   std::vector<std::vector<double>>: all possible Minkowski sums (rows)
 */
void minkowski_sum_recursive(
    int c,
    std::vector<double>& prevSum,
    const std::vector<std::vector<std::vector<double>>>& arg,
    std::vector<std::vector<double>>& result,
    int n_fric_edge,
    int numContacts,
    int vec_dim
) {
    if (c == numContacts) {
        result.push_back(prevSum);
        return;
    }

    // Exclude this contact
    minkowski_sum_recursive(c + 1, prevSum, arg, result, n_fric_edge, numContacts, vec_dim);

    // Include each friction edge of this contact
    const auto& this_contact = arg[c];
    for (int w = 0; w < n_fric_edge; ++w) {
        std::vector<double> nextSum(prevSum);
        for (int d = 0; d < vec_dim; ++d) {
            nextSum[d] += this_contact[w][d];
        }
        minkowski_sum_recursive(c + 1, nextSum, arg, result, n_fric_edge, numContacts, vec_dim);
    }
}

// Python binding for Minkowski sum
py::array_t<double> minkowski_sum(py::list arg) {
    // arg: list of numContacts arrays, each (n_fric_edge, vec_dim)
    int numContacts = arg.size();
    if (numContacts == 0)
        return py::array_t<double>(std::vector<ssize_t>{0, 0});

    std::vector<std::vector<std::vector<double>>> arg_vec;
    int n_fric_edge = -1, vec_dim = -1;
    for (int c = 0; c < numContacts; ++c) {
        py::array_t<double> arr = py::cast<py::array_t<double>>(arg[c]);
        auto buf = arr.request();
        if (buf.ndim != 2)
            throw std::runtime_error("Each contact must be a 2D array");
        if (n_fric_edge == -1) n_fric_edge = buf.shape[0];
        if (vec_dim == -1) vec_dim = buf.shape[1];
        if (buf.shape[0] != n_fric_edge || buf.shape[1] != vec_dim)
            throw std::runtime_error("All contacts must have the same shape");
        std::vector<std::vector<double>> contact(n_fric_edge, std::vector<double>(vec_dim));
        double* data = static_cast<double*>(buf.ptr);
        for (int i = 0; i < n_fric_edge; ++i)
            for (int j = 0; j < vec_dim; ++j)
                contact[i][j] = data[i * vec_dim + j];
        arg_vec.push_back(std::move(contact));
    }

    std::vector<double> prevSum(vec_dim, 0.0);
    std::vector<std::vector<double>> result;
    minkowski_sum_recursive(0, prevSum, arg_vec, result, n_fric_edge, numContacts, vec_dim);

    // Convert result to numpy array
    ssize_t n_rows = result.size();
    py::array_t<double> out(std::vector<ssize_t>{n_rows, vec_dim});
    auto out_buf = out.request();
    double* out_ptr = static_cast<double*>(out_buf.ptr);
    for (ssize_t i = 0; i < n_rows; ++i)
        for (int j = 0; j < vec_dim; ++j)
            out_ptr[i * vec_dim + j] = result[i][j];
    return out;
}


PYBIND11_MODULE(pyqhull, m) {
    m.doc() = "Batch convex hull computation using qhull";
    m.def("convex_hull_batch", &convex_hull_batch, "Compute convex hulls for batched 3D points",
          pybind11::arg("points"));
    m.def("set_threadpool_size", &set_threadpool_size, "Set the global threadpool size for convex hull computation",
          pybind11::arg("num_threads"));
    m.def("convex_hull_hyperplanes_from_mask", &convex_hull_hyperplanes_from_mask,
          "Compute convex hull hyperplanes from a mask",
          pybind11::arg("points"), pybind11::arg("mask"));
    m.def("min_distance_to_hyperplanes", &min_distance_to_hyperplanes,
        "Compute the minimum distance from a point to a set of hyperplanes",
        pybind11::arg("points"), pybind11::arg("ref_point"));
    m.def("minkowski_sum", &minkowski_sum, "Compute the Minkowski sum of multiple sets of vectors",
          pybind11::arg("arg"));
}