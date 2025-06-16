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

PYBIND11_MODULE(pyqhull, m) {
    m.doc() = "Batch convex hull computation using qhull";
    m.def("convex_hull_batch", &convex_hull_batch, "Compute convex hulls for batched 3D points",
          pybind11::arg("points"));
    m.def("set_threadpool_size", &set_threadpool_size, "Set the global threadpool size for convex hull computation",
          pybind11::arg("num_threads"));
}