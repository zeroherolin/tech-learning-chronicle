# Pybind绑定生成器

## 准备工作

安装依赖

```bash
sudo apt install python3-dev

pip install pybind11
```
## 示例

elementwise_vector.cpp

```cpp
#include <pybind11/pybind11.h>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

class Vector {
public:
    std::vector<double> data;
    
    Vector(py::list list = py::list()) {
        for (auto item : list) {
            data.push_back(item.cast<double>());
        }
    }

    Vector add(const Vector& other) const {
        Vector result;
        for (size_t i = 0; i < data.size(); i++) {
            result.data.push_back(data[i] + other.data[i]);
        }
        return result;
    }

    Vector sub(const Vector& other) const {
        Vector result;
        for (size_t i = 0; i < data.size(); i++) {
            result.data.push_back(data[i] - other.data[i]);
        }
        return result;
    }

    std::string to_string() const {
        if (data.empty()) return "Vector([])";
        
        std::string result = "Vector([";
        for (size_t i = 0; i < data.size(); i++) {
            result += std::to_string(data[i]);
            if (i < data.size() - 1) result += ", ";
        }
        result += "])";
        return result;
    }

    size_t size() const {
        return data.size();
    }
};

PYBIND11_MODULE(elementwise_vector, m) {
    m.doc() = "Element-wise vector operations";
    
    py::class_<Vector>(m, "Vector")
        .def(py::init<py::list>(), 
             py::arg("data") = py::list(),
             "Create a vector from a list of numbers")
        .def("add", &Vector::add, "Element-wise vector addition")
        .def("sub", &Vector::sub, "Element-wise vector subtraction")
        .def("size", &Vector::size, "Get vector size")
        .def("__repr__", &Vector::to_string);
}
```

## 编译

gcc

```bash
c++ -O3 -Wall -shared -std=c++11 -fPIC \
    $(python3 -m pybind11 --includes) \
    elementwise_vector.cpp \
    -o elementwise_vector$(python3-config --extension-suffix)
```

cmake

```cmake
cmake_minimum_required(VERSION 3.12)
project(elementwise_vector)

# 查找 Python 和 pybind11
find_package(Python3 REQUIRED COMPONENTS Development.Module Interpreter)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

# 添加模块
pybind11_add_module(
    elementwise_vector 
    elementwise_vector.cpp
)
```

## Python测试

```python
import elementwise_vector as ev

v1 = ev.Vector([1.0, 2.0, 3.0])
v2 = ev.Vector([4.0, 5.0, 6.0])

print("Vector 1:", v1)
print("Vector 2:", v2)

v_add = v1.add(v2)
print("Element-wise addition:", v_add)

v_sub = v1.sub(v2)
print("Element-wise subtraction:", v_sub)
```

***
🔙 [Go Back](README.md)
