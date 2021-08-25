#ifndef SRC_ALGORITHM_ALGORITHM_HPP_
#define SRC_ALGORITHM_ALGORITHM_HPP_

#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace Algorithm {

template <typename T>
class Point {
 public:
  Point() = default;
  Point(T x_t, T y_t, float score_t = -1) : x(x_t), y(y_t), score(score_t) {}
  T x = 0, y = 0;
  float score = -1;
};

template <typename T>
class Rect {
 public:
  Rect() = default;
  Rect(T x_t, T y_t, T w_t, T h_t) : x(x_t), y(y_t), w(w_t), h(h_t) {}
  T x = 0, y = 0, w = 0, h = 0;
};

template <typename T>
class Box {
 public:
  Box() = default;
  Box(T xmin_t, T ymin_t, T xmax_t, T ymax_t)
      : xmin(xmin_t), ymin(ymin_t), xmax(xmax_t), ymax(ymax_t) {}

  template <typename V>
  Rect<V> rect() const {
    return Rect<V>(xmin, ymin, xmax - xmin, ymax - ymin);
  }

  T xmin = 0, ymin = 0, xmax = 0, ymax = 0;
  float score = 0;
  int label = -1;
};

template <typename T>
class Rel {
 public:
  Rel() = default;
  Rel(Box<T> box_sub_t, Box<T> box_obj_t)
      : box_sub(box_sub_t), box_obj(box_obj_t) {}

  Box<T> box_sub, box_obj;
  float score = 0;
  int label = -1;
};

template <typename T>
class Polygon {
 public:
  Polygon() = default;
  explicit Polygon(const std::vector<Point<T>>& points_t) : points(points_t) {}
  Polygon(const std::vector<T>& x_t, const std::vector<T>& y_t) {
    for (int n = 0; n < std::min(x_t.size(), y_t.size()); ++n) {
      points.emplace_back(x_t[n], y_t[n]);
    }
  }

  template <typename V>
  Box<V> box() const {
    Box<V> box;
    if (!points.empty()) {
      box.xmin = points[0].x, box.ymin = points[0].y;
      box.xmax = points[0].x, box.ymax = points[0].y;
      for (const auto& point : points) {
        box.xmin = std::min(box.xmin, point.x);
        box.ymin = std::min(box.ymin, point.y);
        box.xmax = std::max(box.xmax, point.x);
        box.ymax = std::max(box.ymax, point.y);
      }
      box.score = score;
      box.label = label;
    }
    return box;
  }

  std::vector<Point<T>> points;
  float score = 0;
  int label = -1;
};

using PointI = Point<int>;
using PointF = Point<float>;
using RectI = Rect<int>;
using RectF = Rect<float>;
using BoxI = Box<int>;
using BoxF = Box<float>;
using RelI = Rel<int>;
using RelF = Rel<float>;
using PolygonI = Polygon<int>;
using PolygonF = Polygon<float>;

using VecPointI = std::vector<PointI>;
using VecPointF = std::vector<PointF>;
using VecRectI = std::vector<RectI>;
using VecRectF = std::vector<RectF>;
using VecBoxI = std::vector<BoxI>;
using VecBoxF = std::vector<BoxF>;
using VecRelI = std::vector<RelI>;
using VecRelF = std::vector<RelF>;
using VecPolygonI = std::vector<PolygonI>;
using VecPolygonF = std::vector<PolygonF>;

using VecChar = std::vector<char>;
using VecUChar = std::vector<unsigned char>;
using VecBool = std::vector<bool>;
using VecInt = std::vector<int>;
using VecFloat = std::vector<float>;
using VecDouble = std::vector<double>;
using VecString = std::vector<std::string>;

}  // namespace Algorithm

namespace Algorithm {

bool ReadBinaryFile(const std::string& binary_file,
                    std::vector<char>* binary_data);

}  // namespace Algorithm

namespace Algorithm {

class Argument {
 public:
  Argument();

  template <typename T>
  explicit Argument(const T& def);

  bool HasArgument(const std::string& name) const;

  template <typename T>
  T GetSingleArgument(const std::string& name, const T& default_value) const;
  template <typename T>
  std::vector<T> GetRepeatedArgument(
      const std::string& name,
      const std::vector<T>& default_value = std::vector<T>()) const;

  template <typename T>
  void AddSingleArgument(const std::string& name, const T& value);
  template <typename T>
  void AddRepeatedArgument(const std::string& name,
                           const std::vector<T>& value);

  friend std::ostream& operator<<(std::ostream& os, const Argument& argument);

  struct Impl;
  std::shared_ptr<Impl> impl_ = nullptr;
};

class IPipeLine {
 public:
  virtual ~IPipeLine() = default;

  virtual std::string Name() const = 0;

  virtual int NumInput() const = 0;
  virtual int NumOutput() const = 0;

  virtual void Run(const std::vector<const void*> inputs,
                   std::vector<void*> outputs) = 0;

  virtual bool CanRunAsync() const { return true; }
};

template <typename Impl>
class ModuleHandle {
 public:
  ModuleHandle(const void* meta_net_data, int meta_net_data_size,
               const Argument& argument);

  template <typename... Args>
  void Run(Args&&... args);

 private:
  std::shared_ptr<Impl> impl_ = nullptr;
};

class DetectModule;
class HoiModule;
class ExtractModule;
class LandmarkModule;
class MapModule;
class OCRModule;

using Detect = ModuleHandle<DetectModule>;
using Hoi = ModuleHandle<HoiModule>;
using Extract = ModuleHandle<ExtractModule>;
using Landmark = ModuleHandle<LandmarkModule>;
using Map = ModuleHandle<MapModule>;
using OCR = ModuleHandle<OCRModule>;

}  // namespace Algorithm

#endif  // SRC_ALGORITHM_ALGORITHM_HPP_
