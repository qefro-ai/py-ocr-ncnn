#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ocr_engine.h"
#include "plog/Init.h"
#include "plog/Log.h"

namespace py = pybind11;

namespace
{

cv::Mat numpy_to_bgr_mat(const py::array_t<uint8_t> &input)
{
    py::buffer_info buf = input.request();
    if (buf.ndim != 3)
    {
        throw std::runtime_error("image must be HxWxC (uint8)");
    }
    if (buf.shape[2] != 3)
    {
        throw std::runtime_error("image must have 3 channels (BGR)");
    }
    if (buf.strides[2] != 1 || buf.strides[1] != static_cast<py::ssize_t>(buf.shape[2]))
    {
        throw std::runtime_error("image must be contiguous BGR (HxWx3)");
    }
    return cv::Mat(static_cast<int>(buf.shape[0]),
                   static_cast<int>(buf.shape[1]),
                   CV_8UC3,
                   static_cast<unsigned char *>(buf.ptr));
}

float mean_score(const std::vector<float> &scores)
{
    if (scores.empty())
    {
        return 0.0f;
    }
    float total = 0.0f;
    for (float score : scores)
    {
        total += score;
    }
    return total / static_cast<float>(scores.size());
}

}  // namespace

class PyOCREngine
{
public:
    explicit PyOCREngine(const std::string &config_path)
    {
        if (!engine_.Initialize(config_path))
        {
            throw std::runtime_error("Failed to initialize OCR engine from config: " + config_path);
        }
    }

    py::list recognize(const py::array_t<uint8_t> &image) const
    {
        const cv::Mat mat = numpy_to_bgr_mat(image);
        const std::vector<OCR::OCRResult> results = engine_.Run(mat);

        py::list out;
        for (const OCR::OCRResult &result : results)
        {
            py::list box;
            for (const cv::Point &point : result.box.points)
            {
                box.append(py::make_tuple(point.x, point.y));
            }

            py::dict item;
            item["text"] = result.line.text;
            item["confidence"] = mean_score(result.line.scores);
            item["scores"] = result.line.scores;
            item["box"] = box;
            item["box_score"] = result.box.score;
            item["rotated"] = result.angle.is_rot;
            item["angle_score"] = result.angle.score;
            out.append(item);
        }
        return out;
    }

    py::list recognize_crops(const py::list &images) const
    {
        std::vector<cv::Mat> mats;
        mats.reserve(images.size());
        for (const auto &item : images)
        {
            mats.push_back(numpy_to_bgr_mat(item.cast<py::array_t<uint8_t>>()));
        }
        const std::vector<OCR::TextLine> lines = engine_.RecognizeCrops(mats);
        py::list out;
        for (const auto &line : lines)
        {
            py::dict item;
            item["text"] = line.text;
            item["confidence"] = mean_score(line.scores);
            item["scores"] = line.scores;
            out.append(item);
        }
        return out;
    }

private:
    OCR::OCREngine engine_{};
};

PYBIND11_MODULE(_ncnn_ocr, m)
{
    plog::init(plog::warning);

    m.doc() = "NCNN-backed PaddleOCR PP-OCR inference extension";

    py::class_<PyOCREngine>(m, "Engine")
        .def(py::init<const std::string &>(), py::arg("config_path"))
        .def("recognize", &PyOCREngine::recognize, py::arg("image"),
             "Run det+cls+rec on a uint8 BGR numpy array (HxWx3).")
        .def("recognize_crops", &PyOCREngine::recognize_crops, py::arg("images"),
             "Run CRNN recognition on a list of cropped BGR numpy arrays (HxWx3).");
}
