#include <utility>
#include <fstream>
#include <algorithm>
#include <iterator>

#include "plog/Log.h"

#include "crnn_net.h"
#include "utils.h"

namespace OCR
{

CRNNNet::CRNNNet(const RecConfig &config)
{
    Initialize(config);
}

CRNNNet::CRNNNet(CRNNNet &&other) noexcept
    : config_(std::exchange(other.config_, {}))
    , net_(std::move(other.net_))
    , keys_(std::move(other.keys_))
{

}

CRNNNet & CRNNNet::operator = (CRNNNet &&other) noexcept
{
    if (this != &other)
    {
        config_ = std::exchange(other.config_, {});
        net_ = std::move(other.net_);
        keys_ = std::move(other.keys_);
    }
    return *this;
}

bool CRNNNet::Initialize(const RecConfig &config)
{
    config_ = config;

    // create net
    net_ = std::make_unique<ncnn::Net>();
    net_->opt.num_threads = config_.infer_threads;
    net_->opt.use_fp16_packed = config_.is_fp16;
    net_->opt.use_fp16_storage = config_.is_fp16;
    net_->opt.use_fp16_arithmetic = config_.is_fp16;

    // load keys
    std::string line;
    std::ifstream ifs{config_.keys_path};
    if (!ifs.is_open())
    {
        PLOGE << "Failed to load keys " << config_.keys_path;
        return false;
    }
    keys_.clear();
    while (std::getline(ifs, line))
        keys_.emplace_back(line);
    PLOGD << "Total keys: " << keys_.size();

    if (net_->load_param((config_.model_path + ".param").c_str()) ||
        net_->load_model((config_.model_path + ".bin").c_str()))
    {
        PLOGE << "Failed to load model " << config_.model_path;
        net_.reset();
        return false;
    }

    return true;
}

std::vector<TextLine> CRNNNet::Rec(const std::vector<cv::Mat> &text_images) const
{
    std::vector<TextLine> text_lines(text_images.size());

    int num_lines = static_cast<int>(text_images.size());
    #pragma omp parallel for num_threads(config_.reco_threads) schedule(dynamic)
    for (int i = 0; i < num_lines; ++i)
    {
        text_lines[i] = Rec(text_images[i]);
    }

    return text_lines;
}

TextLine CRNNNet::Rec(const cv::Mat &text_image) const
{
    if (text_image.empty() || text_image.rows <= 0 || text_image.cols <= 0)
        return TextLine{};

    // resize (cap width to avoid huge tensors on long lines)
    float ratio = static_cast<float>(target_h_) / text_image.rows;
    int rsz_w = static_cast<int>(text_image.cols * ratio);
    rsz_w = std::max(1, std::min(rsz_w, max_width_));

    ncnn::Mat blob = ncnn::Mat::from_pixels_resize(text_image.data, ncnn::Mat::PIXEL_BGR,
        text_image.cols, text_image.rows, rsz_w, target_h_);
    blob.substract_mean_normalize(mean_values_, norm_values_);

    // inference
    ncnn::Extractor ex = net_->create_extractor();
    ex.input("input", blob);
    ncnn::Mat out;
    ex.extract("output", out);

    // Decode directly from ncnn output (no full-buffer copy).
    return Score2TextLine(reinterpret_cast<const float *>(out.data), out.h, out.w);
}

TextLine CRNNNet::Score2TextLine(const float *scores, const int rows, const int cols) const
{
    if (scores == nullptr || cols != static_cast<int>(keys_.size()))
    {
        PLOGE << "Unmatched scores: " << cols << " != " << keys_.size();
        return TextLine{};
    }

    std::string text;
    std::vector<float> text_scores;
    text.reserve(static_cast<size_t>(rows));
    text_scores.reserve(static_cast<size_t>(rows));
    int prev_i = -1;
    const int blank_i = 0;

    for (int i = 0; i < rows; ++i)
    {
        const float *row = scores + i * cols;
        auto max_it = std::max_element(row, row + cols);
        int max_i = static_cast<int>(std::distance(row, max_it));
        float max_v = *max_it;

        if (max_i != blank_i && max_i != prev_i)
        {
            text.append(keys_[max_i]);
            text_scores.emplace_back(max_v);
        }
        prev_i = max_i;
    }

    Trim(text);

    return {std::move(text), std::move(text_scores)};
}

}   // namespace OCR
