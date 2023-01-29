#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include "net.hpp"

using namespace cv;

class Net
{
public:
    std::vector<double> norm_mean = {0.485, 0.456, 0.406};
    std::vector<double> norm_std = {0.229, 0.224, 0.225};
    std::string weights;
    std::string fileName;
    std::string labelFile;
    std::string _cls;
    std::string _probability;

std::vector<std::string> load_labels(std::string& fName)
{   
    std::ifstream ins(fName);
    std::vector<std::string> labels;
    std::string line;

    while (getline(ins, line))
        labels.push_back(line);

    ins.close();

    return labels;
}

torch::Tensor read_image(std::string& imageName)
{   
    Mat img;
    img = imread(imageName);
    img = crop_center(img);
    resize(img, img, Size(224,224));

    if (img.channels()==1)
        cvtColor(img, img, COLOR_GRAY2RGB);
    else
        cvtColor(img, img, COLOR_BGR2RGB);

    img.convertTo( img, CV_32FC3, 1/255.0 );

    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, c10::kFloat);
    img_tensor = img_tensor.permute({2, 0, 1});
    img_tensor.unsqueeze_(0);

    img_tensor = torch::data::transforms::Normalize<>(norm_mean, norm_std)(img_tensor);

    return img_tensor.clone();
}

Mat crop_center(Mat &img)
{
    const int rows = img.rows;
    const int cols = img.cols;

    const int cropSize = std::min(rows,cols);
    const int offsetW = (cols - cropSize) / 2;
    const int offsetH = (rows - cropSize) / 2;
    const Rect roi(offsetW, offsetH, cropSize, cropSize);

    return img(roi);
}


void net_work(){

    auto model = torch::jit::load(weights);
    model.eval();

    std::vector<std::string> labels = load_labels(labelFile);

    std::vector<torch::jit::IValue> inputs;
    torch::Tensor in = read_image(fileName);
    inputs.push_back(in);
    torch::Tensor output = torch::softmax(model.forward(inputs).toTensor(), 1);

    std::tuple<torch::Tensor, torch::Tensor> result = torch::max(output, 1);

    torch::Tensor prob = std::get<0>(result);
    torch::Tensor index = std::get<1>(result);

    auto probability = prob.accessor<float,1>();
    auto idx = index.accessor<long,1>();

    _probability = std::to_string(probability[0]);
    _cls = labels[idx[0]];
}

const std::string& cls() const
    {
        return _cls;
    }
const std::string& prob() const
    {
        return _probability;
    }

};