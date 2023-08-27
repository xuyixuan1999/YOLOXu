#pragma once
#include <time.h>
#include<sys/time.h> 
#include <opencv2/opencv.hpp>
#include "utils.h"

double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((double)tv.tv_usec * 0.000001 + tv.tv_sec);
}


cv::Mat static_resize(cv::Mat& img) {
    float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

void generate_grids_and_stride(const int (&strides)[3], int* grid_strides)
{
    int size = sizeof(strides)/sizeof(strides[0]);
    for (int i = 0; i < size; i++)
    {
        int stride = strides[i];
        int num_grid_y = INPUT_H / stride;
        int num_grid_x = INPUT_W / stride;
        for (int g1 = 0; g1 < num_grid_y; g1++)
        {
            for (int g0 = 0; g0 < num_grid_x; g0++)
            {
                *grid_strides++ = g0;
                *grid_strides++ = g1;
                *grid_strides++ = stride;
            }
        }
    }
}

void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

void generate_yolox_proposals(int* grid_strides, int grid_strides_size, float* feat_blob, float prob_threshold, std::vector<Object>& objects)
{

    const int num_anchors = grid_strides_size;

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        // const int grid0 = grid_strides[3 * anchor_idx + 0];
        // const int grid1 = grid_strides[3 * anchor_idx + 1];
        // const int stride = grid_strides[3 * anchor_idx + 2];

        const int grid0 = *grid_strides++;
        const int grid1 = *grid_strides++;
        const int stride = *grid_strides++;

        const int basic_pos = anchor_idx * (NUM_CLASSES + 5);

        // yolox/models/yolo_head.py decode logic
        float x_center = (feat_blob[basic_pos+0] + grid0) * stride;
        float y_center = (feat_blob[basic_pos+1] + grid1) * stride;
        float w = exp(feat_blob[basic_pos+2]) * stride;
        float h = exp(feat_blob[basic_pos+3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_blob[basic_pos+4];
        for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                Object obj;
                obj.x = x0;
                obj.y = y0;
                obj.w = w;
                obj.h = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop

    } // point anchor loop
}

// void decode_outputs(float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h) {
//         std::vector<Object> proposals;
//         std::vector<int> strides = {8, 16, 32};
//         std::vector<GridAndStride> grid_strides;
//         generate_grids_and_stride(strides, grid_strides);
//         generate_yolox_proposals(grid_strides, prob,  BBOX_CONF_THRESH, proposals);
//         std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

//         qsort_descent_inplace(proposals);

//         std::vector<int> picked;
//         nms_sorted_bboxes(proposals, picked, NMS_THRESH);


//         int count = picked.size();

//         std::cout << "num of boxes: " << count << std::endl;

//         objects.resize(count);
//         for (int i = 0; i < count; i++)
//         {
//             objects[i] = proposals[picked[i]];

//             // adjust offset to original unpadded
//             float x0 = (objects[i].x) / scale;
//             float y0 = (objects[i].y) / scale;
//             float x1 = (objects[i].x + objects[i].w) / scale;
//             float y1 = (objects[i].y + objects[i].h) / scale;

//             // clip
//             x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
//             y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
//             x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
//             y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

//             objects[i].x = x0;
//             objects[i].y = y0;
//             objects[i].w = x1 - x0;
//             objects[i].h = y1 - y0;
//         }
// }

void decode_outputs(float* prob, std::vector<Object>& objects, float scale, const int img_w, const int img_h) {
        std::vector<Object> proposals;
        const int strides[] = {8, 16, 32};
        int grid_strides_size = 25200;
        int* grid_strides = new int[grid_strides_size];
        generate_grids_and_stride(strides, grid_strides);

        generate_yolox_proposals(grid_strides, grid_strides_size, prob,  BBOX_CONF_THRESH, proposals);
        std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

        qsort_descent_inplace(proposals);

        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);


        int count = picked.size();

        std::cout << "num of boxes: " << count << std::endl;

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].x) / scale;
            float y0 = (objects[i].y) / scale;
            float x1 = (objects[i].x + objects[i].w) / scale;
            float y1 = (objects[i].y + objects[i].h) / scale;

            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects[i].x = x0;
            objects[i].y = y0;
            objects[i].w = x1 - x0;
            objects[i].h = y1 - y0;
        }
        delete[] grid_strides;
}

void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].w * faceobjects[i].h;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];
            // iou 
            float iou = box_iou(a, b);
            if (iou > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}


void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, std::string f)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "Lable: %s, Conf: %.5f at %.2f %.2f %.2f %.2f\n", class_names[obj.label], obj.prob,
                obj.x, obj.y, obj.x + obj.w, obj.y + obj.h);

        cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
        float c_mean = cv::mean(color)[0];
        cv::Scalar txt_color;
        if (c_mean > 0.5){
            txt_color = cv::Scalar(0, 0, 0);
        }else{
            txt_color = cv::Scalar(255, 255, 255);
        }

        cv::rectangle(image, cv::Rect(obj.x, obj.y, obj.w, obj.h), color * 255, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = obj.x;
        int y = obj.y + 1;
        //int y = obj.rect.y - label_size.height - baseLine;
        if (y > image.rows)
            y = image.rows;
        //if (x + label_size.width > image.cols)
            //x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
    }

    cv::imwrite("det_res.jpg", image);
    fprintf(stderr, "save vis file\n");
    /* cv::imshow("image", image); */
    /* cv::waitKey(0); */
}

inline float box_iou(const Object& a, const Object& b)
{
    // cv::Rect_<float> inter = a.rect & b.rect;
    // object [x, y, w, h]
    // iou 
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x+a.w, b.x+b.w);
    float y2 = std::min(a.y+a.h, b.y+b.h);
    float w = std::max(float(0), x2 - x1);
    float h = std::max(float(0), y2 - y1);

    float inter_area = w * h;
    float union_area = a.w * a.h + b.w * b.h - inter_area;
    float iou = inter_area / union_area;

    return iou;
}

float* blobFromImage(cv::Mat& img){
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < static_cast<size_t>(channels); c++) 
    {
        for (size_t h = 0; h < static_cast<size_t>(img_h); h++) 
        {
            for (size_t w = 0; w < static_cast<size_t>(img_w); w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (float)img.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
    return blob;
}