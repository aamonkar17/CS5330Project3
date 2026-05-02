/*
Object Recognition System
Author: Ankit Amonkar & Vedant Bharvirkar

Features:
- Thresholding (FROM SCRATCH, optional Gaussian blur and ISODATA dynamic threshold)
- Morphological filtering (FROM SCRATCH: erosion, dilation, open/clean)
- Connected Components analysis (FROM SCRATCH: two-pass union-find labeling)
- Region Map Visualization (randomized color per component)
- Axis of Least Central Moment (orientation via central second-order moments)
- Axis-Aligned Bounding Box (derived from per-label min/max extents)
- Feature Extraction (percent filled, aspect ratio)
- Deep Embedding Extraction (ResNet18 via OpenCV DNN + SSD classifier)
- Screenshot capture ('s': saves original, binary, cleaned, region map)
- Feature vector training mode ('n': appends label + features to objectDB.csv)
- Embedding training mode ('e': appends label + embedding to embeddingDB.csv)
- Batch training from folder (--train <folder>: builds objectDB.csv from images)
- Batch testing with confusion matrix (--test <folder>: evaluates against objectDB.csv)
- Works in real-time (webcam) or offline (image folder / single image)
- EXTENSION: Recognizes any number of object classes — labels are discovered
              automatically from objectDB.csv, no hardcoding required.

Compile:
g++ objectRecognition.cpp -o objectRecognition `pkg-config --cflags --libs opencv4`

Run (real-time webcam):
./objectRecognition

Run (single image or glob):
./objectRecognition path/to/image.jpg

Run (batch train from labeled folders):
./objectRecognition --train path/to/train_folder

Run (batch test with confusion matrix):
./objectRecognition --test path/to/test_folder

Controls (real-time mode):
  ESC  →  quit
  s    →  save report images + print features to console
  n    →  enter label and save feature vector to objectDB.csv
  e    →  enter label and save embedding vector to embeddingDB.csv
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
#include <sstream> 
#include <map> 
#include <filesystem>
#include <cctype>
#include "utilities.h"
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

struct DBSample {          
    std::string label;
    std::vector<double> feat; // [percentFilled, aspectRatio]
};

struct CCStats {
    int area = 0;
    int minX = INT_MAX, minY = INT_MAX;
    int maxX = INT_MIN, maxY = INT_MIN;
};

struct UnionFind {
    std::vector<int> parent;

    UnionFind(int n) {
        parent.resize(n);
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    int find(int x) {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    }

    void unite(int a, int b) {
        int pa = find(a);
        int pb = find(b);
        if (pa != pb)
            parent[pb] = pa;
    }
};

struct EmbSample {
    std::string label;
    std::vector<float> emb;
};

std::vector<DBSample> gDB;
std::vector<double> gStdev;   // per-feature stdev (size 2)
bool gDBLoaded = false;

std::vector<EmbSample> gEmbDB;
bool gEmbLoaded = false;

// Discovered at load time from objectDB.csv — no hardcoding needed.
// Index i  <->  gKnownLabels[i]  for the confusion matrix.
std::vector<std::string> gKnownLabels;


// Simple global thresholding function: converts grayscale to binary
Mat thresholdImage(const Mat& gray, int T, bool useBlur = false, bool useDynamic = false)
{
    Mat img = gray.clone();

    // Gaussian blur for noise reduction
    if (useBlur)
        GaussianBlur(img, img, Size(5, 5), 0);

    int thresholdVal = T;

    // Dynamic threshold using ISODATA/k-means approximation
    // Compute dynamic threshold using ISODATA algorithm
    if (useDynamic)
    {
        int T_old = 127;
        int T_new = T_old;

        while (true)
        {
            long sum1 = 0, sum2 = 0;
            int count1 = 0, count2 = 0;

            for (int y = 0; y < img.rows; y++)
            {
                for (int x = 0; x < img.cols; x++)
                {
                    int pixel = img.at<uchar>(y, x);

                    if (pixel < T_old)
                    {
                        sum1 += pixel;
                        count1++;
                    }
                    else
                    {
                        sum2 += pixel;
                        count2++;
                    }
                }
            }

            int mean1 = count1 ? sum1 / count1 : 0;
            int mean2 = count2 ? sum2 / count2 : 0;

            T_new = (mean1 + mean2) / 2;

            if (abs(T_new - T_old) < 1)
                break;

            T_old = T_new;
        }

        thresholdVal = T_new;
    }
   

    Mat binary(img.rows, img.cols, CV_8UC1);
    for (int y = 0; y < img.rows; y++)
        for (int x = 0; x < img.cols; x++)
            binary.at<uchar>(y, x) = img.at<uchar>(y, x) < thresholdVal ? 255 : 0;

    return binary;
}

// 3x3 erosion: shrinks white regions, removes noise
Mat erodeImage(const Mat& binary)
{
    Mat output = binary.clone();
    for (int y = 1; y < binary.rows - 1; y++)
        for (int x = 1; x < binary.cols - 1; x++)
        {
            bool erode = false;
            for (int j = -1; j <= 1; j++)
                for (int i = -1; i <= 1; i++)
                    if (binary.at<uchar>(y + j, x + i) == 0)
                        erode = true;
            if (erode) output.at<uchar>(y, x) = 0;
        }
    return output;
}

// 3x3 dilation: expands white regions, fills holes
Mat dilateImage(const Mat& binary)
{
    Mat output = binary.clone();
    for (int y = 1; y < binary.rows - 1; y++)
        for (int x = 1; x < binary.cols - 1; x++)
        {
            bool dilate = false;
            for (int j = -1; j <= 1; j++)
                for (int i = -1; i <= 1; i++)
                    if (binary.at<uchar>(y + j, x + i) == 255)
                        dilate = true;
            if (dilate) output.at<uchar>(y, x) = 255;
        }
    return output;
}

// Morphological cleaning: double erosion + double dilation
Mat cleanImage(const Mat& binary)
{
    Mat eroded = erodeImage(binary);
    eroded = erodeImage(eroded); // double erosion
    Mat cleaned = dilateImage(eroded);
    cleaned = dilateImage(cleaned); // double dilation
    return cleaned;
}

// Compute raw moments for a labeled region
Moments computeMoments(const Mat& labels, int targetLabel)
{
    Moments m;
    for (int y = 0; y < labels.rows; y++)
        for (int x = 0; x < labels.cols; x++)
            if (labels.at<int>(y, x) == targetLabel)
            {
                m.m00 += 1;
                m.m10 += x;
                m.m01 += y;
                m.m11 += x * y;
                m.m20 += x * x;
                m.m02 += y * y;
            }
    return m;
}

// Draw primary axis (axis of least central moment)
void drawAxis(Mat& frame, Moments& m)
{
    if (m.m00 == 0) return;
    double cx = m.m10 / m.m00, cy = m.m01 / m.m00;
    double mu20 = m.m20 / m.m00 - cx * cx;
    double mu02 = m.m02 / m.m00 - cy * cy;
    double mu11 = m.m11 / m.m00 - cx * cy;
    double theta = 0.5 * atan2(2 * mu11, mu20 - mu02);
    Point center(cx, cy);
    Point axisEnd(cx + 150 * cos(theta), cy + 150 * sin(theta));
    line(frame, center, axisEnd, Scalar(0, 0, 255), 3);
}

// Draw axis-aligned bounding box
void drawBoundingBox(Mat& frame, const std::vector<CCStats>& stats, int label)
{
    if (label <= 0 || label >= (int)stats.size()) return;

    int x = stats[label].minX;
    int y = stats[label].minY;
    int w = stats[label].maxX - stats[label].minX + 1;
    int h = stats[label].maxY - stats[label].minY + 1;

    rectangle(frame, Rect(x, y, w, h), Scalar(0, 255, 0), 3);
}

// Generate colored region map
Mat colorRegions(const Mat& labels, int numLabels)
{
    vector<Vec3b> colors(numLabels);
    colors[0] = Vec3b(0, 0, 0);
    RNG rng(12345);
    for (int i = 1; i < numLabels; i++)
        colors[i] = Vec3b(rng.uniform(50, 255), rng.uniform(50, 255), rng.uniform(50, 255));

    Mat colorMap(labels.size(), CV_8UC3);
    for (int y = 0; y < labels.rows; y++)
        for (int x = 0; x < labels.cols; x++)
            colorMap.at<Vec3b>(y, x) = colors[labels.at<int>(y, x)];
    return colorMap;
}

int pickBestComponentScratch(
    const cv::Mat& cleaned,
    const std::vector<CCStats>& stats)
{
    int imgArea = cleaned.rows * cleaned.cols;
    int bestLabel = -1;
    int bestArea = 0;

    for (int i = 1; i < (int)stats.size(); i++) {

        int area = stats[i].area;
        if (area < 2000) continue;
        if (area > 0.35 * imgArea) continue;

        int w = stats[i].maxX - stats[i].minX + 1;
        int h = stats[i].maxY - stats[i].minY + 1;

        if (stats[i].minX <= 3 || stats[i].minY <= 3 ||
            stats[i].maxX >= cleaned.cols - 3 ||
            stats[i].maxY >= cleaned.rows - 3)
            continue;

        double aspect = (double)w / h;
        if (aspect > 5.0 || aspect < 0.2) continue;

        if (area > bestArea) {
            bestArea = area;
            bestLabel = i;
        }
    }

    return bestLabel;
}

int pickBestComponent(const Mat& cleaned, const Mat& stats, int numLabels)
{
    int bestLabel = -1;
    int bestArea = 0;

    const int imgArea = cleaned.rows * cleaned.cols;

    for (int i = 1; i < numLabels; i++)
    {
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int w = stats.at<int>(i, CC_STAT_WIDTH);
        int h = stats.at<int>(i, CC_STAT_HEIGHT);
        int area = stats.at<int>(i, CC_STAT_AREA);

        // 1) ignore tiny noise
        if (area < 2000) continue;

        // 2) ignore HUGE blobs (towel/background)
        if (area > 0.35 * imgArea) continue;

        // 3) ignore blobs touching border
        if (x <= 3 || y <= 3 || (x + w) >= cleaned.cols - 3 || (y + h) >= cleaned.rows - 3)
            continue;

        // 4) ignore extreme shapes
        double aspect = (double)w / (double)h;
        if (aspect > 5.0 || aspect < 0.2) continue;

        if (area > bestArea)
        {
            bestArea = area;
            bestLabel = i;
        }
    }
    return bestLabel;
}

int connectedComponentsFromScratch(
    const cv::Mat& binary,
    cv::Mat& labels,
    std::vector<CCStats>& stats)
{
    int rows = binary.rows;
    int cols = binary.cols;

    labels = cv::Mat::zeros(rows, cols, CV_32S);

    int nextLabel = 1;
    UnionFind uf(rows * cols / 2 + 2);

    // First Pass
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {

            if (binary.at<uchar>(y, x) == 0)
                continue;

            int left = (x > 0) ? labels.at<int>(y, x - 1) : 0;
            int up   = (y > 0) ? labels.at<int>(y - 1, x) : 0;

            if (left == 0 && up == 0) {
                labels.at<int>(y, x) = nextLabel;
                nextLabel++;
            }
            else if (left != 0 && up == 0) {
                labels.at<int>(y, x) = left;
            }
            else if (left == 0 && up != 0) {
                labels.at<int>(y, x) = up;
            }
            else {
                labels.at<int>(y, x) = std::min(left, up);
                if (left != up)
                    uf.unite(left, up);
            }
        }
    }

    // Second Pass + Stats
    std::map<int, int> labelMap;
    int newLabel = 1;

    stats.clear();
    stats.resize(nextLabel);

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {

            int lbl = labels.at<int>(y, x);
            if (lbl == 0) continue;

            int root = uf.find(lbl);

            if (labelMap.count(root) == 0)
                labelMap[root] = newLabel++;

            int finalLabel = labelMap[root];
            labels.at<int>(y, x) = finalLabel;

            CCStats& s = stats[finalLabel];
            s.area++;
            s.minX = std::min(s.minX, x);
            s.minY = std::min(s.minY, y);
            s.maxX = std::max(s.maxX, x);
            s.maxY = std::max(s.maxY, y);
        }
    }

    stats.resize(newLabel);

    return newLabel; // number of labels
}

bool extractFeaturesFromImage(const cv::Mat& inputBGR, std::vector<double>& featOut)
{
    if (inputBGR.empty()) return false;

    cv::Mat gray;
    cv::cvtColor(inputBGR, gray, cv::COLOR_BGR2GRAY);

    // same pipeline as webcam
    cv::Mat binary = thresholdImage(gray, 150, true, true);

    int white = countNonZero(binary);
    if (white > 0.5 * binary.rows * binary.cols)
    {
        // invert so object becomes the smaller white blob
        for (int y = 0; y < binary.rows; y++)
            for (int x = 0; x < binary.cols; x++)
                binary.at<uchar>(y, x) = 255 - binary.at<uchar>(y, x);
    }

    cv::Mat cleaned = cleanImage(binary);

    // Implemented connected components from scratch to get stats in a convenient struct format, the bottom two lines are the OpenCV built-in version
    // Mat labels, stats, centroids;
    // int numLabels = connectedComponentsWithStats(cleaned, labels, stats, centroids);
    // int largestLabel = pickBestComponent(cleaned, stats, numLabels);

    cv::Mat labels;
    std::vector<CCStats> stats;
    int numLabels = connectedComponentsFromScratch(cleaned, labels, stats);
    int largestLabel = pickBestComponentScratch(cleaned, stats);

    if (largestLabel == -1) return false;

    cv::Moments m = computeMoments(labels, largestLabel);

    int w = stats[largestLabel].maxX - stats[largestLabel].minX + 1;
    int h = stats[largestLabel].maxY - stats[largestLabel].minY + 1;
    if (w <= 0 || h <= 0 || m.m00 <= 0) return false;

    double percentFilled = m.m00 / (double)(w * h);
    double aspectRatio = (double)w / (double)h;

    featOut = { percentFilled, aspectRatio };
    return true;
}

void trainFromFolder(const std::string& rootFolder)
{
    std::ofstream out("objectDB.csv", std::ios::out); // overwrite
    if (!out.is_open()) {
        std::cout << "Cannot create objectDB.csv in working directory.\n";
        return;
    }

    int written = 0;

    for (const auto& classDir : fs::directory_iterator(rootFolder))
    {
        if (!classDir.is_directory()) continue;

        std::string label = classDir.path().filename().string();

        for (const auto& file : fs::directory_iterator(classDir))
        {
            if (!file.is_regular_file()) continue;

            std::string ext = file.path().extension().string();
            for (char& c : ext) c = (char)std::tolower((unsigned char)c);

            if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

            cv::Mat img = cv::imread(file.path().string());
            if (img.empty()) {
                std::cout << "Skip unreadable: " << file.path().string() << "\n";
                continue;
            }

            std::vector<double> feat;
            if (!extractFeaturesFromImage(img, feat)) {
                std::cout << "Skip (no region): " << file.path().filename().string() << "\n";
                continue;
            }

            out << label << "," << feat[0] << "," << feat[1] << "\n";
            written++;

            std::cout << "Trained: " << label << " from " << file.path().filename().string() << "\n";
        }
    }

    out.close();
    std::cout << "Training complete. Wrote " << written << " samples to objectDB.csv\n";
}

// Loads gDB, gStdev, and gKnownLabels from objectDB.csv.
// gKnownLabels is built dynamically — no class names are hardcoded anywhere.
bool loadObjectDB(const std::string& path)
{
    gDB.clear();
    gStdev.clear();
    gKnownLabels.clear();

    std::ifstream in(path);
    if (!in.is_open()) {
        std::cout << "Could not open DB file: " << path << std::endl;
        return false;
    }

    // Tracks insertion order of unique labels
    std::map<std::string, int> labelToIndex;

    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty()) continue;
        std::stringstream ss(line);

        std::string label, s1, s2;
        if (!std::getline(ss, label, ',')) continue;
        if (!std::getline(ss, s1, ',')) continue;
        if (!std::getline(ss, s2, ',')) continue;

        try {
            double f1 = std::stod(s1);
            double f2 = std::stod(s2);
            gDB.push_back({ label, {f1, f2} });

            // Register each label the first time we see it
            if (labelToIndex.find(label) == labelToIndex.end()) {
                labelToIndex[label] = (int)gKnownLabels.size();
                gKnownLabels.push_back(label);
            }
        }
        catch (...) {
            // skip bad line
        }
    }

    if (gDB.empty()) {
        std::cout << "DB is empty after reading." << std::endl;
        return false;
    }

    // compute stdev for each feature (2 features)
    const int d = 2;
    std::vector<double> mean(d, 0.0);
    for (auto& s : gDB) {
        for (int i = 0; i < d; i++) mean[i] += s.feat[i];
    }
    for (int i = 0; i < d; i++) mean[i] /= (double)gDB.size();

    std::vector<double> var(d, 0.0);
    for (auto& s : gDB) {
        for (int i = 0; i < d; i++) {
            double diff = s.feat[i] - mean[i];
            var[i] += diff * diff;
        }
    }

    gStdev.resize(d, 1.0);
    for (int i = 0; i < d; i++) {
        var[i] /= (double)gDB.size();
        gStdev[i] = std::sqrt(var[i]);
        if (gStdev[i] < 1e-6) gStdev[i] = 1e-6;
    }

    std::cout << "Loaded " << gDB.size() << " samples, "
              << gKnownLabels.size() << " classes: ";
    for (int i = 0; i < (int)gKnownLabels.size(); i++)
        std::cout << gKnownLabels[i]
                  << (i < (int)gKnownLabels.size() - 1 ? ", " : "\n");

    return true;
}

double scaledEuclidean(const std::vector<double>& a, const std::vector<double>& b)
{
    double sum = 0.0;
    for (int i = 0; i < (int)a.size(); i++) {
        double z = (a[i] - b[i]) / gStdev[i];
        sum += z * z;
    }
    return std::sqrt(sum);
}

std::pair<std::string, double> predictLabel(const std::vector<double>& x)
{
    std::string bestLabel = "UNKNOWN";
    double bestDist = 1e18;

    for (auto& s : gDB) {
        double d = scaledEuclidean(x, s.feat);
        if (d < bestDist) {
            bestDist = d;
            bestLabel = s.label;
        }
    }
    return { bestLabel, bestDist };
}

// Returns the index of s in gKnownLabels, or -1 if not present.
// Works for any number of classes — nothing is hardcoded here.
int labelIndex(const std::string& s)
{
    for (int i = 0; i < (int)gKnownLabels.size(); i++)
        if (s == gKnownLabels[i]) return i;
    return -1;
}

static double computeThetaFromMoments(const cv::Moments& m)
{
    if (m.m00 <= 0) return 0.0;

    double cx = m.m10 / m.m00;
    double cy = m.m01 / m.m00;

    double mu20 = m.m20 / m.m00 - cx * cx;
    double mu02 = m.m02 / m.m00 - cy * cy;
    double mu11 = m.m11 / m.m00 - cx * cy;

    return 0.5 * atan2(2 * mu11, mu20 - mu02); // radians
}

static void computeExtentsE1E2(const cv::Mat& labels, int targetLabel,
    double cx, double cy, double theta,
    float& minE1, float& maxE1,
    float& minE2, float& maxE2)
{
    double c = cos(theta), s = sin(theta); // E1=(c,s), E2=(-s,c)

    minE1 = minE2 = 1e9f;
    maxE1 = maxE2 = -1e9f;

    for (int y = 0; y < labels.rows; y++) {
        const int* row = labels.ptr<int>(y);
        for (int x = 0; x < labels.cols; x++) {
            if (row[x] != targetLabel) continue;

            double dx = x - cx;
            double dy = y - cy;

            float p1 = (float)(dx * c + dy * s);
            float p2 = (float)(-dx * s + dy * c);

            minE1 = std::min(minE1, p1);
            maxE1 = std::max(maxE1, p1);
            minE2 = std::min(minE2, p2);
            maxE2 = std::max(maxE2, p2);
        }
    }
}

// -------------------- EMBEDDING DB + SSD CLASSIFIER --------------------
static void appendEmbeddingRow(const std::string& label, const cv::Mat& embedding,
    const std::string& path = "embeddingDB.csv")
{
    // embedding should be 1 x D (CV_32F)
    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) {
        std::cout << "ERROR: could not open " << path << " for writing.\n";
        return;
    }

    out << label;
    for (int i = 0; i < embedding.cols; i++) {
        out << "," << embedding.at<float>(0, i);
    }
    out << "\n";
}

static bool loadEmbeddingDB(const std::string& path, std::vector<EmbSample>& db)
{
    db.clear();
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cout << "Could not open embedding DB: " << path << "\n";
        return false;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        EmbSample s;

        if (!std::getline(ss, s.label, ',')) continue;

        std::string tok;
        while (std::getline(ss, tok, ',')) {
            try {
                s.emb.push_back((float)std::stof(tok));
            }
            catch (...) {
                // skip bad token
            }
        }

        if (!s.label.empty() && !s.emb.empty())
            db.push_back(std::move(s));
    }

    std::cout << "Loaded " << db.size() << " embeddings from " << path << "\n";
    return !db.empty();
}

static double ssdDistance(const cv::Mat& embedding, const std::vector<float>& ref)
{
    int D = embedding.cols;
    if ((int)ref.size() != D) return 1e18;

    const float* p = embedding.ptr<float>(0);

    double sum = 0.0;
    for (int i = 0; i < D; i++) {
        double d = (double)p[i] - (double)ref[i];
        sum += d * d;
    }
    return sum; // SSD (no sqrt needed)
}

static std::pair<std::string, double> predictEmbeddingLabel(const cv::Mat& embedding,
    const std::vector<EmbSample>& db)
{
    std::string bestLabel = "UNKNOWN";
    double bestDist = 1e18;

    for (const auto& s : db) {
        double d = ssdDistance(embedding, s.emb);
        if (d < bestDist) {
            bestDist = d;
            bestLabel = s.label;
        }
    }
    return { bestLabel, bestDist };
}

// Process a single frame/image
void processFrame(Mat& frame, cv::dnn::Net& net, bool isTrainingMode = false)
{
    cv::Mat embedding;
    bool haveEmbedding = false;

    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    Mat binary = thresholdImage(gray, 150, true, true);

    int white = countNonZero(binary);
    if (white > 0.5 * binary.rows * binary.cols)
    {
        // invert so object becomes the smaller white blob
        for (int y = 0; y < binary.rows; y++)
            for (int x = 0; x < binary.cols; x++)
                binary.at<uchar>(y, x) = 255 - binary.at<uchar>(y, x);
    }

    Mat cleaned = cleanImage(binary);

    // Implemented connected components from scratch to get stats in a convenient struct format, the bottom two lines are the OpenCV built-in version
    // Mat labels, stats, centroids;
    // int numLabels = connectedComponentsWithStats(cleaned, labels, stats, centroids);
    // int largestLabel = pickBestComponent(cleaned, stats, numLabels);

    cv::Mat labels;
    std::vector<CCStats> stats;
    int numLabels = connectedComponentsFromScratch(cleaned, labels, stats);
    int largestLabel = pickBestComponentScratch(cleaned, stats);

    Mat regionMap = colorRegions(labels, numLabels);

    double percentFilled = 0, aspectRatio = 0;
    if (largestLabel > 0 && largestLabel < (int)stats.size())
    {
        Moments m = computeMoments(labels, largestLabel);
        if (m.m00 <= 0) {
            cout << "Moment area zero, skipping.\n";
            return;
        }
        drawAxis(frame, m);
        drawBoundingBox(frame, stats, largestLabel);
        double cx = m.m10 / m.m00, cy = m.m01 / m.m00;

        double theta = computeThetaFromMoments(m);

        float minE1, maxE1, minE2, maxE2;
        computeExtentsE1E2(labels, largestLabel, cx, cy, theta, minE1, maxE1, minE2, maxE2);

        // Optional safety: ensure minE2 < maxE2
        if (minE2 > maxE2) std::swap(minE2, maxE2);

        cv::Mat embROI;
        prepEmbeddingImage(frame, embROI,
            (int)cx, (int)cy,
            (float)theta,
            minE1, maxE1, minE2, maxE2,
            0);

        if (!embROI.empty())
        {
            getEmbedding(embROI, embedding, net, 0);
            haveEmbedding = !embedding.empty();
        }

        if (gEmbLoaded && !embedding.empty())
        {
            auto pred = predictEmbeddingLabel(embedding, gEmbDB);
            std::string msg = "EmbPred: " + pred.first + "  ssd=" + std::to_string(pred.second);
            putText(frame, msg, Point(20, 150), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        }

        circle(frame, Point(cx, cy), 5, Scalar(255, 0, 0), -1);
        int w = stats[largestLabel].maxX - stats[largestLabel].minX + 1;
        int h = stats[largestLabel].maxY - stats[largestLabel].minY + 1;
        percentFilled = m.m00 / (double)(w * h);
        aspectRatio = (double)w / h;

        if (gDBLoaded)
        {
            std::vector<double> x = { percentFilled, aspectRatio };
            auto pred = predictLabel(x);

            // Optional unknown threshold (tune later)
            std::string label = pred.first;
            double dist = pred.second;
            if (dist > 3.0) label = "UNKNOWN";

            std::string msg = "Pred: " + label + "  d=" + std::to_string(dist);
            putText(frame, msg, Point(20, 110), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
        }

        string text1 = "Filled: " + to_string(percentFilled);
        string text2 = "Aspect: " + to_string(aspectRatio);
        putText(frame, text1, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);
        putText(frame, text2, Point(20, 70), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);
    }

    imshow("Original", frame);
    imshow("Binary", binary);
    imshow("Cleaned", cleaned);
    imshow("Region Map", regionMap);

    char key;

    if (isTrainingMode)
        key = (char)waitKey(1);   // webcam mode
    else
        key = (char)waitKey(0);   // image mode (block)

    if (key == 27) exit(0);

    // Save images
    if (key == 's')
    {
        imwrite("original.png", frame);
        imwrite("binary.png", binary);
        imwrite("cleaned.png", cleaned);
        imwrite("regionMap.png", regionMap);
        cout<<"Percent Filled: "<<percentFilled<<endl;
        cout<<"Aspect Ratio: "<<aspectRatio<<endl;
        cout << "Saved report images" << endl;
    }

    // Training mode
    if (isTrainingMode && largestLabel != -1 && key == 'n')
    {
        string label;
        cout << "Enter object label: ";
        cin >> label;
        ofstream out("objectDB.csv", ios::app);
        out << label << "," << percentFilled << "," << aspectRatio << endl;
        out.close();
        // Keep gKnownLabels in sync without a full reload
        if (labelIndex(label) == -1)
            gKnownLabels.push_back(label);
        cout << "Saved feature vector for " << label << endl;
    }

    if (key == 'e' && haveEmbedding)
    {
        std::string label;
        std::cout << "Enter label for embedding sample: ";
        std::cin >> label;

        appendEmbeddingRow(label, embedding);
        std::cout << "Saved embedding row for " << label << "\n";

        // reload DB immediately so it starts working without restart
        gEmbLoaded = loadEmbeddingDB("embeddingDB.csv", gEmbDB);
    }
}

// Confusion matrix rows/cols are sized dynamically from gKnownLabels.
// Adding new classes never requires editing this function.
void testFromFolder(const std::string& rootFolder)
{
    const int K = (int)gKnownLabels.size();
    if (K == 0) { std::cout << "No known labels — load DB first.\n"; return; }

    std::vector<std::vector<int>> M(K, std::vector<int>(K, 0));

    int total = 0, correct = 0;

    for (const auto& classDir : fs::directory_iterator(rootFolder))
    {
        if (!classDir.is_directory()) continue;

        std::string trueLabel = classDir.path().filename().string();
        int ti = labelIndex(trueLabel);
        if (ti < 0) {
            std::cout << "Skip unknown class folder: " << trueLabel << "\n";
            continue;
        }

        for (const auto& file : fs::directory_iterator(classDir))
        {
            if (!file.is_regular_file()) continue;

            std::string ext = file.path().extension().string();
            for (char& c : ext) c = (char)std::tolower((unsigned char)c);
            if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

            cv::Mat img = cv::imread(file.path().string());
            if (img.empty()) continue;

            std::vector<double> feat;
            if (!extractFeaturesFromImage(img, feat)) {
                std::cout << "No region in: " << file.path().filename().string() << "\n";
                continue;
            }

            auto pred = predictLabel(feat);
            std::string predLabel = pred.first;

            int pi = labelIndex(predLabel);
            if (pi < 0) {
                std::cout << "Pred UNKNOWN for: " << file.path().filename().string() << "\n";
                continue;
            }

            M[ti][pi]++;

            total++;
            if (ti == pi) correct++;

            std::cout << trueLabel << " -> " << predLabel
                << "   d=" << pred.second
                << "   (" << file.path().filename().string() << ")\n";
        }
    }

    std::cout << "\nAccuracy: " << correct << "/" << total
        << " = " << (total ? (100.0 * correct / total) : 0.0) << "%\n\n";

    // Print confusion matrix with dynamic header
    std::cout << "Confusion Matrix (rows=true, cols=pred):\n";
    std::cout << std::string(14, ' ');
    for (const auto& n : gKnownLabels)
        std::cout << n.substr(0, 9) << "\t";
    std::cout << "\n";

    for (int r = 0; r < K; r++) {
        std::string rowName = gKnownLabels[r];
        rowName.resize(13, ' ');
        std::cout << rowName << " ";
        for (int c = 0; c < K; c++)
            std::cout << M[r][c] << (c == K - 1 ? "" : "\t");
        std::cout << "\n";
    }

    // Save to CSV with dynamic column headers
    std::ofstream out("confusion.csv");
    out << "true/pred";
    for (const auto& n : gKnownLabels) out << "," << n;
    out << "\n";
    for (int r = 0; r < K; r++) {
        out << gKnownLabels[r];
        for (int c = 0; c < K; c++) out << "," << M[r][c];
        out << "\n";
    }
    out.close();
    std::cout << "Saved confusion.csv\n";
}

int main(int argc, char** argv)
{
    // TRAIN MODE (phone images)
    if (argc == 3 && std::string(argv[1]) == "--train")
    {
        trainFromFolder(argv[2]);
        return 0;
    }

    // TEST MODE
    if (argc == 3 && std::string(argv[1]) == "--test")
    {
        gDBLoaded = loadObjectDB("objectDB.csv");
        if (!gDBLoaded) return -1;

        testFromFolder(argv[2]);
        return 0;
    }

    // NORMAL MODE (webcam / classification)
    gDBLoaded = loadObjectDB("objectDB.csv");

    cv::dnn::Net net = cv::dnn::readNetFromONNX("resnet18-v2-7.onnx");
    if (net.empty()) {
        std::cout << "ERROR: Could not load resnet18-v2-7.onnx\n";
        return -1;
    }
    gEmbLoaded = loadEmbeddingDB("embeddingDB.csv", gEmbDB);

    // Webcam mode
    if (argc == 1)
    {
        VideoCapture cap(0);
        if (!cap.isOpened()) { cout << "Error opening camera" << endl; return -1; }

        cout << "Press 's' to save report images" << endl;
        cout << "Press 'n' to save feature vector (training mode)" << endl;
        cout << "Press 'e' to save embedding sample" << endl;
        cout << "Press ESC to quit" << endl;

        while (true)
        {
            Mat frame;
            cap >> frame;
            if (frame.empty()) break;

            processFrame(frame, net, true);
        }
        cap.release();
    }
    else
    {
        // Optional: process images passed by glob pattern
        vector<String> imageFiles;
        glob(argv[1], imageFiles);

        for (const auto& file : imageFiles)
        {
            Mat frame = imread(file);
            if (frame.empty()) continue;
            processFrame(frame, net, false);
        }
    }

    return 0;
}