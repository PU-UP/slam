#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "slam/feature_extractors.hpp"

using namespace slam;

void printUsage() {
    std::cout << "Usage: test_feature_extractor <image_path> [feature_type]" << std::endl;
    std::cout << "Feature types: ORB, ShiTomasi, FAST_ORB" << std::endl;
    std::cout << "Example: ./test_feature_extractor ../data/image.jpg ORB" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 1;
    }
    
    std::string image_path = argv[1];
    std::string feature_type = (argc > 2) ? argv[2] : "ORB";
    
    // Load image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Could not load image from " << image_path << std::endl;
        return 1;
    }
    
    std::cout << "Loaded image: " << image_path << " (" 
              << image.cols << "x" << image.rows << ")" << std::endl;
    
    // Create feature extractor
    FeatureExtractor::Options options;
    options.max_features = 2000;
    options.quality_level = 0.01;
    options.min_distance = 10.0;
    
    auto extractor = createFeatureExtractor(feature_type, options);
    
    if (!extractor) {
        std::cerr << "Error: Could not create feature extractor of type " << feature_type << std::endl;
        return 1;
    }
    
    // Set debug options
    ModuleBase::DebugOptions debug_opts;
    debug_opts.enable_visualization = true;
    debug_opts.save_intermediate = true;
    debug_opts.output_dir = "./debug_output";
    debug_opts.module_name = feature_type + "_extractor";
    debug_opts.wait_key_delay = 0;
    extractor->setDebugOptions(debug_opts);
    
    std::cout << "Extracting features using " << feature_type << "..." << std::endl;
    
    // Extract features
    auto features = extractor->extract(image);
    
    // Print results
    std::cout << "\nResults:" << std::endl;
    std::cout << "  Number of features: " << features.keypoints.size() << std::endl;
    std::cout << "  Descriptor size: " << features.descriptors.cols << "x" << features.descriptors.rows << std::endl;
    std::cout << "  Descriptor type: " << features.descriptors.type() << std::endl;
    std::cout << "  Extraction time: " << features.extraction_time_ms << " ms" << std::endl;
    
    if (!features.keypoints.empty()) {
        // Print some statistics
        double avg_response = 0.0;
        double min_response = std::numeric_limits<double>::max();
        double max_response = 0.0;
        
        for (const auto& kp : features.keypoints) {
            avg_response += kp.response;
            min_response = std::min(min_response, static_cast<double>(kp.response));
            max_response = std::max(max_response, static_cast<double>(kp.response));
        }
        avg_response /= features.keypoints.size();
        
        std::cout << "  Response statistics:" << std::endl;
        std::cout << "    Average: " << avg_response << std::endl;
        std::cout << "    Min: " << min_response << std::endl;
        std::cout << "    Max: " << max_response << std::endl;
        
        // Print octave distribution
        std::vector<int> octave_count(8, 0);
        for (const auto& kp : features.keypoints) {
            if (kp.octave >= 0 && kp.octave < 8) {
                octave_count[kp.octave]++;
            }
        }
        
        std::cout << "  Octave distribution:" << std::endl;
        for (size_t i = 0; i < octave_count.size(); ++i) {
            if (octave_count[i] > 0) {
                std::cout << "    Octave " << i << ": " << octave_count[i] << " features" << std::endl;
            }
        }
    }
    
    std::cout << "\nDebug files saved to: " << debug_opts.output_dir << std::endl;
    
    return 0;
}