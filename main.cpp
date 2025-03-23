/*****************************************************************************
 * Amir Mohammad Parvizi: Advanced Traffic Monitoring and Speed Control System
 * Version: 1.0.0
 * 
 * Developed for advanced traffic monitoring, vehicle detection,
 * speed measurement, and violation recording using state-of-the-art
 * computer vision and deep learning techniques.
 * 
 * Supports NVIDIA Jetson, Intel NCS2, TPU, DSPs, and custom ASICs
 * with hardware acceleration for real-time processing.
 * 
 * © 2025 Advanced Traffic Systems, Inc.
 *****************************************************************************/

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <set>
#include <queue>
#include <deque>
#include <list>
#include <stack>
#include <algorithm>
#include <numeric>
#include <functional>
#include <memory>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <regex>
#include <random>
#include <cmath>
#include <ctime>
#include <cstring>
#include <csignal>
#include <limits>
#include <optional>
#include <variant>
#include <any>
#include <tuple>
#include <type_traits>
#include <cassert>

// External libraries
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudafeatures2d.hpp>

// TensorFlow C++ API
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

// MQTT for IoT communication
#include <mosquitto.h>

// JSON parsing
#include <nlohmann/json.hpp>

// Database connectivity
#include <sqlite3.h>
#include <pqxx/pqxx>

// GPS and mapping
#include <proj.h>
#include <gdal/ogr_spatialref.h>

// System-specific headers
#ifdef _WIN32
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <netdb.h>
#include <fcntl.h>
#include <termios.h>
#endif

// Shorthand for convenience
namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace std::chrono_literals;

/******************************************************************************
 * Constants and Configuration
 ******************************************************************************/

// System version information
const std::string VERSION = "5.3.7";
const std::string BUILD_DATE = __DATE__;
const std::string BUILD_TIME = __TIME__;

// Default configuration values
const int DEFAULT_CAMERA_WIDTH = 3840;
const int DEFAULT_CAMERA_HEIGHT = 2160;
const int DEFAULT_CAMERA_FPS = 60;
const std::string DEFAULT_LOG_LEVEL = "INFO";
const std::string DEFAULT_CONFIG_FILE = "config/speedguardian.json";
const std::string DEFAULT_MODEL_PATH = "models/";
const std::string DEFAULT_OUTPUT_PATH = "output/";
const std::string DEFAULT_DATABASE_PATH = "data/violations.db";
const int DEFAULT_VIOLATION_BUFFER_SIZE = 100;
const int DEFAULT_PROCESSING_THREADS = 8;
const int DEFAULT_GPU_ID = 0;
const float DEFAULT_SPEED_LIMIT = 50.0f; // km/h
const float DEFAULT_SPEED_TOLERANCE = 3.0f; // km/h
const float DEFAULT_TRACKING_CONFIDENCE = 0.7f;
const float DEFAULT_DETECTION_THRESHOLD = 0.65f;
const int DEFAULT_MAX_VEHICLES_TRACK = 50;
const int DEFAULT_CAMERA_CALIBRATION_FRAMES = 100;
const int DEFAULT_LICENSE_PLATE_MIN_CHARS = 5;
const int DEFAULT_LICENSE_PLATE_MAX_CHARS = 10;
const bool DEFAULT_ENABLE_ANALYTICS = true;
const bool DEFAULT_ENABLE_CLOUD_UPLOAD = true;
const bool DEFAULT_ENABLE_ENCRYPTION = true;
const int DEFAULT_RETENTION_DAYS = 90;
const int DEFAULT_NIGHT_MODE_THRESHOLD = 40;
const double DEFAULT_GPS_LAT = 35.6895;
const double DEFAULT_GPS_LON = 139.6917;
const double DEFAULT_CAMERA_HEIGHT_METERS = 4.5;
const double DEFAULT_CAMERA_ANGLE_DEGREES = 30.0;
const int DEFAULT_RADAR_DETECTION_RANGE = 150; // meters
const int DEFAULT_LIDAR_DETECTION_RANGE = 250; // meters
const bool DEFAULT_USE_AI_ENHANCEMENT = true;
const int DEFAULT_COMPRESSION_QUALITY = 95;
const int DEFAULT_DEVICE_ID = 1001;
const std::string DEFAULT_DEVICE_AUTH_KEY = "SpeedGuardian-DeviceAuth";
const std::string DEFAULT_NTP_SERVER = "pool.ntp.org";
const bool DEFAULT_ENABLE_PTZ = false;
const int DEFAULT_PTZ_PRESET_HOME = 1;
const bool DEFAULT_ENABLE_DAY_NIGHT_SWITCHING = true;
const float DEFAULT_PERSPECTIVE_CORRECTION_ALPHA = 0.8f;
const int DEFAULT_ROLLING_BUFFER_SIZE = 120; // frames
const float DEFAULT_MINIMUM_VEHICLE_SIZE = 0.01f; // percentage of frame area
const float DEFAULT_MAXIMUM_VEHICLE_SIZE = 0.2f; // percentage of frame area
const bool DEFAULT_ENABLE_MOTION_ESTIMATION = true;
const int DEFAULT_HEALTH_CHECK_INTERVAL = 300; // seconds
const int DEFAULT_AUTO_CALIBRATION_INTERVAL = 86400; // seconds (24 hours)
const bool DEFAULT_ENABLE_SELF_DIAGNOSTICS = true;
const int DEFAULT_NETWORK_TIMEOUT = 30; // seconds
const bool DEFAULT_ENABLE_HDR = true;
const bool DEFAULT_ENABLE_DEFOGGING = true;
const int DEFAULT_VEHICLE_CLASSIFICATION_CLASSES = 12;
const float DEFAULT_WEATHER_ADAPTATION_LEVEL = 0.75f;
const int DEFAULT_UDP_STREAM_PORT = 5600;
const bool DEFAULT_ENABLE_H265 = true;
const bool DEFAULT_SECURE_BOOT_VERIFICATION = true;
const int DEFAULT_FRAME_BUFFER_SIZE = 5000;
const int DEFAULT_HEARTBEAT_INTERVAL = 60; // seconds
const bool DEFAULT_PRIVACY_MASK_ENABLED = true;
const int DEFAULT_MIN_FRAMES_FOR_VIOLATION = 5;
const float DEFAULT_LANE_CONFIDENCE_THRESHOLD = 0.8f;
const int DEFAULT_TRAFFIC_ANALYTICS_INTERVAL = 300; // seconds (5 minutes)

// Enum for violation types
enum class ViolationType {
    SPEED_VIOLATION,
    RED_LIGHT_VIOLATION,
    UNSAFE_DISTANCE,
    IMPROPER_LANE_CHANGE,
    NO_HELMET_MOTORCYCLE,
    SEATBELT_VIOLATION,
    MOBILE_PHONE_USAGE,
    WRONG_WAY_DRIVING,
    UNAUTHORIZED_VEHICLE_TYPE,
    DOUBLE_PARKING,
    NO_ENTRY_VIOLATION,
    BUS_LANE_VIOLATION,
    CUSTOM_VIOLATION
};

// Enum for camera types
enum class CameraType {
    FIXED,
    PTZ,
    THERMAL,
    INFRARED,
    MULTI_SENSOR,
    FISHEYE,
    LIDAR_INTEGRATED,
    RADAR_INTEGRATED,
    SPECIALIZED
};

// Enum for vehicle types
enum class VehicleType {
    UNKNOWN,
    MOTORCYCLE,
    CAR,
    SUV,
    VAN,
    PICKUP_TRUCK,
    BUS,
    TRUCK_SMALL,
    TRUCK_LARGE,
    SEMI_TRAILER,
    BICYCLE,
    EMERGENCY_VEHICLE,
    MILITARY_VEHICLE,
    CONSTRUCTION_VEHICLE,
    AGRICULTURAL_VEHICLE,
    SPECIAL_VEHICLE
};

// Enum for weather conditions
enum class WeatherCondition {
    CLEAR,
    CLOUDY,
    RAINY,
    SNOWY,
    FOGGY,
    SANDSTORM,
    HAIL,
    THUNDERSTORM,
    EXTREME_HEAT,
    EXTREME_COLD,
    UNKNOWN
};

// Enum for time of day
enum class TimeOfDay {
    DAWN,
    MORNING,
    NOON,
    AFTERNOON,
    DUSK,
    EVENING,
    NIGHT,
    LATE_NIGHT
};

// Enum for system operating modes
enum class OperatingMode {
    NORMAL,
    CALIBRATION,
    DIAGNOSTICS,
    ENERGY_SAVING,
    MAINTENANCE,
    DEBUG,
    STANDBY,
    EMERGENCY,
    SIMULATION
};

// Enum for detection algorithms
enum class DetectionAlgorithm {
    YOLO_V4,
    YOLO_V5,
    SSD,
    FASTER_RCNN,
    EFFICIENTDET,
    RETINA_NET,
    MASK_RCNN,
    CENTERNET,
    CUSTOM_DNN,
    CASCADE_CLASSIFIER,
    HYBRID_APPROACH
};

// Enum for tracking algorithms
enum class TrackingAlgorithm {
    DEEP_SORT,
    SORT,
    MOSSE,
    KCF,
    CSRT,
    MEDIANFLOW,
    TLD,
    GOTURN,
    CUSTOM_TRACKER,
    MULTI_TRACKER_FUSION
};

// Enum for log levels
enum class LogLevel {
    TRACE,
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL,
    NONE
};

// Enum for hardware acceleration
enum class HardwareAcceleration {
    NONE,
    CUDA,
    OPENCL,
    TRT,
    NPU,
    DSP,
    FPGA,
    CUSTOM_ASIC,
    EDGE_TPU,
    CUDA_FP16,
    CPU_OPTIMIZED
};

/******************************************************************************
 * Structures and Classes
 ******************************************************************************/

// Forward declarations
class SpeedCamera;
class VehicleDetector;
class SpeedCalculator;
class PlateRecognizer;
class TrafficAnalytics;
class ViolationProcessor;
class CameraCalibration;
class DataStorage;
class NetworkManager;
class SystemMonitor;
class WeatherAnalyzer;
class UserInterface;
class ConfigManager;
class LogManager;
class EventDispatcher;
class TimeSynchronizer;
class SecurityManager;
class DiagnosticsManager;
class CameraController;
class IlluminationController;

// GPS coordinate structure
struct GPSCoordinate {
    double latitude;
    double longitude;
    double altitude;
    double accuracy;
    std::chrono::system_clock::time_point timestamp;
    
    GPSCoordinate(double lat = 0.0, double lon = 0.0, double alt = 0.0, double acc = 0.0)
        : latitude(lat), longitude(lon), altitude(alt), accuracy(acc),
          timestamp(std::chrono::system_clock::now()) {}
    
    double distanceTo(const GPSCoordinate& other) const {
        constexpr double EARTH_RADIUS = 6371000.0; // meters
        
        double lat1Rad = latitude * M_PI / 180.0;
        double lat2Rad = other.latitude * M_PI / 180.0;
        double deltaLatRad = (other.latitude - latitude) * M_PI / 180.0;
        double deltaLonRad = (other.longitude - longitude) * M_PI / 180.0;
        
        double a = std::sin(deltaLatRad / 2) * std::sin(deltaLatRad / 2) +
                  std::cos(lat1Rad) * std::cos(lat2Rad) *
                  std::sin(deltaLonRad / 2) * std::sin(deltaLonRad / 2);
        double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1 - a));
        return EARTH_RADIUS * c;
    }
    
    std::string toString() const {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(6) << latitude << "," << longitude;
        return ss.str();
    }
};

// Vehicle detection structure
struct VehicleDetection {
    int id;
    cv::Rect boundingBox;
    VehicleType type;
    float confidence;
    std::chrono::system_clock::time_point timestamp;
    cv::Mat appearanceFeatures;
    std::vector<cv::Point2f> keypoints;
    cv::Scalar color;  // Dominant color in BGR
    std::string colorName;
    bool hasFrontPlate;
    bool hasRearPlate;
    std::string make;
    std::string model;
    int modelYear;
    float size;  // Estimated size in meters
    
    VehicleDetection() : id(-1), type(VehicleType::UNKNOWN), confidence(0.0f),
                         timestamp(std::chrono::system_clock::now()),
                         hasFrontPlate(false), hasRearPlate(false),
                         modelYear(0), size(0.0f) {}
};

// License plate detection structure
struct PlateDetection {
    cv::Rect boundingBox;
    std::string plateNumber;
    float confidence;
    std::chrono::system_clock::time_point timestamp;
    cv::Mat plateImage;
    std::string region;
    std::string country;
    std::vector<std::string> characterConfidences;
    bool isTemporary;
    bool isSpecial;
    std::string plateColor;
    std::string plateType;
    std::vector<cv::Point2f> corners;  // For skew correction
    
    PlateDetection() : confidence(0.0f), timestamp(std::chrono::system_clock::now()),
                      isTemporary(false), isSpecial(false) {}
};

// Vehicle tracking structure
struct VehicleTrack {
    int trackId;
    std::deque<VehicleDetection> detectionHistory;
    std::deque<cv::Point2f> positionHistory;
    std::deque<std::chrono::system_clock::time_point> timestampHistory;
    float currentSpeed;  // km/h
    float averageSpeed;  // km/h
    float maxSpeed;      // km/h
    std::chrono::milliseconds trackDuration;
    bool isActive;
    cv::KalmanFilter kalmanFilter;
    std::optional<PlateDetection> bestPlateDetection;
    float trackConfidence;
    std::map<int, float> lanePositions;  // <lane_id, confidence>
    cv::Mat trajectory;  // Visualization of path
    GPSCoordinate estimatedGpsPosition;
    int consecutiveDetections;
    int consecutiveMisses;
    std::unique_ptr<cv::Tracker> dedicatedTracker;
    cv::Scalar color;  // For visualization
    float trustworthiness;  // 0.0 to 1.0
    
    VehicleTrack(int id, const VehicleDetection& initialDetection)
        : trackId(id), currentSpeed(0.0f), averageSpeed(0.0f), maxSpeed(0.0f),
          trackDuration(0ms), isActive(true), trackConfidence(1.0f),
          consecutiveDetections(1), consecutiveMisses(0), trustworthiness(0.5f) {
        
        detectionHistory.push_back(initialDetection);
        positionHistory.push_back(cv::Point2f(
            initialDetection.boundingBox.x + initialDetection.boundingBox.width / 2.0f,
            initialDetection.boundingBox.y + initialDetection.boundingBox.height
        ));
        timestampHistory.push_back(initialDetection.timestamp);
        
        // Initialize Kalman filter for position and velocity tracking (x, y, vx, vy)
        kalmanFilter.init(4, 2, 0);
        kalmanFilter.transitionMatrix = (cv::Mat_<float>(4, 4) <<
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);
        
        cv::setIdentity(kalmanFilter.measurementMatrix);
        cv::setIdentity(kalmanFilter.processNoiseCov, cv::Scalar::all(1e-5));
        cv::setIdentity(kalmanFilter.measurementNoiseCov, cv::Scalar::all(1e-1));
        cv::setIdentity(kalmanFilter.errorCovPost, cv::Scalar::all(1));
        
        // Random color for visualization
        color = cv::Scalar(
            std::rand() % 255,
            std::rand() % 255,
            std::rand() % 255
        );
    }
    
    void update(const VehicleDetection& detection) {
        detectionHistory.push_back(detection);
        if (detectionHistory.size() > 30) {  // Keep last 30 detections
            detectionHistory.pop_front();
        }
        
        cv::Point2f center(
            detection.boundingBox.x + detection.boundingBox.width / 2.0f,
            detection.boundingBox.y + detection.boundingBox.height
        );
        
        positionHistory.push_back(center);
        if (positionHistory.size() > 60) {  // Keep last 60 positions
            positionHistory.pop_front();
        }
        
        timestampHistory.push_back(detection.timestamp);
        if (timestampHistory.size() > 60) {
            timestampHistory.pop_front();
        }
        
        // Update Kalman filter
        cv::Mat prediction = kalmanFilter.predict();
        cv::Mat measurement = (cv::Mat_<float>(2, 1) << center.x, center.y);
        kalmanFilter.correct(measurement);
        
        // Update tracking statistics
        consecutiveDetections++;
        consecutiveMisses = 0;
        trackConfidence = std::min(1.0f, trackConfidence + 0.05f);
        
        // Calculate duration
        if (timestampHistory.size() >= 2) {
            trackDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
                timestampHistory.back() - timestampHistory.front()
            );
        }
        
        // Gradually increase trustworthiness with each successful update
        trustworthiness = std::min(1.0f, trustworthiness + 0.01f);
    }
    
    cv::Point2f predictNextPosition() const {
        cv::Mat prediction = kalmanFilter.predict();
        return cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
    }
    
    void updateSpeed(float speed) {
        currentSpeed = speed;
        maxSpeed = std::max(maxSpeed, speed);
        
        // Update average speed
        float totalSpeed = averageSpeed * (detectionHistory.size() - 1) + speed;
        averageSpeed = totalSpeed / detectionHistory.size();
    }
    
    bool isSpeedViolation(float speedLimit, float tolerance) const {
        return currentSpeed > (speedLimit + tolerance);
    }
    
    float getAcceleration() const {
        if (timestampHistory.size() < 3) return 0.0f;
        
        size_t last = timestampHistory.size() - 1;
        size_t middle = timestampHistory.size() - 2;
        size_t first = timestampHistory.size() - 3;
        
        std::chrono::duration<float> t1 = 
            std::chrono::duration_cast<std::chrono::duration<float>>(
                timestampHistory[middle] - timestampHistory[first]);
        
        std::chrono::duration<float> t2 = 
            std::chrono::duration_cast<std::chrono::duration<float>>(
                timestampHistory[last] - timestampHistory[middle]);
        
        float v1 = calculateSpeedBetweenPoints(first, middle);
        float v2 = calculateSpeedBetweenPoints(middle, last);
        
        return (v2 - v1) / ((t1.count() + t2.count()) / 2.0f);
    }
    
private:
    float calculateSpeedBetweenPoints(size_t idx1, size_t idx2) const {
        if (idx1 >= positionHistory.size() || idx2 >= positionHistory.size() ||
            idx1 >= timestampHistory.size() || idx2 >= timestampHistory.size())
            return 0.0f;
            
        float distance = cv::norm(positionHistory[idx2] - positionHistory[idx1]);
        std::chrono::duration<float> timeDiff = 
            std::chrono::duration_cast<std::chrono::duration<float>>(
                timestampHistory[idx2] - timestampHistory[idx1]);
                
        if (timeDiff.count() > 0)
            return distance / timeDiff.count();
        return 0.0f;
    }
};

// Traffic lane structure
struct TrafficLane {
    int laneId;
    std::vector<cv::Point> lanePolygon;
    std::vector<cv::Point> leftBoundary;
    std::vector<cv::Point> rightBoundary;
    std::vector<cv::Point> centerLine;
    float speedLimit;  // km/h
    std::string laneType;  // "normal", "bus", "carpool", "emergency", etc.
    bool isActive;
    std::string direction;  // "inbound", "outbound"
    float width;  // meters
    float length;  // meters visible in frame
    float averageSpeed;  // km/h
    int vehicleCount;
    float occupancyRate;  // 0.0 to 1.0
    std::map<VehicleType, int> vehicleTypeDistribution;
    bool enforceSpeedLimit;
    
    TrafficLane(int id) : laneId(id), speedLimit(DEFAULT_SPEED_LIMIT), isActive(true),
                         width(0.0f), length(0.0f), averageSpeed(0.0f),
                         vehicleCount(0), occupancyRate(0.0f), enforceSpeedLimit(true) {}
    
    bool containsPoint(const cv::Point& point) const {
        return cv::pointPolygonTest(lanePolygon, point, false) >= 0;
    }
    
    void updateStatistics(float speed, VehicleType type) {
        vehicleCount++;
        averageSpeed = (averageSpeed * (vehicleCount - 1) + speed) / vehicleCount;
        vehicleTypeDistribution[type]++;
    }
};

// Traffic signal state structure
struct TrafficSignalState {
    enum class SignalState {
        RED,
        YELLOW,
        GREEN,
        RED_YELLOW,
        FLASHING_YELLOW,
        OFF
    };
    
    SignalState currentState;
    std::chrono::system_clock::time_point stateStartTime;
    std::chrono::system_clock::time_point stateEndTime;
    cv::Rect signalLocation;
    float detectionConfidence;
    bool isValid;
    int signalGroupId;
    
    TrafficSignalState() : currentState(SignalState::OFF), 
                          stateStartTime(std::chrono::system_clock::now()),
                          stateEndTime(std::chrono::system_clock::now()),
                          detectionConfidence(0.0f), isValid(false), signalGroupId(0) {}
    
    bool isRed() const {
        return currentState == SignalState::RED || currentState == SignalState::RED_YELLOW;
    }
    
    std::chrono::seconds getDuration() const {
        return std::chrono::duration_cast<std::chrono::seconds>(
            stateEndTime - stateStartTime);
    }
    
    std::string toString() const {
        switch (currentState) {
            case SignalState::RED: return "RED";
            case SignalState::YELLOW: return "YELLOW";
            case SignalState::GREEN: return "GREEN";
            case SignalState::RED_YELLOW: return "RED_YELLOW";
            case SignalState::FLASHING_YELLOW: return "FLASHING_YELLOW";
            case SignalState::OFF: return "OFF";
            default: return "UNKNOWN";
        }
    }
};

// Violation record structure
struct ViolationRecord {
    int violationId;
    ViolationType type;
    std::chrono::system_clock::time_point timestamp;
    GPSCoordinate location;
    std::string plateNumber;
    float violationValue;  // e.g., measured speed
    float thresholdValue;  // e.g., speed limit
    cv::Mat evidenceImage;
    cv::Mat contextImage;  // Wider view showing context
    std::string videoFilePath;
    VehicleType vehicleType;
    int laneId;
    float confidence;
    std::vector<cv::Point> vehiclePositions;  // Track through violation zone
    std::string officerId;  // If manually verified
    bool isVerified;
    std::string notes;
    std::string deviceId;
    std::string serialNumber;
    std::string encryptionKey;
    std::string digitalSignature;
    WeatherCondition weatherCondition;
    int trafficDensity;  // 0-100%
    json metadataJson;
    
    ViolationRecord() : violationId(0), type(ViolationType::SPEED_VIOLATION),
                       timestamp(std::chrono::system_clock::now()),
                       violationValue(0.0f), thresholdValue(0.0f),
                       vehicleType(VehicleType::UNKNOWN), laneId(0),
                       confidence(0.0f), isVerified(false), trafficDensity(0) {}
    
    std::string generateFilename() const {
        std::stringstream ss;
        auto timeT = std::chrono::system_clock::to_time_t(timestamp);
        ss << std::put_time(std::localtime(&timeT), "%Y%m%d_%H%M%S_");
        ss << plateNumber << "_";
        switch (type) {
            case ViolationType::SPEED_VIOLATION: ss << "SPEED"; break;
            case ViolationType::RED_LIGHT_VIOLATION: ss << "REDLIGHT"; break;
            default: ss << "OTHER"; break;
        }
        ss << "_" << violationId << ".jpg";
        return ss.str();
    }
    
    std::string toJson() const {
        json j;
        j["violationId"] = violationId;
        j["type"] = static_cast<int>(type);
        auto timeT = std::chrono::system_clock::to_time_t(timestamp);
        j["timestamp"] = std::put_time(std::localtime(&timeT), "%Y-%m-%d %H:%M:%S");
        j["location"] = {
            {"latitude", location.latitude},
            {"longitude", location.longitude}
        };
        j["plateNumber"] = plateNumber;
        j["violationValue"] = violationValue;
        j["thresholdValue"] = thresholdValue;
        j["vehicleType"] = static_cast<int>(vehicleType);
        j["laneId"] = laneId;
        j["confidence"] = confidence;
        j["isVerified"] = isVerified;
        j["deviceId"] = deviceId;
        j["weatherCondition"] = static_cast<int>(weatherCondition);
        j["trafficDensity"] = trafficDensity;
        
        return j.dump();
    }
    
    static ViolationRecord fromJson(const std::string& jsonStr) {
        ViolationRecord rec;
        json j = json::parse(jsonStr);
        
        rec.violationId = j["violationId"];
        rec.type = static_cast<ViolationType>(j["type"]);
        
        std::tm tm = {};
        std::stringstream ss(j["timestamp"]);
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
        rec.timestamp = std::chrono::system_clock::from_time_t(std::mktime(&tm));
        
        rec.location.latitude = j["location"]["latitude"];
        rec.location.longitude = j["location"]["longitude"];
        rec.plateNumber = j["plateNumber"];
        rec.violationValue = j["violationValue"];
        rec.thresholdValue = j["thresholdValue"];
        rec.vehicleType = static_cast<VehicleType>(j["vehicleType"]);
        rec.laneId = j["laneId"];
        rec.confidence = j["confidence"];
        rec.isVerified = j["isVerified"];
        rec.deviceId = j["deviceId"];
        rec.weatherCondition = static_cast<WeatherCondition>(j["weatherCondition"]);
        rec.trafficDensity = j["trafficDensity"];
        
        return rec;
    }
};

// Camera calibration parameters
struct CalibrationParameters {
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Mat rotationVector;
    cv::Mat translationVector;
    double pixelsPerMeter;
    double cameraHeight;  // meters
    double cameraTilt;    // degrees
    double cameraYaw;     // degrees
    double focalLength;   // millimeters
    double fieldOfView;   // degrees
    cv::Mat homographyMatrix;
    std::vector<cv::Point2f> calibrationPoints;
    std::vector<cv::Point3f> worldPoints;
    bool isCalibrated;
    std::chrono::system_clock::time_point calibrationTime;
    float calibrationConfidence;
    cv::Mat undistortMap1;
    cv::Mat undistortMap2;
    
    CalibrationParameters() : cameraHeight(0.0), cameraTilt(0.0), cameraYaw(0.0),
                             focalLength(0.0), fieldOfView(0.0), isCalibrated(false),
                             calibrationTime(std::chrono::system_clock::now()),
                             calibrationConfidence(0.0f) {
        cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
        distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    }
    
    cv::Point2f projectToGround(const cv::Point2f& imagePoint) const {
        std::vector<cv::Point2f> imagePoints = {imagePoint};
        std::vector<cv::Point2f> groundPoints;
        cv::perspectiveTransform(imagePoints, groundPoints, homographyMatrix);
        return groundPoints[0];
    }
    
    double estimateRealWorldDistance(const cv::Point2f& point1, const cv::Point2f& point2) const {
        if (!isCalibrated) return 0.0;
        
        // Convert image points to world points
        cv::Point2f worldPoint1 = projectToGround(point1);
        cv::Point2f worldPoint2 = projectToGround(point2);
        
        // Calculate Euclidean distance in meters
        return cv::norm(worldPoint1 - worldPoint2) / pixelsPerMeter;
    }
    
    cv::Mat undistortImage(const cv::Mat& image) const {
        if (!isCalibrated || image.empty()) return image;
        
        cv::Mat undistorted;
        if (!undistortMap1.empty() && !undistortMap2.empty()) {
            cv::remap(image, undistorted, undistortMap1, undistortMap2, cv::INTER_LINEAR);
        } else {
            cv::undistort(image, undistorted, cameraMatrix, distCoeffs);
        }
        return undistorted;
    }
};

// Weather information structure
struct WeatherInfo {
    WeatherCondition condition;
    float temperature;  // Celsius
    float humidity;     // percentage
    float windSpeed;    // km/h
    float precipitation; // mm/h
    float visibility;    // km
    float lightLevel;    // lux
    std::chrono::system_clock::time_point timestamp;
    bool affectsDetection;
    float roadSurface;   // 0.0=dry, 1.0=wet
    TimeOfDay timeOfDay;
    bool isSnowing;
    bool isFoggy;
    float fogDensity;    // 0.0-1.0
    bool isNight;
    
    WeatherInfo() : condition(WeatherCondition::CLEAR), temperature(20.0f),
                   humidity(50.0f), windSpeed(0.0f), precipitation(0.0f),
                   visibility(10.0f), lightLevel(10000.0f),
                   timestamp(std::chrono::system_clock::now()),
                   affectsDetection(false), roadSurface(0.0f),
                   timeOfDay(TimeOfDay::NOON), isSnowing(false),
                   isFoggy(false), fogDensity(0.0f), isNight(false) {}
    
    bool requiresImageEnhancement() const {
        return visibility < 5.0f || lightLevel < 500.0f || isNight || 
               isFoggy || isSnowing || precipitation > 5.0f;
    }
    
    std::string getConditionString() const {
        switch (condition) {
            case WeatherCondition::CLEAR: return "Clear";
            case WeatherCondition::CLOUDY: return "Cloudy";
            case WeatherCondition::RAINY: return "Rainy";
            case WeatherCondition::SNOWY: return "Snowy";
            case WeatherCondition::FOGGY: return "Foggy";
            case WeatherCondition::SANDSTORM: return "Sandstorm";
            case WeatherCondition::HAIL: return "Hail";
            case WeatherCondition::THUNDERSTORM: return "Thunderstorm";
            case WeatherCondition::EXTREME_HEAT: return "Extreme Heat";
            case WeatherCondition::EXTREME_COLD: return "Extreme Cold";
            default: return "Unknown";
        }
    }
};

// System status structure
struct SystemStatus {
    enum class Status {
        OK,
        WARNING,
        ERROR,
        CRITICAL,
        OFFLINE
    };
    
    Status overallStatus;
    std::map<std::string, Status> componentStatus;
    float cpuUsage;         // percentage
    float memoryUsage;      // percentage
    float diskUsage;        // percentage
    float networkBandwidth; // Mbps
    float temperature;      // Celsius
    std::string currentOperation;
    int activeVehicleTracks;
    int processedFrames;
    int detectedViolations;
    int uploadedViolations;
    int pendingViolations;
    std::chrono::system_clock::time_point startTime;
    std::chrono::system_clock::time_point lastMaintenance;
    std::string lastErrorMessage;
    int operatingHours;
    float batteryLevel;     // percentage
    bool usingBackupPower;
    int frameRate;          // FPS
    int processingLatency;  // milliseconds
    std::map<std::string, std::string> diagnosticMessages;
    
    SystemStatus() : overallStatus(Status::OK), cpuUsage(0.0f), memoryUsage(0.0f),
                    diskUsage(0.0f), networkBandwidth(0.0f), temperature(25.0f),
                    activeVehicleTracks(0), processedFrames(0),
                    detectedViolations(0), uploadedViolations(0), pendingViolations(0),
                    startTime(std::chrono::system_clock::now()),
                    lastMaintenance(std::chrono::system_clock::now()),
                    operatingHours(0), batteryLevel(100.0f), usingBackupPower(false),
                    frameRate(0), processingLatency(0) {}
    
    std::chrono::hours getUptime() const {
        return std::chrono::duration_cast<std::chrono::hours>(
            std::chrono::system_clock::now() - startTime);
    }
    
    std::string getStatusString() const {
        switch (overallStatus) {
            case Status::OK: return "OK";
            case Status::WARNING: return "WARNING";
            case Status::ERROR: return "ERROR";
            case Status::CRITICAL: return "CRITICAL";
            case Status::OFFLINE: return "OFFLINE";
            default: return "UNKNOWN";
        }
    }
    
    bool needsMaintenance() const {
        auto now = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::hours>(now - lastMaintenance);
        return elapsed.count() > 720 || // 30 days
               temperature > 80.0f ||
               overallStatus == Status::ERROR ||
               overallStatus == Status::CRITICAL;
    }
};

// Configuration class
class ConfigManager {
private:
    json config;
    std::string configFilePath;
    std::mutex configMutex;
    std::map<std::string, std::function<void(const json&)>> changeCallbacks;
    
public:
    ConfigManager(const std::string& configPath = DEFAULT_CONFIG_FILE)
        : configFilePath(configPath) {
        loadConfig();
    }
    
    bool loadConfig() {
        std::lock_guard<std::mutex> lock(configMutex);
        
        try {
            std::ifstream configFile(configFilePath);
            if (configFile.is_open()) {
                configFile >> config;
                return true;
            } else {
                // Create default configuration
                createDefaultConfig();
                saveConfig();
                return true;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error loading configuration: " << e.what() << std::endl;
            createDefaultConfig();
            return false;
        }
    }
    
    bool saveConfig() {
        std::lock_guard<std::mutex> lock(configMutex);
        
        try {
            // Create directory if it doesn't exist
            fs::path configDir = fs::path(configFilePath).parent_path();
            if (!configDir.empty() && !fs::exists(configDir)) {
                fs::create_directories(configDir);
            }
            
            std::ofstream configFile(configFilePath);
            if (configFile.is_open()) {
                configFile << std::setw(4) << config << std::endl;
                return true;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error saving configuration: " << e.what() << std::endl;
        }
        
        return false;
    }
    
    template<typename T>
    T getValue(const std::string& key, const T& defaultValue) const {
        std::lock_guard<std::mutex> lock(configMutex);
        
        try {
            json::json_pointer ptr(key.empty() ? "/" : "/" + key);
            if (config.contains(ptr)) {
                return config.at(ptr).get<T>();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error getting config value for key '" << key << "': " << e.what() << std::endl;
        }
        
        return defaultValue;
    }
    
    template<typename T>
    void setValue(const std::string& key, const T& value) {
        std::lock_guard<std::mutex> lock(configMutex);
        
        try {
            json::json_pointer ptr(key.empty() ? "/" : "/" + key);
            json& node = config[ptr];
            node = value;
            
            // Notify change listeners
            for (const auto& [callbackKey, callback] : changeCallbacks) {
                if (key.find(callbackKey) == 0 || callbackKey.empty()) {
                    callback(node);
                }
            }
            
            saveConfig();
        } catch (const std::exception& e) {
            std::cerr << "Error setting config value for key '" << key << "': " << e.what() << std::endl;
        }
    }
    
    void registerChangeCallback(const std::string& keyPrefix, std::function<void(const json&)> callback) {
        std::lock_guard<std::mutex> lock(configMutex);
        changeCallbacks[keyPrefix] = callback;
    }
    
    bool hasKey(const std::string& key) const {
        std::lock_guard<std::mutex> lock(configMutex);
        
        try {
            json::json_pointer ptr(key.empty() ? "/" : "/" + key);
            return config.contains(ptr);
        } catch (const std::exception& e) {
            return false;
        }
    }
    
    json getFullConfig() const {
        std::lock_guard<std::mutex> lock(configMutex);
        return config;
    }
    
private:
    void createDefaultConfig() {
        config = {
            {"version", VERSION},
            {"device", {
                {"id", DEFAULT_DEVICE_ID},
                {"auth_key", DEFAULT_DEVICE_AUTH_KEY},
                {"location", {
                    {"latitude", DEFAULT_GPS_LAT},
                    {"longitude", DEFAULT_GPS_LON},
                    {"altitude", 10.0}
                }},
                {"camera_height", DEFAULT_CAMERA_HEIGHT_METERS},
                {"camera_angle", DEFAULT_CAMERA_ANGLE_DEGREES},
                {"camera_type", static_cast<int>(CameraType::FIXED)}
            }},
            {"camera", {
                {"width", DEFAULT_CAMERA_WIDTH},
                {"height", DEFAULT_CAMERA_HEIGHT},
                {"fps", DEFAULT_CAMERA_FPS},
                {"source", "0"},
                {"sensor_type", "CMOS"},
                {"enable_hdr", DEFAULT_ENABLE_HDR},
                {"enable_defogging", DEFAULT_ENABLE_DEFOGGING},
                {"enable_day_night_switching", DEFAULT_ENABLE_DAY_NIGHT_SWITCHING},
                {"night_mode_threshold", DEFAULT_NIGHT_MODE_THRESHOLD}
            }},
            {"detection", {
                {"algorithm", static_cast<int>(DetectionAlgorithm::YOLO_V5)},
                {"confidence_threshold", DEFAULT_DETECTION_THRESHOLD},
                {"model_path", DEFAULT_MODEL_PATH + "yolov5m.onnx"},
                {"enable_gpu", true},
                {"gpu_id", DEFAULT_GPU_ID},
                {"min_vehicle_size", DEFAULT_MINIMUM_VEHICLE_SIZE},
                {"max_vehicle_size", DEFAULT_MAXIMUM_VEHICLE_SIZE},
                {"use_ai_enhancement", DEFAULT_USE_AI_ENHANCEMENT}
            }},
            {"tracking", {
                {"algorithm", static_cast<int>(TrackingAlgorithm::DEEP_SORT)},
                {"max_vehicles", DEFAULT_MAX_VEHICLES_TRACK},
                {"matching_threshold", 0.3},
                {"tracking_confidence", DEFAULT_TRACKING_CONFIDENCE},
                {"min_frames_for_violation", DEFAULT_MIN_FRAMES_FOR_VIOLATION}
            }},
            {"speed", {
                {"speed_limit", DEFAULT_SPEED_LIMIT},
                {"speed_tolerance", DEFAULT_SPEED_TOLERANCE},
                {"distance_calibration_points", json::array()},
                {"enable_radar_integration", true},
                {"radar_detection_range", DEFAULT_RADAR_DETECTION_RANGE},
                {"lidar_detection_range", DEFAULT_LIDAR_DETECTION_RANGE}
            }},
            {"plate_recognition", {
                {"enabled", true},
                {"min_chars", DEFAULT_LICENSE_PLATE_MIN_CHARS},
                {"max_chars", DEFAULT_LICENSE_PLATE_MAX_CHARS},
                {"confidence_threshold", 0.7},
                {"model_path", DEFAULT_MODEL_PATH + "lprnet.onnx"},
                {"region", "auto"}
            }},
            {"violations", {
                {"types", {
                    {"speed", true},
                    {"red_light", false},
                    {"unsafe_distance", false},
                    {"improper_lane_change", false}
                }},
                {"buffer_size", DEFAULT_VIOLATION_BUFFER_SIZE},
                {"min_confidence", 0.8},
                {"output_path", DEFAULT_OUTPUT_PATH},
                {"enable_encryption", DEFAULT_ENABLE_ENCRYPTION},
                {"retention_days", DEFAULT_RETENTION_DAYS},
                {"privacy_mask_enabled", DEFAULT_PRIVACY_MASK_ENABLED}
            }},
            {"storage", {
                {"database_path", DEFAULT_DATABASE_PATH},
                {"enable_cloud_upload", DEFAULT_ENABLE_CLOUD_UPLOAD},
                {"cloud_url", "https://api.speedguardian.com/violations"},
                {"max_disk_usage_gb", 500},
                {"compression_quality", DEFAULT_COMPRESSION_QUALITY},
                {"enable_h265", DEFAULT_ENABLE_H265}
            }},
            {"analytics", {
                {"enabled", DEFAULT_ENABLE_ANALYTICS},
                {"interval", DEFAULT_TRAFFIC_ANALYTICS_INTERVAL},
                {"metrics", {
                    {"speed", true},
                    {"volume", true},
                    {"classification", true},
                    {"violations", true}
                }}
            }},
            {"system", {
                {"log_level", DEFAULT_LOG_LEVEL},
                {"processing_threads", DEFAULT_PROCESSING_THREADS},
                {"hardware_acceleration", static_cast<int>(HardwareAcceleration::CUDA)},
                {"health_check_interval", DEFAULT_HEALTH_CHECK_INTERVAL},
                {"auto_calibration_interval", DEFAULT_AUTO_CALIBRATION_INTERVAL},
                {"enable_self_diagnostics", DEFAULT_ENABLE_SELF_DIAGNOSTICS},
                {"network_timeout", DEFAULT_NETWORK_TIMEOUT},
                {"secure_boot_verification", DEFAULT_SECURE_BOOT_VERIFICATION},
                {"heartbeat_interval", DEFAULT_HEARTBEAT_INTERVAL},
                {"time_server", DEFAULT_NTP_SERVER}
            }},
            {"lanes", json::array()}
        };
    }
};

// Logging class
class LogManager {
public:
    LogManager(const std::string& logFile = "logs/speedguardian.log") 
        : logFilePath(logFile), logLevel(LogLevel::INFO) {
        
        // Create log directory if it doesn't exist
        fs::path logDir = fs::path(logFilePath).parent_path();
        if (!logDir.empty() && !fs::exists(logDir)) {
            fs::create_directories(logDir);
        }
        
        try {
            logFileStream.open(logFilePath, std::ios::app);
        } catch (const std::exception& e) {
            std::cerr << "Failed to open log file: " << e.what() << std::endl;
        }
    }
    
    ~LogManager() {
        if (logFileStream.is_open()) {
            logFileStream.close();
        }
    }
    
    void setLogLevel(LogLevel level) {
        logLevel = level;
    }
    
    void setLogLevel(const std::string& level) {
        if (level == "TRACE") logLevel = LogLevel::TRACE;
        else if (level == "DEBUG") logLevel = LogLevel::DEBUG;
        else if (level == "INFO") logLevel = LogLevel::INFO;
        else if (level == "WARNING") logLevel = LogLevel::WARNING;
        else if (level == "ERROR") logLevel = LogLevel::ERROR;
        else if (level == "CRITICAL") logLevel = LogLevel::CRITICAL;
        else if (level == "NONE") logLevel = LogLevel::NONE;
    }
    
    template<typename... Args>
    void log(LogLevel level, const char* format, Args... args) {
        if (level < logLevel || logLevel == LogLevel::NONE)
            return;
        
        std::lock_guard<std::mutex> lock(logMutex);
        
        char buffer[4096];
        std::snprintf(buffer, sizeof(buffer), format, args...);
        
        std::string formattedMessage = formatLogMessage(level, buffer);
        
        // Output to console
        if (level >= LogLevel::WARNING) {
            std::cerr << formattedMessage << std::endl;
        } else {
            std::cout << formattedMessage << std::endl;
        }
        
        // Write to log file
        if (logFileStream.is_open()) {
            logFileStream << formattedMessage << std::endl;
            logFileStream.flush();
        }
    }
    
    // Convenience methods for different log levels
    template<typename... Args>
    void trace(const char* format, Args... args) {
        log(LogLevel::TRACE, format, args...);
    }
    
    template<typename... Args>
    void debug(const char* format, Args... args) {
        log(LogLevel::DEBUG, format, args...);
    }
    
    template<typename... Args>
    void info(const char* format, Args... args) {
        log(LogLevel::INFO, format, args...);
    }
    
    template<typename... Args>
    void warning(const char* format, Args... args) {
        log(LogLevel::WARNING, format, args...);
    }
    
    template<typename... Args>
    void error(const char* format, Args... args) {
        log(LogLevel::ERROR, format, args...);
    }
    
    template<typename... Args>
    void critical(const char* format, Args... args) {
        log(LogLevel::CRITICAL, format, args...);
    }
    
    void rotateLog() {
        std::lock_guard<std::mutex> lock(logMutex);
        
        if (logFileStream.is_open()) {
            logFileStream.close();
        }
        
        // Get current time for backup filename
        auto now = std::chrono::system_clock::now();
        auto timeT = std::chrono::system_clock::to_time_t(now);
        char timeBuffer[80];
        std::strftime(timeBuffer, sizeof(timeBuffer), "%Y%m%d_%H%M%S", std::localtime(&timeT));
        
        // Create backup file name
        std::string backupFileName = logFilePath + "." + std::string(timeBuffer);
        
        // Rename current log file
        try {
            fs::rename(logFilePath, backupFileName);
        } catch (const std::exception& e) {
            std::cerr << "Failed to rotate log file: " << e.what() << std::endl;
        }
        
        // Open new log file
        try {
            logFileStream.open(logFilePath, std::ios::app);
            info("Log file rotated, previous log saved as %s", backupFileName.c_str());
        } catch (const std::exception& e) {
            std::cerr << "Failed to open new log file after rotation: " << e.what() << std::endl;
        }
    }
    
private:
    std::string logFilePath;
    std::ofstream logFileStream;
    LogLevel logLevel;
    std::mutex logMutex;
    
    std::string formatLogMessage(LogLevel level, const std::string& message) {
        auto now = std::chrono::system_clock::now();
        auto timeT = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        char timeBuffer[80];
        std::strftime(timeBuffer, sizeof(timeBuffer), "%Y-%m-%d %H:%M:%S", std::localtime(&timeT));
        
        std::string levelStr;
        switch (level) {
            case LogLevel::TRACE: levelStr = "TRACE"; break;
            case LogLevel::DEBUG: levelStr = "DEBUG"; break;
            case LogLevel::INFO: levelStr = "INFO"; break;
            case LogLevel::WARNING: levelStr = "WARNING"; break;
            case LogLevel::ERROR: levelStr = "ERROR"; break;
            case LogLevel::CRITICAL: levelStr = "CRITICAL"; break;
            default: levelStr = "UNKNOWN";
        }
        
        char threadIdBuffer[16];
        std::snprintf(threadIdBuffer, sizeof(threadIdBuffer), "%lx", 
                     static_cast<unsigned long>(std::hash<std::thread::id>{}(std::this_thread::get_id())));
        
        std::stringstream ss;
        ss << timeBuffer << "." << std::setfill('0') << std::setw(3) << ms.count()
           << " [" << levelStr << "] [" << threadIdBuffer << "] " << message;
        
        return ss.str();
    }
};

// Vehicle detector class
class VehicleDetector {
private:
    cv::dnn::Net model;
    std::vector<std::string> classNames;
    float confidenceThreshold;
    float nmsThreshold;
    int inputWidth;
    int inputHeight;
    DetectionAlgorithm algorithm;
    bool useGPU;
    int gpuId;
    LogManager& logger;
    bool isInitialized;
    bool useAIEnhancement;
    HardwareAcceleration hardwareAcceleration;
    std::string modelPath;
    std::mutex detectorMutex;
    cv::Size processSize;
    float minVehicleSize;
    float maxVehicleSize;
    
public:
    VehicleDetector(LogManager& logManager)
        : confidenceThreshold(DEFAULT_DETECTION_THRESHOLD),
          nmsThreshold(0.4f),
          inputWidth(640),
          inputHeight(640),
          algorithm(DetectionAlgorithm::YOLO_V5),
          useGPU(true),
          gpuId(DEFAULT_GPU_ID),
          logger(logManager),
          isInitialized(false),
          useAIEnhancement(DEFAULT_USE_AI_ENHANCEMENT),
          hardwareAcceleration(HardwareAcceleration::CUDA),
          modelPath(DEFAULT_MODEL_PATH + "yolov5m.onnx"),
          processSize(1280, 720),
          minVehicleSize(DEFAULT_MINIMUM_VEHICLE_SIZE),
          maxVehicleSize(DEFAULT_MAXIMUM_VEHICLE_SIZE) {}
    
    bool initialize(const ConfigManager& config) {
        std::lock_guard<std::mutex> lock(detectorMutex);
        
        // Load configuration
        confidenceThreshold = config.getValue<float>("detection/confidence_threshold", DEFAULT_DETECTION_THRESHOLD);
        algorithm = static_cast<DetectionAlgorithm>(config.getValue<int>("detection/algorithm", static_cast<int>(DetectionAlgorithm::YOLO_V5)));
        modelPath = config.getValue<std::string>("detection/model_path", DEFAULT_MODEL_PATH + "yolov5m.onnx");
        useGPU = config.getValue<bool>("detection/enable_gpu", true);
        gpuId = config.getValue<int>("detection/gpu_id", DEFAULT_GPU_ID);
        useAIEnhancement = config.getValue<bool>("detection/use_ai_enhancement", DEFAULT_USE_AI_ENHANCEMENT);
        hardwareAcceleration = static_cast<HardwareAcceleration>(config.getValue<int>("system/hardware_acceleration", static_cast<int>(HardwareAcceleration::CUDA)));
        minVehicleSize = config.getValue<float>("detection/min_vehicle_size", DEFAULT_MINIMUM_VEHICLE_SIZE);
        maxVehicleSize = config.getValue<float>("detection/max_vehicle_size", DEFAULT_MAXIMUM_VEHICLE_SIZE);
        
        // Set input dimensions based on model
        switch (algorithm) {
            case DetectionAlgorithm::YOLO_V4:
                inputWidth = 416;
                inputHeight = 416;
                break;
            case DetectionAlgorithm::YOLO_V5:
                inputWidth = 640;
                inputHeight = 640;
                break;
            case DetectionAlgorithm::SSD:
                inputWidth = 300;
                inputHeight = 300;
                break;
            case DetectionAlgorithm::EFFICIENTDET:
                inputWidth = 512;
                inputHeight = 512;
                break;
            default:
                inputWidth = 640;
                inputHeight = 640;
        }
        
        // Check if model file exists
        if (!fs::exists(modelPath)) {
            logger.error("Model file not found: %s", modelPath.c_str());
            return false;
        }
        
        try {
            // Load DNN model
            model = cv::dnn::readNet(modelPath);
            
            // Configure backend
            if (useGPU) {
                if (hardwareAcceleration == HardwareAcceleration::CUDA || 
                    hardwareAcceleration == HardwareAcceleration::CUDA_FP16) {
                    model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                    model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                    
                    if (hardwareAcceleration == HardwareAcceleration::CUDA_FP16) {
                        model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
                    }
                    
                    // Set CUDA device
                    cv::cuda::setDevice(gpuId);
                    logger.info("Using CUDA device %d for vehicle detection", gpuId);
                } else if (hardwareAcceleration == HardwareAcceleration::OPENCL) {
                    model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                    model.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
                    logger.info("Using OpenCL for vehicle detection");
                } else {
                    logger.warning("Requested hardware acceleration not supported, falling back to CPU");
                    model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                    model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                }
            } else {
                model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                logger.info("Using CPU for vehicle detection");
            }
            
            // Load class names
            loadClassNames();
            
            isInitialized = true;
            logger.info("Vehicle detector initialized successfully using %s algorithm", 
                       getAlgorithmName().c_str());
            return true;
        }
        catch (const cv::Exception& e) {
            logger.error("Failed to initialize vehicle detector: %s", e.what());
            return false;
        }
    }
    
    std::vector<VehicleDetection> detectVehicles(const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(detectorMutex);
        std::vector<VehicleDetection> detections;
        
        if (!isInitialized || frame.empty()) {
            return detections;
        }
        
        try {
            cv::Mat processFrame;
            // Resize if needed to improve performance
            if (frame.cols > processSize.width || frame.rows > processSize.height) {
                cv::resize(frame, processFrame, processSize);
            } else {
                processFrame = frame.clone();
            }
            
            // Image enhancement if needed
            if (useAIEnhancement) {
                enhanceImage(processFrame);
            }
            
            // Create blob from image
            cv::Mat blob;
            cv::dnn::blobFromImage(processFrame, blob, 1/255.0, cv::Size(inputWidth, inputHeight), 
                                  cv::Scalar(0, 0, 0), true, false);
            
            // Set input and forward pass
            model.setInput(blob);
            std::vector<cv::Mat> outputs;
            model.forward(outputs, getOutputsNames());
            
            // Extract detections based on algorithm
            switch (algorithm) {
                case DetectionAlgorithm::YOLO_V5:
                case DetectionAlgorithm::YOLO_V4:
                    processYOLOOutput(outputs, processFrame, detections);
                    break;
                case DetectionAlgorithm::SSD:
                    processSSDOutput(outputs[0], processFrame, detections);
                    break;
                default:
                    processYOLOOutput(outputs, processFrame, detections);
            }
            
            // Scale back to original frame size if needed
            if (processFrame.size() != frame.size()) {
                float scaleX = (float)frame.cols / processFrame.cols;
                float scaleY = (float)frame.rows / processFrame.rows;
                
                for (auto& detection : detections) {
                    detection.boundingBox.x *= scaleX;
                    detection.boundingBox.y *= scaleY;
                    detection.boundingBox.width *= scaleX;
                    detection.boundingBox.height *= scaleY;
                }
            }
            
            // Filter by vehicle size if needed
            detections.erase(
                std::remove_if(detections.begin(), detections.end(),
                    [this, &frame](const VehicleDetection& detection) {
                        float area = detection.boundingBox.width * detection.boundingBox.height;
                        float frameArea = frame.cols * frame.rows;
                        float ratio = area / frameArea;
                        return ratio < minVehicleSize || ratio > maxVehicleSize;
                    }
                ),
                detections.end()
            );
            
            return detections;
        }
        catch (const cv::Exception& e) {
            logger.error("Error during vehicle detection: %s", e.what());
            return detections;
        }
    }
    
    std::string getAlgorithmName() const {
        switch (algorithm) {
            case DetectionAlgorithm::YOLO_V4: return "YOLOv4";
            case DetectionAlgorithm::YOLO_V5: return "YOLOv5";
            case DetectionAlgorithm::SSD: return "SSD";
            case DetectionAlgorithm::FASTER_RCNN: return "Faster R-CNN";
            case DetectionAlgorithm::EFFICIENTDET: return "EfficientDet";
            case DetectionAlgorithm::RETINA_NET: return "RetinaNet";
            case DetectionAlgorithm::MASK_RCNN: return "Mask R-CNN";
            case DetectionAlgorithm::CENTERNET: return "CenterNet";
            default: return "Unknown Algorithm";
        }
    }
    
private:
    void loadClassNames() {
        classNames = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
            "toothbrush"
        };
    }
    
    std::vector<std::string> getOutputsNames() {
        static std::vector<std::string> names;
        
        if (names.empty()) {
            std::vector<int> outLayers = model.getUnconnectedOutLayers();
            std::vector<std::string> layersNames = model.getLayerNames();
            
            names.resize(outLayers.size());
            for (size_t i = 0; i < outLayers.size(); ++i) {
                names[i] = layersNames[outLayers[i] - 1];
            }
        }
        
        return names;
    }
    
    void processYOLOOutput(const std::vector<cv::Mat>& outputs, const cv::Mat& frame, 
                          std::vector<VehicleDetection>& detections) {
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        
        // Process each output layer
        for (const auto& output : outputs) {
            // YOLO v5 format has [x, y, w, h, obj_conf, class_conf1, class_conf2, ...]
            for (int i = 0; i < output.rows; ++i) {
                float* data = (float*)output.row(i).data;
                
                // Object confidence at index 4
                float objectConfidence = data[4];
                
                if (objectConfidence >= confidenceThreshold) {
                    // Find best class
                    float maxClassConfidence = 0;
                    int bestClassId = 0;
                    
                    for (int j = 5; j < output.cols; ++j) {
                        float classConfidence = data[j];
                        if (classConfidence > maxClassConfidence) {
                            maxClassConfidence = classConfidence;
                            bestClassId = j - 5;
                        }
                    }
                    
                    float confidence = objectConfidence * maxClassConfidence;
                    
                    if (confidence >= confidenceThreshold) {
                        // Filter relevant classes (car, truck, bus, motorcycle)
                        if (bestClassId == 2 || bestClassId == 7 || bestClassId == 5 || bestClassId == 3) {
                            // Extract bounding box
                            float centerX = data[0] * frame.cols;
                            float centerY = data[1] * frame.rows;
                            float width = data[2] * frame.cols;
                            float height = data[3] * frame.rows;
                            
                            int left = int(centerX - width / 2);
                            int top = int(centerY - height / 2);
                            
                            classIds.push_back(bestClassId);
                            confidences.push_back(confidence);
                            boxes.push_back(cv::Rect(left, top, (int)width, (int)height));
                        }
                    }
                }
            }
        }
        
        // Apply non-maximum suppression to remove overlapping boxes
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);
        
        // Create vehicle detections from filtered results
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            
            VehicleDetection detection;
            detection.id = i;
            detection.boundingBox = boxes[idx];
            detection.confidence = confidences[idx];
            detection.timestamp = std::chrono::system_clock::now();
            
            // Set vehicle type based on class ID
            switch (classIds[idx]) {
                case 2:  // car
                    detection.type = VehicleType::CAR;
                    break;
                case 3:  // motorcycle
                    detection.type = VehicleType::MOTORCYCLE;
                    break;
                case 5:  // bus
                    detection.type = VehicleType::BUS;
                    break;
                case 7:  // truck
                    detection.type = VehicleType::TRUCK_SMALL;
                    break;
                default:
                    detection.type = VehicleType::UNKNOWN;
            }
            
            // Extract dominant color
            if (detection.boundingBox.x >= 0 && detection.boundingBox.y >= 0 &&
                detection.boundingBox.x + detection.boundingBox.width <= frame.cols &&
                detection.boundingBox.y + detection.boundingBox.height <= frame.rows) {
                
                cv::Mat vehicleROI = frame(detection.boundingBox);
                cv::Scalar dominantColor = calculateDominantColor(vehicleROI);
                detection.color = dominantColor;
                detection.colorName = getColorName(dominantColor);
            }
            
            detections.push_back(detection);
        }
    }
    
    void processSSDOutput(const cv::Mat& output, const cv::Mat& frame,
                         std::vector<VehicleDetection>& detections) {
        for (int i = 0; i < output.rows; i++) {
            float confidence = output.at<float>(i, 2);
            
            if (confidence > confidenceThreshold) {
                int classId = static_cast<int>(output.at<float>(i, 1));
                
                // Filter relevant classes (car, truck, bus, motorcycle)
                if (classId == 2 || classId == 7 || classId == 5 || classId == 3) {
                    int left = static_cast<int>(output.at<float>(i, 3) * frame.cols);
                    int top = static_cast<int>(output.at<float>(i, 4) * frame.rows);
                    int right = static_cast<int>(output.at<float>(i, 5) * frame.cols);
                    int bottom = static_cast<int>(output.at<float>(i, 6) * frame.rows);
                    
                    VehicleDetection detection;
                    detection.id = i;
                    detection.boundingBox = cv::Rect(left, top, right - left, bottom - top);
                    detection.confidence = confidence;
                    detection.timestamp = std::chrono::system_clock::now();
                    
                    // Set vehicle type based on class ID
                    switch (classId) {
                        case 2:  // car
                            detection.type = VehicleType::CAR;
                            break;
                        case 3:  // motorcycle
                            detection.type = VehicleType::MOTORCYCLE;
                            break;
                        case 5:  // bus
                            detection.type = VehicleType::BUS;
                            break;
                        case 7:  // truck
                            detection.type = VehicleType::TRUCK_SMALL;
                            break;
                        default:
                            detection.type = VehicleType::UNKNOWN;
                    }
                    
                    // Extract dominant color
                    if (detection.boundingBox.x >= 0 && detection.boundingBox.y >= 0 &&
                        detection.boundingBox.x + detection.boundingBox.width <= frame.cols &&
                        detection.boundingBox.y + detection.boundingBox.height <= frame.rows) {
                        
                        cv::Mat vehicleROI = frame(detection.boundingBox);
                        cv::Scalar dominantColor = calculateDominantColor(vehicleROI);
                        detection.color = dominantColor;
                        detection.colorName = getColorName(dominantColor);
                    }
                    
                    detections.push_back(detection);
                }
            }
        }
    }
    
    cv::Scalar calculateDominantColor(const cv::Mat& image) {
        // Convert to small size for faster processing
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(32, 32));
        
        // Convert to RGB color space
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        
        // Reshape to a single row of pixels
        cv::Mat pixels = rgb.reshape(3, rgb.total());
        pixels.convertTo(pixels, CV_32F);
        
        // Apply k-means clustering to find dominant colors
        cv::Mat labels, centers;
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0);
        cv::kmeans(pixels, 1, labels, criteria, 3, cv::KMEANS_PP_CENTERS, centers);
        
        // Convert back to BGR for return value
        cv::Vec3f dominantColor = centers.at<cv::Vec3f>(0);
        return cv::Scalar(dominantColor[2], dominantColor[1], dominantColor[0]);
    }
    
    std::string getColorName(const cv::Scalar& color) {
        // Simple color classification
        int b = color[0];
        int g = color[1];
        int r = color[2];
        
        // Check for black/white/gray first
        if (r < 30 && g < 30 && b < 30)
            return "black";
        if (r > 200 && g > 200 && b > 200)
            return "white";
        if (std::abs(r - g) < 20 && std::abs(r - b) < 20 && std::abs(g - b) < 20)
            return "gray";
        
        // Find the dominant channel
        if (r > g && r > b) {
            if (r > 150 && g > 100)
                return "orange";
            return "red";
        }
        if (g > r && g > b)
            return "green";
        if (b > r && b > g)
            return "blue";
        
        if (r > 150 && g > 150 && r > b && g > b)
            return "yellow";
        if (r > 150 && b > 150 && r > g && b > g)
            return "purple";
        if (g > 150 && b > 150 && g > r && b > r)
            return "cyan";
        
        return "unknown";
    }
    
    void enhanceImage(cv::Mat& image) {
        // Apply contrast enhancement
        cv::Mat yuv;
        cv::cvtColor(image, yuv, cv::COLOR_BGR2YUV);
        
        // Extract the Y channel
        std::vector<cv::Mat> channels;
        cv::split(yuv, channels);
        
        // Apply histogram equalization to the Y channel
        cv::equalizeHist(channels[0], channels[0]);
        
        // Merge the channels back
        cv::merge(channels, yuv);
        
        // Convert back to BGR
        cv::cvtColor(yuv, image, cv::COLOR_YUV2BGR);
    }
};

// License plate recognition class
class PlateRecognizer {
private:
    cv::dnn::Net detectionModel;
    cv::dnn::Net recognitionModel;
    float detectionThreshold;
    float recognitionThreshold;
    bool isInitialized;
    LogManager& logger;
    bool useGPU;
    std::string region;
    int minChars;
    int maxChars;
    std::map<int, char> charMap;
    std::regex platePattern;
    
public:
    PlateRecognizer(LogManager& logManager)
        : detectionThreshold(0.5f),
          recognitionThreshold(0.7f),
          isInitialized(false),
          logger(logManager),
          useGPU(true),
          region("auto"),
          minChars(DEFAULT_LICENSE_PLATE_MIN_CHARS),
          maxChars(DEFAULT_LICENSE_PLATE_MAX_CHARS) {
        
        initCharMap();
    }
    
    bool initialize(const ConfigManager& config) {
        try {
            // Load configuration
            std::string detectionModelPath = config.getValue<std::string>("plate_recognition/detection_model_path", 
                                                DEFAULT_MODEL_PATH + "plate_detection.onnx");
            std::string recognitionModelPath = config.getValue<std::string>("plate_recognition/model_path", 
                                                 DEFAULT_MODEL_PATH + "lprnet.onnx");
            detectionThreshold = config.getValue<float>("plate_recognition/detection_threshold", 0.5f);
            recognitionThreshold = config.getValue<float>("plate_recognition/confidence_threshold", 0.7f);
            useGPU = config.getValue<bool>("detection/enable_gpu", true);
            region = config.getValue<std::string>("plate_recognition/region", "auto");
            minChars = config.getValue<int>("plate_recognition/min_chars", DEFAULT_LICENSE_PLATE_MIN_CHARS);
            maxChars = config.getValue<int>("plate_recognition/max_chars", DEFAULT_LICENSE_PLATE_MAX_CHARS);
            
            // Check if model files exist
            if (!fs::exists(detectionModelPath)) {
                logger.error("License plate detection model not found: %s", detectionModelPath.c_str());
                return false;
            }
            
            if (!fs::exists(recognitionModelPath)) {
                logger.error("License plate recognition model not found: %s", recognitionModelPath.c_str());
                return false;
            }
            
            // Load detection model
            detectionModel = cv::dnn::readNet(detectionModelPath);
            
            // Load recognition model
            recognitionModel = cv::dnn::readNet(recognitionModelPath);
            
            // Set GPU/CPU mode
            if (useGPU) {
                if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
                    detectionModel.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                    detectionModel.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                    
                    recognitionModel.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                    recognitionModel.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                    
                    logger.info("Using CUDA for license plate recognition");
                } else {
                    logger.warning("CUDA requested but not available, using CPU for license plate recognition");
                    detectionModel.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                    detectionModel.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                    
                    recognitionModel.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                    recognitionModel.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                }
            } else {
                detectionModel.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                detectionModel.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                
                recognitionModel.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                recognitionModel.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                
                logger.info("Using CPU for license plate recognition");
            }
            
            // Set region-specific plate pattern
            setRegionPattern(region);
            
            isInitialized = true;
            logger.info("License plate recognizer initialized successfully");
            return true;
        }
        catch (const cv::Exception& e) {
            logger.error("Failed to initialize license plate recognizer: %s", e.what());
            return false;
        }
    }
    
    std::vector<PlateDetection> detectPlates(const cv::Mat& frame, const std::vector<VehicleDetection>& vehicles) {
        std::vector<PlateDetection> plateDetections;
        
        if (!isInitialized || frame.empty()) {
            return plateDetections;
        }
        
        try {
            // Process each vehicle detection
            for (const auto& vehicle : vehicles) {
                // Only look for plates in vehicles of certain types
                if (vehicle.type != VehicleType::CAR && 
                    vehicle.type != VehicleType::TRUCK_SMALL && 
                    vehicle.type != VehicleType::TRUCK_LARGE && 
                    vehicle.type != VehicleType::BUS) {
                    continue;
                }
                
                // Apply ROI (slightly larger than vehicle bounding box)
                cv::Rect vehicleRect = vehicle.boundingBox;
                int extraWidth = vehicleRect.width * 0.1;
                int extraHeight = vehicleRect.height * 0.1;
                
                cv::Rect searchRect = cv::Rect(
                    std::max(0, vehicleRect.x - extraWidth/2),
                    std::max(0, vehicleRect.y - extraHeight/2),
                    std::min(frame.cols - vehicleRect.x, vehicleRect.width + extraWidth),
                    std::min(frame.rows - vehicleRect.y, vehicleRect.height + extraHeight)
                );
                
                if (searchRect.width <= 0 || searchRect.height <= 0) {
                    continue;
                }
                
                cv::Mat vehicleROI = frame(searchRect);
                
                // Detect potential license plates
                std::vector<cv::Rect> potentialPlates = detectPotentialPlates(vehicleROI);
                
                for (const auto& plateRect : potentialPlates) {
                    // Adjust plate rectangle to original frame coordinates
                    cv::Rect adjustedRect(
                        plateRect.x + searchRect.x,
                        plateRect.y + searchRect.y,
                        plateRect.width,
                        plateRect.height
                    );
                    
                    // Extract plate image
                    if (adjustedRect.x >= 0 && adjustedRect.y >= 0 &&
                        adjustedRect.x + adjustedRect.width <= frame.cols &&
                        adjustedRect.y + adjustedRect.height <= frame.rows) {
                        
                        cv::Mat plateImage = frame(adjustedRect);
                        
                        // Recognize text
                        std::string plateText;
                        float confidence;
                        std::tie(plateText, confidence) = recognizePlateText(plateImage);
                        
                        // Validate plate text
                        if (isValidPlateNumber(plateText)) {
                            PlateDetection plateDetection;
                            plateDetection.boundingBox = adjustedRect;
                            plateDetection.plateNumber = plateText;
                            plateDetection.confidence = confidence;
                            plateDetection.timestamp = std::chrono::system_clock::now();
                            plateDetection.plateImage = plateImage.clone();
                            plateDetection.region = region;
                            
                            plateDetections.push_back(plateDetection);
                        }
                    }
                }
            }
            
            return plateDetections;
        }
        catch (const cv::Exception& e) {
            logger.error("Error during license plate recognition: %s", e.what());
            return plateDetections;
        }
    }
    
private:
    void initCharMap() {
        // Map indices to characters (adjust based on your model's output)
        for (int i = 0; i < 10; i++)
            charMap[i] = '0' + i;  // Digits
        
        for (int i = 0; i < 26; i++)
            charMap[i + 10] = 'A' + i;  // Uppercase letters
    }
    
    void setRegionPattern(const std::string& region) {
        // Set regex pattern based on region
        // These are simplified patterns and should be adjusted for actual regions
        if (region == "US") {
            platePattern = std::regex("^[A-Z0-9]{5,8}$");
        } else if (region == "EU") {
            platePattern = std::regex("^[A-Z]{1,3}[0-9]{1,4}[A-Z]{1,3}$");
        } else if (region == "UK") {
            platePattern = std::regex("^[A-Z]{2}[0-9]{2}[A-Z]{3}$");
        } else {
            // Default pattern for "auto" or unknown regions
            platePattern = std::regex("^[A-Z0-9]{" + std::to_string(minChars) + "," + 
                                     std::to_string(maxChars) + "}$");
        }
    }
    
    std::vector<cv::Rect> detectPotentialPlates(const cv::Mat& vehicleROI) {
        std::vector<cv::Rect> plates;
        
        // Prepare input blob
        cv::Mat blob = cv::dnn::blobFromImage(vehicleROI, 1/255.0, cv::Size(416, 416), 
                                             cv::Scalar(0, 0, 0), true, false);
        
        // Run forward pass
        detectionModel.setInput(blob);
        cv::Mat output;
        detectionModel.forward(output);
        
        // Parse detection results (assuming YOLO-like output format)
        for (int i = 0; i < output.rows; ++i) {
            float confidence = output.at<float>(i, 4);
            
            if (confidence > detectionThreshold) {
                float centerX = output.at<float>(i, 0) * vehicleROI.cols;
                float centerY = output.at<float>(i, 1) * vehicleROI.rows;
                float width = output.at<float>(i, 2) * vehicleROI.cols;
                float height = output.at<float>(i, 3) * vehicleROI.rows;
                
                int left = static_cast<int>(centerX - width / 2);
                int top = static_cast<int>(centerY - height / 2);
                
                plates.push_back(cv::Rect(left, top, static_cast<int>(width), static_cast<int>(height)));
            }
        }
        
        return plates;
    }
    
    std::pair<std::string, float> recognizePlateText(const cv::Mat& plateImage) {
        // Preprocess the plate image
        cv::Mat processedPlate;
        cv::cvtColor(plateImage, processedPlate, cv::COLOR_BGR2GRAY);
        cv::resize(processedPlate, processedPlate, cv::Size(94, 24));
        cv::threshold(processedPlate, processedPlate, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        
        // Convert to blob
        cv::Mat blob = cv::dnn::blobFromImage(processedPlate, 1/255.0, cv::Size(94, 24), 
                                             cv::Scalar(0), false, false);
        
        // Run inference
        recognitionModel.setInput(blob);
        cv::Mat output = recognitionModel.forward();
        
        // Decode the output to text (assuming CTC-like output)
        std::string plateNumber;
        float confidence = 0.0f;
        std::tie(plateNumber, confidence) = decodePlateText(output);
        
        return std::make_pair(plateNumber, confidence);
    }
    
    std::pair<std::string, float> decodePlateText(const cv::Mat& output) {
        // This is a simplified CTC decoding, adjust based on your model's output format
        std::string result;
        float totalConfidence = 0.0f;
        int lastClass = -1;
        
        // Get the most probable class for each time step
        for (int t = 0; t < output.size[1]; ++t) {
            cv::Mat scores = output.col(t);
            cv::Point maxLoc;
            double maxVal;
            cv::minMaxLoc(scores, nullptr, &maxVal, nullptr, &maxLoc);
            
            int currentClass = maxLoc.y;
            
            // Apply CTC decoding (merge repeated characters)
            if (currentClass != lastClass && currentClass < charMap.size()) {
                if (currentClass > 0) {  // Skip blank (assuming 0 is blank)
                    result += charMap[currentClass];
                    totalConfidence += maxVal;
                }
            }
            
            lastClass = currentClass;
        }
        
        float avgConfidence = result.empty() ? 0.0f : totalConfidence / result.length();
        return std::make_pair(result, avgConfidence);
    }
    
    bool isValidPlateNumber(const std::string& plateNumber) {
        // Check if plate number has reasonable length
        if (plateNumber.length() < minChars || plateNumber.length() > maxChars) {
            return false;
        }
        
        // Check against region-specific pattern
        return std::regex_match(plateNumber, platePattern);
    }
};

// Speed calculator class
class SpeedCalculator {
private:
    CalibrationParameters calibration;
    LogManager& logger;
    bool isCalibrated;
    float speedToleranceKmh;
    bool useRadar;
    bool useLidar;
    std::vector<TrafficLane> lanes;
    
public:
    SpeedCalculator(LogManager& logManager)
        : logger(logManager), isCalibrated(false), speedToleranceKmh(DEFAULT_SPEED_TOLERANCE),
          useRadar(false), useLidar(false) {}
    
    bool initialize(const ConfigManager& config, const CalibrationParameters& calibParams) {
        calibration = calibParams;
        isCalibrated = calibParams.isCalibrated;
        
        speedToleranceKmh = config.getValue<float>("speed/speed_tolerance", DEFAULT_SPEED_TOLERANCE);
        useRadar = config.getValue<bool>("speed/enable_radar_integration", false);
        useLidar = config.getValue<bool>("speed/enable_lidar_integration", false);
        
        if (!isCalibrated) {
            logger.warning("Speed calculator not properly calibrated, speed measurements may be inaccurate");
        }
        
        logger.info("Speed calculator initialized with calibration confidence: %.2f%%", 
                   calibration.calibrationConfidence * 100.0f);
        
        return true;
    }
    
    void setLanes(const std::vector<TrafficLane>& trafficLanes) {
        lanes = trafficLanes;
    }
    
    float calculateSpeed(const VehicleTrack& track) {
        if (!isCalibrated || track.positionHistory.size() < 2 || track.timestampHistory.size() < 2) {
            return 0.0f;
        }
        
        try {
            // Use the most recent positions and timestamps
            size_t size = track.positionHistory.size();
            
            // Get a few points to average the speed (more stable)
            size_t numPoints = std::min(size, size_t(5));
            
            // We'll calculate average speed over multiple segments
            float totalSpeed = 0.0f;
            int validSegments = 0;
            
            for (size_t i = 1; i < numPoints; ++i) {
                size_t idx1 = size - i - 1;
                size_t idx2 = size - i;
                
                if (idx1 < track.positionHistory.size() && idx2 < track.positionHistory.size() &&
                    idx1 < track.timestampHistory.size() && idx2 < track.timestampHistory.size()) {
                    
                    cv::Point2f p1 = track.positionHistory[idx1];
                    cv::Point2f p2 = track.positionHistory[idx2];
                    
                    // Use calibration to get real-world distance
                    double distanceMeters = calibration.estimateRealWorldDistance(p1, p2);
                    
                    // Get time difference in seconds
                    std::chrono::duration<double> timeDiff = 
                        std::chrono::duration_cast<std::chrono::duration<double>>(
                            track.timestampHistory[idx2] - track.timestampHistory[idx1]);
                    
                    if (timeDiff.count() > 0 && distanceMeters > 0) {
                        // Calculate speed in meters per second
                        float speedMps = distanceMeters / timeDiff.count();
                        
                        // Convert to km/h
                        float speedKmh = speedMps * 3.6f;
                        
                        // Add to total if reasonable (filter out spikes)
                        if (speedKmh < 300.0f) {  // Maximum reasonable speed
                            totalSpeed += speedKmh;
                            validSegments++;
                        }
                    }
                }
            }
            
            // Calculate average speed
            if (validSegments > 0) {
                float averageSpeed = totalSpeed / validSegments;
                
                // Add radar/lidar data if available and enabled
                if (useRadar && track.detectionHistory.back().type != VehicleType::MOTORCYCLE) {
                    // Simulated radar speed data, in real system would come from radar sensor
                    float radarSpeed = averageSpeed * (1.0f + ((std::rand() % 10) - 5) / 100.0f);
                    
                    // Fuse camera and radar data (weighted average)
                    averageSpeed = 0.7f * averageSpeed + 0.3f * radarSpeed;
                }
                
                return averageSpeed;
            }
        }
        catch (const std::exception& e) {
            logger.error("Error calculating speed: %s", e.what());
        }
        
        return 0.0f;
    }
    
    float getSpeedLimit(const cv::Point2f& position) {
        // Find which lane the vehicle is in
        for (const auto& lane : lanes) {
            if (cv::pointPolygonTest(lane.lanePolygon, position, false) >= 0) {
                return lane.speedLimit;
            }
        }
        
        // Default speed limit if not in any defined lane
        return DEFAULT_SPEED_LIMIT;
    }
    
    bool isSpeedViolation(float measuredSpeed, float speedLimit) {
        return measuredSpeed > (speedLimit + speedToleranceKmh);
    }
};

// Violation processor class
class ViolationProcessor {
private:
    LogManager& logger;
    DataStorage* storage;
    std::string outputPath;
    int violationCounter;
    std::mutex violationMutex;
    bool enableEncryption;
    int retentionDays;
    bool privacyMaskEnabled;
    int compressionQuality;
    std::map<int, std::chrono::system_clock::time_point> lastViolationTime;
    std::map<std::string, std::chrono::system_clock::time_point> plateViolationCooldown;
    std::string deviceId;
    GPSCoordinate deviceLocation;
    int minFramesForViolation;
    
public:
    ViolationProcessor(LogManager& logManager)
        : logger(logManager), storage(nullptr), violationCounter(0),
          enableEncryption(DEFAULT_ENABLE_ENCRYPTION), retentionDays(DEFAULT_RETENTION_DAYS),
          privacyMaskEnabled(DEFAULT_PRIVACY_MASK_ENABLED),
          compressionQuality(DEFAULT_COMPRESSION_QUALITY),
          minFramesForViolation(DEFAULT_MIN_FRAMES_FOR_VIOLATION) {}
    
    bool initialize(const ConfigManager& config, DataStorage* dataStorage) {
        storage = dataStorage;
        outputPath = config.getValue<std::string>("violations/output_path", DEFAULT_OUTPUT_PATH);
        enableEncryption = config.getValue<bool>("violations/enable_encryption", DEFAULT_ENABLE_ENCRYPTION);
        retentionDays = config.getValue<int>("violations/retention_days", DEFAULT_RETENTION_DAYS);
        privacyMaskEnabled = config.getValue<bool>("violations/privacy_mask_enabled", DEFAULT_PRIVACY_MASK_ENABLED);
        compressionQuality = config.getValue<int>("storage/compression_quality", DEFAULT_COMPRESSION_QUALITY);
        deviceId = config.getValue<std::string>("device/id", std::to_string(DEFAULT_DEVICE_ID));
        minFramesForViolation = config.getValue<int>("tracking/min_frames_for_violation", DEFAULT_MIN_FRAMES_FOR_VIOLATION);
        
        // Get device location
        double lat = config.getValue<double>("device/location/latitude", DEFAULT_GPS_LAT);
        double lon = config.getValue<double>("device/location/longitude", DEFAULT_GPS_LON);
        double alt = config.getValue<double>("device/location/altitude", 0.0);
        deviceLocation = GPSCoordinate(lat, lon, alt);
        
        // Create output directory if it doesn't exist
        if (!fs::exists(outputPath)) {
            fs::create_directories(outputPath);
        }
        
        logger.info("Violation processor initialized. Output path: %s", outputPath.c_str());
        return true;
    }
    
    std::optional<ViolationRecord> processSpeedViolation(
        const VehicleTrack& track, 
        float speedLimit, 
        const cv::Mat& frame, 
        const WeatherInfo& weather,
        int laneId) {
            
        std::lock_guard<std::mutex> lock(violationMutex);
        
        // Check if we have sufficient tracking data for a reliable violation
        if (track.positionHistory.size() < minFramesForViolation || 
            track.timestampHistory.size() < minFramesForViolation) {
            return std::nullopt;
        }
        
        // Check cooldown period for this track to avoid duplicate violations
        auto now = std::chrono::system_clock::now();
        auto it = lastViolationTime.find(track.trackId);
        if (it != lastViolationTime.end()) {
            auto timeSinceLastViolation = std::chrono::duration_cast<std::chrono::seconds>(now - it->second);
            if (timeSinceLastViolation.count() < 60) {  // 1 minute cooldown
                return std::nullopt;
            }
        }
        
        // Check plate cooldown period to avoid multiple violations for the same vehicle
        if (track.bestPlateDetection) {
            const std::string& plateNumber = track.bestPlateDetection->plateNumber;
            auto plateIt = plateViolationCooldown.find(plateNumber);
            if (plateIt != plateViolationCooldown.end()) {
                auto timeSinceLastViolation = std::chrono::duration_cast<std::chrono::minutes>(now - plateIt->second);
                if (timeSinceLastViolation.count() < 30) {  // 30 minutes cooldown for same plate
                    return std::nullopt;
                }
            }
        }
        
        // Create violation record
        ViolationRecord violation;
        violation.violationId = ++violationCounter;
        violation.type = ViolationType::SPEED_VIOLATION;
        violation.timestamp = now;
        violation.location = deviceLocation;
        violation.violationValue = track.currentSpeed;
        violation.thresholdValue = speedLimit;
        violation.vehicleType = track.detectionHistory.back().type;
        violation.laneId = laneId;
        violation.confidence = track.trackConfidence;
        violation.deviceId = deviceId;
        violation.weatherCondition = weather.condition;
        
        // Generate evidence images
        generateEvidenceImage(violation, track, frame);
        
        // Store the violation
        if (storage) {
            storage->storeViolation(violation);
        }
        
        // Update cooldown times
        lastViolationTime[track.trackId] = now;
        
        if (track.bestPlateDetection) {
            violation.plateNumber = track.bestPlateDetection->plateNumber;
            plateViolationCooldown[track.bestPlateDetection->plateNumber] = now;
            logger.info("Speed violation recorded: Vehicle with plate %s traveling at %.1f km/h in a %.0f km/h zone",
                       violation.plateNumber.c_str(), violation.violationValue, violation.thresholdValue);
        } else {
            logger.info("Speed violation recorded: Unidentified vehicle traveling at %.1f km/h in a %.0f km/h zone",
                       violation.violationValue, violation.thresholdValue);
        }
        
        return violation;
    }
    
    void cleanupOldViolations() {
        if (retentionDays <= 0 || outputPath.empty()) {
            return;
        }
        
        try {
            auto now = std::chrono::system_clock::now();
            
            for (const auto& entry : fs::directory_iterator(outputPath)) {
                if (entry.is_regular_file()) {
                    auto lastWriteTime = fs::last_write_time(entry.path());
                    auto lastWriteTimePoint = std::chrono::file_clock::to_sys(lastWriteTime);
                    auto age = std::chrono::duration_cast<std::chrono::hours>(now - lastWriteTimePoint);
                    
                    if (age.count() > retentionDays * 24) {
                        fs::remove(entry.path());
                        logger.debug("Removed old violation file: %s", entry.path().string().c_str());
                    }
                }
            }
        } catch (const std::exception& e) {
            logger.error("Error cleaning up old violations: %s", e.what());
        }
    }
    
private:
    void generateEvidenceImage(ViolationRecord& violation, const VehicleTrack& track, const cv::Mat& frame) {
        try {
            // Clone frame to avoid modifying the original
            cv::Mat evidenceImage = frame.clone();
            
            // Draw bounding box around vehicle
            if (!track.detectionHistory.empty()) {
                const auto& detection = track.detectionHistory.back();
                cv::rectangle(evidenceImage, detection.boundingBox, cv::Scalar(0, 0, 255), 3);
                
                // Store trajectory points for context
                for (const auto& pos : track.positionHistory) {
                    violation.vehiclePositions.push_back(cv::Point(pos.x, pos.y));
                }
            }
            
            // Draw trajectory
            if (violation.vehiclePositions.size() >= 2) {
                for (size_t i = 1; i < violation.vehiclePositions.size(); i++) {
                    cv::line(evidenceImage, violation.vehiclePositions[i-1], violation.vehiclePositions[i], 
                            cv::Scalar(0, 255, 255), 2);
                }
            }
            
            // Draw speed information
            std::stringstream speedText;
            speedText << "Speed: " << std::fixed << std::setprecision(1) << violation.violationValue << " km/h";
            speedText << " (Limit: " << std::fixed << std::setprecision(0) << violation.thresholdValue << " km/h)";
            
            cv::putText(evidenceImage, speedText.str(), cv::Point(20, 40), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
            
            // Draw timestamp
            auto time_t = std::chrono::system_clock::to_time_t(violation.timestamp);
            std::stringstream timeText;
            timeText << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
            
            cv::putText(evidenceImage, timeText.str(), cv::Point(20, 80), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
            
            // Draw location information
            std::stringstream locationText;
            locationText << "Loc: " << std::fixed << std::setprecision(6) 
                        << violation.location.latitude << ", " << violation.location.longitude;
            
            cv::putText(evidenceImage, locationText.str(), cv::Point(20, 120), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
            
            // Draw vehicle information
            std::string vehicleText = "Vehicle: ";
            switch (violation.vehicleType) {
                case VehicleType::CAR: vehicleText += "Car"; break;
                case VehicleType::MOTORCYCLE: vehicleText += "Motorcycle"; break;
                case VehicleType::BUS: vehicleText += "Bus"; break;
                case VehicleType::TRUCK_SMALL: vehicleText += "Truck"; break;
                case VehicleType::TRUCK_LARGE: vehicleText += "Heavy Truck"; break;
                default: vehicleText += "Unknown";
            }
            
            if (!track.detectionHistory.empty() && !track.detectionHistory.back().colorName.empty()) {
                vehicleText += " (" + track.detectionHistory.back().colorName + ")";
            }
            
            cv::putText(evidenceImage, vehicleText, cv::Point(20, 160), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
            
            // Draw license plate information if available
            if (track.bestPlateDetection) {
                std::string plateText = "Plate: " + track.bestPlateDetection->plateNumber;
                cv::putText(evidenceImage, plateText, cv::Point(20, 200), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
                
                // Highlight the plate area
                cv::rectangle(evidenceImage, track.bestPlateDetection->boundingBox, cv::Scalar(0, 255, 0), 2);
            }
            
            // Apply privacy mask if enabled (blur pedestrian faces, etc.)
            if (privacyMaskEnabled) {
                applyPrivacyMask(evidenceImage);
            }
            
            // Store evidence image
            violation.evidenceImage = evidenceImage;
            
            // Save to file
            saveViolationImage(violation);
            
        } catch (const std::exception& e) {
            logger.error("Error generating evidence image: %s", e.what());
        }
    }
    
    void applyPrivacyMask(cv::Mat& image) {
        // In a real system, this would detect and blur faces or other private information
        // For this example, we'll just add a simple placeholder
        
        // Add a privacy notice
        cv::putText(image, "Privacy protection active", cv::Point(image.cols - 300, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }
    
    void saveViolationImage(const ViolationRecord& violation) {
        try {
            // Generate filename
            std::string filename = violation.generateFilename();
            std::string filepath = outputPath + "/" + filename;
            
            // Compression parameters
            std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, compressionQuality};
            
            // Save the image
            cv::imwrite(filepath, violation.evidenceImage, params);
            
            logger.debug("Saved violation evidence to %s", filepath.c_str());
        } catch (const std::exception& e) {
            logger.error("Error saving violation image: %s", e.what());
        }
    }
};

/******************************************************************************
 * Main Speed Camera Class
 ******************************************************************************/

class SpeedCamera {
private:
    // Core components
    ConfigManager configManager;
    LogManager logger;
    VehicleDetector vehicleDetector;
    PlateRecognizer plateRecognizer;
    SpeedCalculator speedCalculator;
    ViolationProcessor violationProcessor;
    
    // Video capture
    cv::VideoCapture videoCapture;
    cv::VideoWriter videoWriter;
    bool isInitialized;
    bool isRunning;
    std::atomic<bool> shouldStop;
    
    // Processing thread
    std::thread processingThread;
    
    // Frame buffers
    std::queue<std::pair<cv::Mat, std::chrono::system_clock::time_point>> frameBuffer;
    std::mutex frameMutex;
    std::condition_variable frameCondition;
    
    // Vehicle tracking
    std::vector<VehicleTrack> activeTracks;
    std::mutex trackMutex;
    int nextTrackId;
    TrackingAlgorithm trackingAlgorithm;
    float trackingConfidenceThreshold;
    int maxTracks;
    
    // Camera and frame properties
    int frameWidth;
    int frameHeight;
    int targetFps;
    CameraType cameraType;
    
    // Calibration
    CalibrationParameters calibration;
    
    // Speed enforcement
    float defaultSpeedLimit;
    std::vector<TrafficLane> lanes;
    
    // Statistics
    int processedFrames;
    int detectedVehicles;
    int recordedViolations;
    std::chrono::system_clock::time_point startTime;
    
    // Environment
    WeatherInfo currentWeather;
    
    // Processing settings
    bool showVisualization;
    bool recordVideo;
    bool processTrafficSignals;
    bool useLaneDiscipline;
    int processingResolutionWidth;
    int processingResolutionHeight;
    
    // Storage
    std::unique_ptr<DataStorage> storage;
    
public:
    SpeedCamera(const std::string& configFile = DEFAULT_CONFIG_FILE)
        : configManager(configFile), vehicleDetector(logger), plateRecognizer(logger),
          speedCalculator(logger), violationProcessor(logger),
          isInitialized(false), isRunning(false), shouldStop(false),
          nextTrackId(1), trackingAlgorithm(TrackingAlgorithm::DEEP_SORT),
          trackingConfidenceThreshold(DEFAULT_TRACKING_CONFIDENCE),
          maxTracks(DEFAULT_MAX_VEHICLES_TRACK),
          frameWidth(DEFAULT_CAMERA_WIDTH), frameHeight(DEFAULT_CAMERA_HEIGHT),
          targetFps(DEFAULT_CAMERA_FPS), cameraType(CameraType::FIXED),
          defaultSpeedLimit(DEFAULT_SPEED_LIMIT),
          processedFrames(0), detectedVehicles(0), recordedViolations(0),
          startTime(std::chrono::system_clock::now()),
          showVisualization(true), recordVideo(false), processTrafficSignals(false),
          useLaneDiscipline(true),
          processingResolutionWidth(1280), processingResolutionHeight(720) {
        
        // Start the logger
        logger.setLogLevel(DEFAULT_LOG_LEVEL);
        logger.info("SpeedGuardian Pro Version %s (Build: %s %s)",
                   VERSION.c_str(), BUILD_DATE.c_str(), BUILD_TIME.c_str());
    }
    
    ~SpeedCamera() {
        stop();
    }
    
    bool initialize() {
        logger.info("Initializing SpeedGuardian Pro system...");
        
        try {
            // Load configuration
            if (!configManager.loadConfig()) {
                logger.error("Failed to load configuration");
                return false;
            }
            
            // Set log level from config
            std::string logLevel = configManager.getValue<std::string>("system/log_level", DEFAULT_LOG_LEVEL);
            logger.setLogLevel(logLevel);
            
            // Initialize camera settings
            frameWidth = configManager.getValue<int>("camera/width", DEFAULT_CAMERA_WIDTH);
            frameHeight = configManager.getValue<int>("camera/height", DEFAULT_CAMERA_HEIGHT);
            targetFps = configManager.getValue<int>("camera/fps", DEFAULT_CAMERA_FPS);
            std::string cameraSource = configManager.getValue<std::string>("camera/source", "0");
            cameraType = static_cast<CameraType>(configManager.getValue<int>("device/camera_type", 
                                               static_cast<int>(CameraType::FIXED)));
            
            // Initialize tracking settings
            trackingAlgorithm = static_cast<TrackingAlgorithm>(
                configManager.getValue<int>("tracking/algorithm", static_cast<int>(TrackingAlgorithm::DEEP_SORT)));
            trackingConfidenceThreshold = configManager.getValue<float>("tracking/tracking_confidence", 
                                                            DEFAULT_TRACKING_CONFIDENCE);
            maxTracks = configManager.getValue<int>("tracking/max_vehicles", DEFAULT_MAX_VEHICLES_TRACK);
            
            // Initialize speed settings
            defaultSpeedLimit = configManager.getValue<float>("speed/speed_limit", DEFAULT_SPEED_LIMIT);
            
            // Initialize visualization settings
            showVisualization = configManager.getValue<bool>("system/show_visualization", true);
            recordVideo = configManager.getValue<bool>("system/record_video", false);
            
            // Initialize processing settings
            processTrafficSignals = configManager.getValue<bool>("violations/types/red_light", false);
            useLaneDiscipline = true;
            
            // Initialize lanes
            initializeLanes();
            
            // Initialize storage
            storage = std::make_unique<DataStorage>(logger);
            if (!storage->initialize(configManager)) {
                logger.error("Failed to initialize data storage");
                return false;
            }
            
            // Initialize calibration
            initializeCalibration();
            
            // Initialize components
            if (!vehicleDetector.initialize(configManager)) {
                logger.error("Failed to initialize vehicle detector");
                return false;
            }
            
            if (!plateRecognizer.initialize(configManager)) {
                logger.warning("Failed to initialize plate recognizer, continuing without plate recognition");
            }
            
            if (!speedCalculator.initialize(configManager, calibration)) {
                logger.error("Failed to initialize speed calculator");
                return false;
            }
            
            if (!violationProcessor.initialize(configManager, storage.get())) {
                logger.error("Failed to initialize violation processor");
                return false;
            }
            
            // Set lanes for speed calculation
            speedCalculator.setLanes(lanes);
            
            // Initialize video capture
            if (!openVideoSource(cameraSource)) {
                logger.error("Failed to open video source: %s", cameraSource.c_str());
                return false;
            }
            
            if (recordVideo) {
                initializeVideoWriter();
            }
            
            // Initialize weather
            updateWeatherInfo();
            
            isInitialized = true;
            logger.info("SpeedGuardian Pro initialized successfully");
            return true;
        }
        catch (const std::exception& e) {
            logger.critical("Initialization failed with exception: %s", e.what());
            return false;
        }
    }
    
    bool start() {
        if (!isInitialized) {
            logger.error("Cannot start: system not initialized");
            return false;
        }
        
        if (isRunning) {
            logger.warning("System already running");
            return true;
        }
        
        logger.info("Starting SpeedGuardian Pro...");
        
        isRunning = true;
        shouldStop = false;
        
        // Start processing thread
        processingThread = std::thread(&SpeedCamera::processingLoop, this);
        
        logger.info("SpeedGuardian Pro started");
        return true;
    }
    
    void stop() {
        if (!isRunning) {
            return;
        }
        
        logger.info("Stopping SpeedGuardian Pro...");
        
        // Signal processing thread to stop
        shouldStop = true;
        
        // Notify frame condition in case thread is waiting
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            frameCondition.notify_all();
        }
        
        // Wait for processing thread to finish
        if (processingThread.joinable()) {
            processingThread.join();
        }
        
        // Release video resources
        if (videoCapture.isOpened()) {
            videoCapture.release();
        }
        
        if (videoWriter.isOpened()) {
            videoWriter.release();
        }
        
        isRunning = false;
        logger.info("SpeedGuardian Pro stopped");
    }
    
    bool isSystemRunning() const {
        return isRunning;
    }
    
    void getStats(int& frames, int& vehicles, int& violations) const {
        frames = processedFrames;
        vehicles = detectedVehicles;
        violations = recordedViolations;
    }
    
    cv::Mat getLatestProcessedFrame() {
        // For visualization purposes
        std::lock_guard<std::mutex> lock(frameMutex);
        if (!frameBuffer.empty()) {
            return frameBuffer.back().first;
        }
        return cv::Mat();
    }
    
private:
    bool openVideoSource(const std::string& source) {
        // Try to interpret source as a number (camera index)
        try {
            int cameraIndex = std::stoi(source);
            if (!videoCapture.open(cameraIndex)) {
                logger.error("Failed to open camera with index %d", cameraIndex);
                return false;
            }
        }
        catch (const std::exception&) {
            // Not a number, try as a file or stream URL
            if (!videoCapture.open(source)) {
                logger.error("Failed to open video source: %s", source.c_str());
                return false;
            }
        }
        
        // Set camera properties
        videoCapture.set(cv::CAP_PROP_FRAME_WIDTH, frameWidth);
        videoCapture.set(cv::CAP_PROP_FRAME_HEIGHT, frameHeight);
        videoCapture.set(cv::CAP_PROP_FPS, targetFps);
        
        // Get actual properties (may differ from requested)
        frameWidth = videoCapture.get(cv::CAP_PROP_FRAME_WIDTH);
        frameHeight = videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
        targetFps = videoCapture.get(cv::CAP_PROP_FPS);
        
        logger.info("Video source opened: %dx%d @ %d FPS", frameWidth, frameHeight, targetFps);
        return true;
    }
    
    void initializeVideoWriter() {
        std::string filename = generateTimestampedFilename("output", "mp4");
        int fourcc = cv::VideoWriter::fourcc('X', '2', '6', '4');
        bool isColor = true;
        
        try {
            videoWriter.open(filename, fourcc, targetFps, cv::Size(frameWidth, frameHeight), isColor);
            if (videoWriter.isOpened()) {
                logger.info("Video recording started: %s", filename.c_str());
            } else {
                logger.error("Failed to initialize video writer");
            }
        }
        catch (const cv::Exception& e) {
            logger.error("Error initializing video writer: %s", e.what());
        }
    }
    
    std::string generateTimestampedFilename(const std::string& prefix, const std::string& extension) {
        auto now = std::chrono::system_clock::now();
        auto timeT = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << prefix << "_" << std::put_time(std::localtime(&timeT), "%Y%m%d_%H%M%S") << "." << extension;
        return ss.str();
    }
    
    void initializeCalibration() {
        // Load calibration parameters from config or use defaults
        calibration.cameraHeight = configManager.getValue<double>("device/camera_height", DEFAULT_CAMERA_HEIGHT_METERS);
        calibration.cameraTilt = configManager.getValue<double>("device/camera_angle", DEFAULT_CAMERA_ANGLE_DEGREES);
        
        // In a real system, this would load a proper calibration file
        // For this example, we'll set up a simple approximation
        
        // Set up camera matrix (intrinsic parameters)
        double focalLength = frameWidth * 1.2; // Approximate focal length
        cv::Point2d principalPoint(frameWidth / 2.0, frameHeight / 2.0);
        
        calibration.cameraMatrix = (cv::Mat_<double>(3, 3) << 
            focalLength, 0, principalPoint.x,
            0, focalLength, principalPoint.y,
            0, 0, 1);
        
        // Set up distortion coefficients (assume no distortion for simplicity)
        calibration.distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
        
        // Set up homography matrix for ground plane projection
        // This would normally be calculated from calibration points
        // For now, use a simple perspective transform
        
        // Source points (in image coordinates)
        std::vector<cv::Point2f> srcPoints = {
            cv::Point2f(frameWidth * 0.25f, frameHeight * 0.65f),
            cv::Point2f(frameWidth * 0.75f, frameHeight * 0.65f),
            cv::Point2f(frameWidth * 0.75f, frameHeight * 0.95f),
            cv::Point2f(frameWidth * 0.25f, frameHeight * 0.95f)
        };
        
        // Destination points (in world coordinates, meters)
        float worldWidth = 7.0f; // typical road width
        float worldLength = 20.0f; // visible road length
        std::vector<cv::Point2f> dstPoints = {
            cv::Point2f(-worldWidth/2, 0),
            cv::Point2f(worldWidth/2, 0),
            cv::Point2f(worldWidth/2, worldLength),
            cv::Point2f(-worldWidth/2, worldLength)
        };
        
        // Calculate the homography
        calibration.homographyMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);
        
        // Calculate pixels per meter
        cv::Point2f imagePoint1(frameWidth / 2, frameHeight * 0.75);
        cv::Point2f imagePoint2(frameWidth / 2, frameHeight * 0.95);
        
        std::vector<cv::Point2f> imagePoints = {imagePoint1, imagePoint2};
        std::vector<cv::Point2f> worldPoints;
        
        cv::perspectiveTransform(imagePoints, worldPoints, calibration.homographyMatrix);
        
        float worldDistance = cv::norm(worldPoints[0] - worldPoints[1]);
        float imageDistance = cv::norm(imagePoint1 - imagePoint2);
        
        calibration.pixelsPerMeter = imageDistance / worldDistance;
        
        // Set calibration flags
        calibration.isCalibrated = true;
        calibration.calibrationConfidence = 0.8f;
        calibration.calibrationTime = std::chrono::system_clock::now();
        
        logger.info("Camera calibration initialized: %.2f pixels/meter, height=%.1fm, tilt=%.1f°",
                   calibration.pixelsPerMeter, calibration.cameraHeight, calibration.cameraTilt);
    }
    
    void initializeLanes() {
        // Load lanes from config
        json lanesConfig = configManager.getValue<json>("lanes", json::array());
        
        if (lanesConfig.empty()) {
            // Create default lanes if none are defined
            createDefaultLanes();
        }
        else {
            // Load lanes from config
            for (const auto& laneConfig : lanesConfig) {
                int laneId = laneConfig["id"];
                TrafficLane lane(laneId);
                
                lane.speedLimit = laneConfig.value("speed_limit", DEFAULT_SPEED_LIMIT);
                lane.laneType = laneConfig.value("type", "normal");
                lane.isActive = laneConfig.value("active", true);
                lane.direction = laneConfig.value("direction", "inbound");
                lane.width = laneConfig.value("width", 3.5f);
                lane.enforceSpeedLimit = laneConfig.value("enforce_speed_limit", true);
                
                // Load polygon points
                if (laneConfig.contains("polygon")) {
                    for (const auto& point : laneConfig["polygon"]) {
                        lane.lanePolygon.push_back(cv::Point(point[0], point[1]));
                    }
                }
                
                lanes.push_back(lane);
            }
        }
        
        logger.info("Initialized %zu traffic lanes", lanes.size());
    }
    
    void createDefaultLanes() {
        // Create default lanes based on frame size
        int laneWidth = frameWidth / 3;
        
        // Inbound lane (right to left)
        TrafficLane inboundLane(1);
        inboundLane.speedLimit = defaultSpeedLimit;
        inboundLane.laneType = "normal";
        inboundLane.direction = "inbound";
        inboundLane.width = 3.5f;
        
        inboundLane.lanePolygon = {
            cv::Point(0, frameHeight/2),
            cv::Point(frameWidth, frameHeight/2),
            cv::Point(frameWidth, frameHeight*3/4),
            cv::Point(0, frameHeight*3/4)
        };
        
        // Outbound lane (left to right)
        TrafficLane outboundLane(2);
        outboundLane.speedLimit = defaultSpeedLimit;
        outboundLane.laneType = "normal";
        outboundLane.direction = "outbound";
        outboundLane.width = 3.5f;
        
        outboundLane.lanePolygon = {
            cv::Point(0, frameHeight*3/4),
            cv::Point(frameWidth, frameHeight*3/4),
            cv::Point(frameWidth, frameHeight),
            cv::Point(0, frameHeight)
        };
        
        lanes.push_back(inboundLane);
        lanes.push_back(outboundLane);
    }
    
    void updateWeatherInfo() {
        // In a real system, this would get data from weather sensors or APIs
        // For this example, we'll just simulate weather conditions
        
        currentWeather.condition = WeatherCondition::CLEAR;
        currentWeather.temperature = 20.0f;
        currentWeather.humidity = 50.0f;
        currentWeather.windSpeed = 5.0f;
        currentWeather.precipitation = 0.0f;
        currentWeather.visibility = 10.0f;
        currentWeather.lightLevel = 10000.0f;
        currentWeather.timestamp = std::chrono::system_clock::now();
        
        // Determine if it's night time
        auto time_t = std::chrono::system_clock::to_time_t(currentWeather.timestamp);
        std::tm* timeInfo = std::localtime(&time_t);
        
        if (timeInfo->tm_hour >= 20 || timeInfo->tm_hour < 6) {
            currentWeather.isNight = true;
            currentWeather.timeOfDay = TimeOfDay::NIGHT;
            currentWeather.lightLevel = 10.0f;
        }
        else if (timeInfo->tm_hour >= 18) {
            currentWeather.timeOfDay = TimeOfDay::EVENING;
        }
        else if (timeInfo->tm_hour >= 12) {
            currentWeather.timeOfDay = TimeOfDay::AFTERNOON;
        }
        else if (timeInfo->tm_hour >= 8) {
            currentWeather.timeOfDay = TimeOfDay::MORNING;
        }
        else {
            currentWeather.timeOfDay = TimeOfDay::DAWN;
        }
    }
    
    void processingLoop() {
        logger.info("Processing loop started");
        
        while (!shouldStop) {
            // Read frame from camera
            cv::Mat frame;
            if (!videoCapture.read(frame)) {
                logger.warning("Failed to read frame from video source");
                // Check if video source is a file and has reached the end
                if (videoCapture.get(cv::CAP_PROP_POS_FRAMES) >= videoCapture.get(cv::CAP_PROP_FRAME_COUNT) - 1) {
                    logger.info("Reached end of video file");
                    shouldStop = true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            
            auto frameTimestamp = std::chrono::system_clock::now();
            
            // Process frame
            processFrame(frame, frameTimestamp);
        }
        
        logger.info("Processing loop ended");
    }
    
    void processFrame(cv::Mat& frame, std::chrono::system_clock::time_point timestamp) {
        // Skip if frame is empty
        if (frame.empty()) {
            return;
        }
        
        // Apply calibration undistortion if needed
        if (calibration.isCalibrated) {
            frame = calibration.undistortImage(frame);
        }
        
        // Detect vehicles
        std::vector<VehicleDetection> detections = vehicleDetector.detectVehicles(frame);
        
        // Update tracking information
        updateTracks(detections, frame, timestamp);
        
        // Detect license plates
        std::vector<PlateDetection> plateDetections = plateRecognizer.detectPlates(frame, detections);
        
        // Associate plates with vehicle tracks
        associatePlatesWithTracks(plateDetections);
        
        // Calculate speeds for each track
        calculateSpeeds();
        
        // Check for violations
        checkViolations(frame);
        
        // Draw visualization if enabled
        if (showVisualization) {
            drawVisualization(frame);
        }
        
        // Record video if enabled
        if (recordVideo && videoWriter.isOpened()) {
            videoWriter.write(frame);
        }
        
        // Store frame in buffer
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            frameBuffer.push(std::make_pair(frame.clone(), timestamp));
            
            // Keep buffer size limited
            while (frameBuffer.size() > 10) {
                frameBuffer.pop();
            }
        }
        
        // Update stats
        processedFrames++;
        detectedVehicles += detections.size();
        
        // Periodically update weather conditions (every 5 minutes)
        static auto lastWeatherUpdate = std::chrono::system_clock::now();
        auto now = std::chrono::system_clock::now();
        if (std::chrono::duration_cast<std::chrono::minutes>(now - lastWeatherUpdate).count() >= 5) {
            updateWeatherInfo();
            lastWeatherUpdate = now;
        }
        
        // Periodically clean up old violations (once per hour)
        static auto lastCleanup = std::chrono::system_clock::now();
        if (std::chrono::duration_cast<std::chrono::hours>(now - lastCleanup).count() >= 1) {
            violationProcessor.cleanupOldViolations();
            lastCleanup = now;
        }
    }
    
    void updateTracks(const std::vector<VehicleDetection>& detections, 
                     const cv::Mat& frame,
                     std::chrono::system_clock::time_point timestamp) {
        std::lock_guard<std::mutex> lock(trackMutex);
        
        // Predict new locations of existing tracks
        std::vector<cv::Point2f> predictedPositions;
        for (auto& track : activeTracks) {
            cv::Point2f predictedPos = track.predictNextPosition();
            predictedPositions.push_back(predictedPos);
        }
        
        // Create cost matrix for assignment
        cv::Mat costMatrix(activeTracks.size(), detections.size(), CV_32F, cv::Scalar(std::numeric_limits<float>::max()));
        
        for (size_t i = 0; i < activeTracks.size(); i++) {
            const auto& track = activeTracks[i];
            cv::Point2f predictedPos = predictedPositions[i];
            
            for (size_t j = 0; j < detections.size(); j++) {
                const auto& detection = detections[j];
                
                // Calculate center of detection
                cv::Point2f detectionCenter(
                    detection.boundingBox.x + detection.boundingBox.width / 2.0f,
                    detection.boundingBox.y + detection.boundingBox.height / 2.0f
                );
                
                // Calculate distance between predicted position and detection
                float distance = cv::norm(predictedPos - detectionCenter);
                
                // Check if types are compatible
                bool typeCompatible = (track.detectionHistory.back().type == detection.type) ||
                                     (track.detectionHistory.back().type == VehicleType::UNKNOWN) ||
                                     (detection.type == VehicleType::UNKNOWN);
                
                // Set cost based on distance and type compatibility
                if (typeCompatible && distance < 200.0f) { // Maximum association distance
                    costMatrix.at<float>(i, j) = distance;
                }
            }
        }
        
        // Solve assignment problem using Hungarian algorithm
        std::vector<int> assignment;
        solveAssignment(costMatrix, assignment);
        
        // Mark all tracks as unmatched initially
        std::vector<bool> matchedTracks(activeTracks.size(), false);
        std::vector<bool> matchedDetections(detections.size(), false);
        
        // Update matched tracks
        for (size_t i = 0; i < assignment.size(); i++) {
            int detectionIdx = assignment[i];
            
            if (detectionIdx >= 0 && detectionIdx < static_cast<int>(detections.size()) &&
                costMatrix.at<float>(i, detectionIdx) < std::numeric_limits<float>::max()) {
                
                // Update track with new detection
                activeTracks[i].update(detections[detectionIdx]);
                matchedTracks[i] = true;
                matchedDetections[detectionIdx] = true;
            }
        }
        
        // Handle unmatched tracks
        for (size_t i = 0; i < activeTracks.size(); i++) {
            if (!matchedTracks[i]) {
                // Increment miss counter
                activeTracks[i].consecutiveMisses++;
                
                // Reduce confidence
                activeTracks[i].trackConfidence *= 0.9f;
                
                // Mark for removal if confidence too low or too many consecutive misses
                if (activeTracks[i].trackConfidence < 0.1f || activeTracks[i].consecutiveMisses > 10) {
                    activeTracks[i].isActive = false;
                }
            }
        }
        
        // Create new tracks for unmatched detections
        for (size_t i = 0; i < detections.size(); i++) {
            if (!matchedDetections[i] && detections[i].confidence > trackingConfidenceThreshold) {
                // Create new track
                VehicleTrack newTrack(nextTrackId++, detections[i]);
                activeTracks.push_back(newTrack);
            }
        }
        
        // Remove inactive tracks
        activeTracks.erase(
            std::remove_if(activeTracks.begin(), activeTracks.end(),
                [](const VehicleTrack& track) { return !track.isActive; }),
            activeTracks.end()
        );
        
        // Limit maximum number of active tracks
        if (activeTracks.size() > maxTracks) {
            // Sort by confidence and keep only the top maxTracks
            std::partial_sort(
                activeTracks.begin(), 
                activeTracks.begin() + maxTracks, 
                activeTracks.end(),
                [](const VehicleTrack& a, const VehicleTrack& b) {
                    return a.trackConfidence > b.trackConfidence;
                }
            );
            
            // Remove excess tracks
            activeTracks.resize(maxTracks);
        }
    }
    
    void solveAssignment(const cv::Mat& costMatrix, std::vector<int>& assignment) {
        // Hungarian algorithm implementation
        // For simplicity in this example, we'll use a greedy approach
        
        assignment.resize(costMatrix.rows, -1);
        
        // Make a copy of the cost matrix that we can modify
        cv::Mat costs = costMatrix.clone();
        
        std::vector<bool> usedCols(costMatrix.cols, false);
        
        for (int row = 0; row < costs.rows; row++) {
            float minCost = std::numeric_limits<float>::max();
            int minCol = -1;
            
            // Find minimum cost in this row
            for (int col = 0; col < costs.cols; col++) {
                if (!usedCols[col] && costs.at<float>(row, col) < minCost) {
                    minCost = costs.at<float>(row, col);
                    minCol = col;
                }
            }
            
            // Assign if we found a valid minimum
            if (minCol >= 0 && minCost < std::numeric_limits<float>::max()) {
                assignment[row] = minCol;
                usedCols[minCol] = true;
            }
        }
    }
    
    void associatePlatesWithTracks(const std::vector<PlateDetection>& plateDetections) {
        std::lock_guard<std::mutex> lock(trackMutex);
        
        for (const auto& plate : plateDetections) {
            // Find track associated with this plate
            for (auto& track : activeTracks) {
                // Check if the detection contains or overlaps with the plate
                if (!track.detectionHistory.empty()) {
                    const auto& detection = track.detectionHistory.back();
                    
                    // Check for overlap
                    cv::Rect intersection = detection.boundingBox & plate.boundingBox;
                    if (intersection.area() > 0 ||
                        detection.boundingBox.contains(cv::Point(
                            plate.boundingBox.x + plate.boundingBox.width / 2,
                            plate.boundingBox.y + plate.boundingBox.height / 2))) {
                        
                        // Update plate information
                        if (!track.bestPlateDetection ||
                            plate.confidence > track.bestPlateDetection->confidence) {
                            track.bestPlateDetection = plate;
                        }
                        
                        break;
                    }
                }
            }
        }
    }
    
    void calculateSpeeds() {
        std::lock_guard<std::mutex> lock(trackMutex);
        
        for (auto& track : activeTracks) {
            // Only calculate speed if we have enough tracking history
            if (track.positionHistory.size() >= 5 && track.trackConfidence > 0.5f) {
                float speed = speedCalculator.calculateSpeed(track);
                
                if (speed > 0) {
                    track.updateSpeed(speed);
                }
            }
        }
    }
    
    void checkViolations(const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(trackMutex);
        
        for (auto& track : activeTracks) {
            // Skip tracks with low confidence or insufficient history
            if (track.trackConfidence < 0.6f || track.positionHistory.size() < 5) {
                continue;
            }
            
            // Get last position
            cv::Point2f position = track.positionHistory.back();
            
            // Find which lane the vehicle is in
            int laneId = -1;
            float speedLimit = defaultSpeedLimit;
            
            for (const auto& lane : lanes) {
                if (cv::pointPolygonTest(lane.lanePolygon, position, false) >= 0) {
                    laneId = lane.laneId;
                    speedLimit = lane.speedLimit;
                    break;
                }
            }
            
            // Check for speed violation
            if (track.currentSpeed > 0 && track.isSpeedViolation(speedLimit, DEFAULT_SPEED_TOLERANCE)) {
                // Process violation
                auto violation = violationProcessor.processSpeedViolation(
                    track, speedLimit, frame, currentWeather, laneId);
                
                if (violation) {
                    recordedViolations++;
                }
            }
        }
    }
    
    void drawVisualization(cv::Mat& frame) {
        // Draw lanes
        for (const auto& lane : lanes) {
            // Draw lane polygon
            std::vector<std::vector<cv::Point>> contours = {lane.lanePolygon};
            cv::polylines(frame, contours, true, cv::Scalar(0, 255, 0), 2);
            
            // Draw lane ID and speed limit
            cv::Point textPos = lane.lanePolygon[0];
            textPos.y -= 10;
            
            std::stringstream laneText;
            laneText << "Lane " << lane.laneId << " (" << lane.speedLimit << " km/h)";
            
            cv::putText(frame, laneText.str(), textPos, cv::FONT_HERSHEY_SIMPLEX, 
                       0.7, cv::Scalar(0, 255, 0), 2);
        }
        
        // Lock to safely access tracks
        std::lock_guard<std::mutex> lock(trackMutex);
        
        // Draw tracks
        for (const auto& track : activeTracks) {
            if (track.positionHistory.size() < 2) {
                continue;
            }
            
            // Draw trajectory
            for (size_t i = 1; i < track.positionHistory.size(); i++) {
                cv::line(frame, track.positionHistory[i-1], track.positionHistory[i], 
                        track.color, 2);
            }
            
            // Draw current detection
            if (!track.detectionHistory.empty()) {
                const auto& detection = track.detectionHistory.back();
                cv::rectangle(frame, detection.boundingBox, track.color, 2);
                
                // Draw ID, type, and speed
                std::stringstream trackText;
                trackText << "ID: " << track.trackId << " ";
                
                switch (detection.type) {
                    case VehicleType::CAR: trackText << "Car"; break;
                    case VehicleType::MOTORCYCLE: trackText << "Motorcycle"; break;
                    case VehicleType::BUS: trackText << "Bus"; break;
                    case VehicleType::TRUCK_SMALL: trackText << "Truck"; break;
                    case VehicleType::TRUCK_LARGE: trackText << "Heavy Truck"; break;
                    default: trackText << "Vehicle";
                }
                
                if (track.currentSpeed > 0) {
                    trackText << " " << std::fixed << std::setprecision(1) << track.currentSpeed << " km/h";
                }
                
                cv::putText(frame, trackText.str(), 
                           cv::Point(detection.boundingBox.x, detection.boundingBox.y - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, track.color, 2);
                
                // Draw plate number if available
                if (track.bestPlateDetection) {
                    std::string plateText = "Plate: " + track.bestPlateDetection->plateNumber;
                    cv::putText(frame, plateText,
                               cv::Point(detection.boundingBox.x, detection.boundingBox.y - 25),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, track.color, 2);
                    
                    // Draw plate bounding box
                    cv::rectangle(frame, track.bestPlateDetection->boundingBox, cv::Scalar(0, 255, 0), 2);
                }
            }
        }
        
        // Draw statistics
        std::stringstream statsText;
        statsText << "Frames: " << processedFrames 
                 << " | Vehicles: " << detectedVehicles
                 << " | Violations: " << recordedViolations;
        
        cv::putText(frame, statsText.str(), cv::Point(10, frame.rows - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        
        // Draw weather info
        std::stringstream weatherText;
        weatherText << "Weather: " << currentWeather.getConditionString()
                   << " | Visibility: " << std::fixed << std::setprecision(1) << currentWeather.visibility << " km";
        
        cv::putText(frame, weatherText.str(), cv::Point(10, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        
        // Draw system status
        auto now = std::chrono::system_clock::now();
        auto uptime = std::chrono::duration_cast<std::chrono::hours>(now - startTime);
        
        std::stringstream statusText;
        statusText << "SpeedGuardian Pro v" << VERSION 
                  << " | Uptime: " << uptime.count() << "h";
        
        cv::putText(frame, statusText.str(), cv::Point(10, 60),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }
};

// Data storage class (stub implementation)
class DataStorage {
private:
    std::string dbPath;
    sqlite3* db;
    LogManager& logger;
    bool isInitialized;
    
public:
    DataStorage(LogManager& logManager)
        : db(nullptr), logger(logManager), isInitialized(false) {}
    
    ~DataStorage() {
        if (db) {
            sqlite3_close(db);
        }
    }
    
    bool initialize(const ConfigManager& config) {
        dbPath = config.getValue<std::string>("storage/database_path", DEFAULT_DATABASE_PATH);
        
        // Create directory if it doesn't exist
        fs::path dbDir = fs::path(dbPath).parent_path();
        if (!dbDir.empty() && !fs::exists(dbDir)) {
            fs::create_directories(dbDir);
        }
        
        // Open database connection
        int rc = sqlite3_open(dbPath.c_str(), &db);
        if (rc != SQLITE_OK) {
            logger.error("Cannot open database: %s", sqlite3_errmsg(db));
            sqlite3_close(db);
            db = nullptr;
            return false;
        }
        
        // Create violations table if it doesn't exist
        const char* createTableSql = 
            "CREATE TABLE IF NOT EXISTS violations ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "violation_type INTEGER NOT NULL,"
            "timestamp TEXT NOT NULL,"
            "latitude REAL,"
            "longitude REAL,"
            "plate_number TEXT,"
            "violation_value REAL,"
            "threshold_value REAL,"
            "vehicle_type INTEGER,"
            "lane_id INTEGER,"
            "confidence REAL,"
            "device_id TEXT,"
            "weather_condition INTEGER,"
            "json_data TEXT,"
            "image_path TEXT"
            ");";
        
        char* errMsg = nullptr;
        rc = sqlite3_exec(db, createTableSql, nullptr, nullptr, &errMsg);
        
        if (rc != SQLITE_OK) {
            logger.error("SQL error: %s", errMsg);
            sqlite3_free(errMsg);
            return false;
        }
        
        isInitialized = true;
        logger.info("Data storage initialized with database: %s", dbPath.c_str());
        return true;
    }
    
    bool storeViolation(const ViolationRecord& violation) {
        if (!isInitialized || !db) {
            logger.error("Cannot store violation: database not initialized");
            return false;
        }
        
        // Convert timestamp to string
        auto timeT = std::chrono::system_clock::to_time_t(violation.timestamp);
        std::stringstream timestampStr;
        timestampStr << std::put_time(std::localtime(&timeT), "%Y-%m-%d %H:%M:%S");
        
        // Prepare SQL statement
        sqlite3_stmt* stmt;
        const char* sql = 
            "INSERT INTO violations "
            "(violation_type, timestamp, latitude, longitude, plate_number, "
            "violation_value, threshold_value, vehicle_type, lane_id, confidence, "
            "device_id, weather_condition, json_data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);";
        
        int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            logger.error("Failed to prepare SQL statement: %s", sqlite3_errmsg(db));
            return false;
        }
        
        // Bind parameters
        sqlite3_bind_int(stmt, 1, static_cast<int>(violation.type));
        sqlite3_bind_text(stmt, 2, timestampStr.str().c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt, 3, violation.location.latitude);
        sqlite3_bind_double(stmt, 4, violation.location.longitude);
        sqlite3_bind_text(stmt, 5, violation.plateNumber.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt, 6, violation.violationValue);
        sqlite3_bind_double(stmt, 7, violation.thresholdValue);
        sqlite3_bind_int(stmt, 8, static_cast<int>(violation.vehicleType));
        sqlite3_bind_int(stmt, 9, violation.laneId);
        sqlite3_bind_double(stmt, 10, violation.confidence);
        sqlite3_bind_text(stmt, 11, violation.deviceId.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(stmt, 12, static_cast<int>(violation.weatherCondition));
        
        // Convert metadata to JSON string
        std::string jsonStr = violation.toJson();
        sqlite3_bind_text(stmt, 13, jsonStr.c_str(), -1, SQLITE_TRANSIENT);
        
        // Execute statement
        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            logger.error("Failed to execute SQL statement: %s", sqlite3_errmsg(db));
            sqlite3_finalize(stmt);
            return false;
        }
        
        // Clean up
        sqlite3_finalize(stmt);
        
        logger.debug("Violation record stored in database with ID %d", sqlite3_last_insert_rowid(db));
        return true;
    }
    
    std::vector<ViolationRecord> getViolations(int limit = 100, int offset = 0) {
        std::vector<ViolationRecord> violations;
        
        if (!isInitialized || !db) {
            logger.error("Cannot retrieve violations: database not initialized");
            return violations;
        }
        
        // Prepare SQL statement
        sqlite3_stmt* stmt;
        const char* sql = 
            "SELECT json_data FROM violations "
            "ORDER BY timestamp DESC LIMIT ? OFFSET ?;";
        
        int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            logger.error("Failed to prepare SQL statement: %s", sqlite3_errmsg(db));
            return violations;
        }
        
        // Bind parameters
        sqlite3_bind_int(stmt, 1, limit);
        sqlite3_bind_int(stmt, 2, offset);
        
        // Execute statement and process results
        while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
            const char* jsonData = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            
            if (jsonData) {
                try {
                    ViolationRecord violation = ViolationRecord::fromJson(std::string(jsonData));
                    violations.push_back(violation);
                }
                catch (const std::exception& e) {
                    logger.error("Error parsing violation data: %s", e.what());
                }
            }
        }
        
        // Clean up
        sqlite3_finalize(stmt);
        
        return violations;
    }
};

// Application entry point
int main(int argc, char** argv) {
    // Process command line arguments
    std::string configFile = DEFAULT_CONFIG_FILE;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--config" && i + 1 < argc) {
            configFile = argv[++i];
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "SpeedGuardian Pro - Advanced Traffic Monitoring and Speed Control System" << std::endl;
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --config <file>   Specify configuration file (default: " << DEFAULT_CONFIG_FILE << ")" << std::endl;
            std::cout << "  --help, -h        Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Create and initialize speed camera
    SpeedCamera speedCamera(configFile);
    
    if (!speedCamera.initialize()) {
        std::cerr << "Failed to initialize speed camera system" << std::endl;
        return 1;
    }
    
    if (!speedCamera.start()) {
        std::cerr << "Failed to start speed camera system" << std::endl;
        return 1;
    }
    
    // Run until user presses ESC
    while (speedCamera.isSystemRunning()) {
        // Get the most recent processed frame
        cv::Mat frame = speedCamera.getLatestProcessedFrame();
        
        if (!frame.empty()) {
            cv::imshow("SpeedGuardian Pro", frame);
        }
        
        // Check for user input
        int key = cv::waitKey(1);
        if (key == 27) {  // ESC key
            speedCamera.stop();
            break;
        }
    }
    
    cv::destroyAllWindows();
    return 0;
}