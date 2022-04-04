//
// Created by hoshino on 2022/4/3.
//

#include <iostream>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "common.h"
#include "result_verify.h"

using namespace std;
using json = nlohmann::json;

Eigen::Matrix3d inner;
string lidar_path, photo_path, intrinsic_path, extrinsic_path;
int error_threshold = 12;
vector<float> init{0.0, -1.0, 0.0, 0.0,
0.0, 0.0, -1.0, 0.0,
1.0, 0.0, 0.0,  0.0};

void getParameters();

void getParameters(){
    // read a JSON file
    std::ifstream i("../resources/config.json");
    if (i.is_open()){
        json config;
        i >> config;
        lidar_path = config["lidar_path"].get<string>();
        photo_path = config["photo_path"].get<string>();
        intrinsic_path = config["intrinsic_path"].get<string>();
        extrinsic_path = config["extrinsic_path"].get<string>();
        error_threshold = config["error_threshold"].get<int>();
    }
    else{
        json config;
        // write prettified JSON to another file
        lidar_path = "../resources/corner_lidar.txt";
        photo_path = "../resources/corner_photo.txt";
        intrinsic_path = "../resources/intrinsic.txt";
        extrinsic_path = "../resources/extrinsic.txt";
        config.emplace("lidar_path", "../resources/corner_lidar.txt");
        config.emplace("photo_path", "../resources/corner_photo.txt");
        config.emplace("intrinsic_path", "../resources/intrinsic.txt");
        config.emplace("extrinsic_path", "../resources/extrinsic.txt");
        config.emplace("error_threshold", 12);
        std::ofstream o("../resources/config.json");
        o << std::setw(4) << config << std::endl;
    }
}


class external_cali {
public:
    external_cali(PnPData p) {
        pd = p;
    }

    template <typename T>
    bool operator()(const T *_q, const T *_t, T *residuals) const {
        Eigen::Matrix<T, 3, 3> innerT = inner.cast<T>();
        Eigen::Quaternion<T> q_incre{_q[ 3 ], _q[ 0 ], _q[ 1 ], _q[ 2 ]};
        Eigen::Matrix<T, 3, 1> t_incre{_t[ 0 ], _t[ 1 ], _t[ 2 ]};

        Eigen::Matrix<T, 3, 1> p_l(T(pd.x), T(pd.y), T(pd.z));
        Eigen::Matrix<T, 3, 1> p_c = q_incre.toRotationMatrix() * p_l + t_incre;
        Eigen::Matrix<T, 3, 1> p_2 = innerT * p_c;

        residuals[0] = p_2[0]/p_2[2] - T(pd.u);
        residuals[1] = p_2[1]/p_2[2] - T(pd.v);

        return true;
    }

    static ceres::CostFunction *Create(PnPData p) {
        return (new ceres::AutoDiffCostFunction<external_cali, 2, 4, 3>(new external_cali(p)));
    }


private:
    PnPData pd;

};

int main(int argc, char **argv) {
    getParameters();

    vector<PnPData> pData;
    getData(lidar_path, photo_path, pData);

    Eigen::Matrix4d extrin;
    // set the intrinsic parameters
    vector<float> intrinsic;
    getIntrinsic(intrinsic_path, intrinsic);
    inner << intrinsic[0], intrinsic[1], intrinsic[2],
            intrinsic[3], intrinsic[4], intrinsic[5],
            intrinsic[6], intrinsic[7], intrinsic[8];

    // init the matrix of extrinsic, matrix of rotation and translation
    // keep this init value, or it is possible to get a local optimal result
    Eigen::Matrix3d R;
    R << init[0], init[1], init[2],
            init[4], init[5], init[6],
            init[8], init[9], init[10];
    Eigen::Quaterniond q(R);
    double ext[7];

    ext[0] = q.x();
    ext[1] = q.y();
    ext[2] = q.z();
    ext[3] = q.w();
    ext[4] = init[3];
    ext[5] = init[7];
    ext[6] = init[11];

    Eigen::Map<Eigen::Quaterniond> m_q = Eigen::Map<Eigen::Quaterniond>(ext);
    Eigen::Map<Eigen::Vector3d> m_t = Eigen::Map<Eigen::Vector3d>(ext + 4);

    ceres::LocalParameterization * q_parameterization = new ceres::EigenQuaternionParameterization();
    ceres::Problem problem;

    problem.AddParameterBlock(ext, 4, q_parameterization);
    problem.AddParameterBlock(ext + 4, 3);

    for(auto val : pData) {
        ceres::CostFunction *cost_function;
        cost_function = external_cali::Create(val);
        problem.AddResidualBlock(cost_function, NULL, ext, ext + 4);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl;

    Eigen::Matrix3d rot = m_q.toRotationMatrix();
    writeExt(extrinsic_path, rot, m_t);
    cout << rot << endl;
    cout << m_t << endl;

    cout << "Use the extrinsic result to reproject the data" << endl;
    float error[2] = {0, 0};
    getUVError(intrinsic_path, extrinsic_path, lidar_path, photo_path, error, error_threshold);

    cout << "u average error is: " << error[0] << endl;
    cout << "v average error is: " << error[1] << endl;
    if (error[0] + error[1] < error_threshold) {
        cout << endl << "The reprojection error smaller than the threshold, extrinsic result seems ok" << endl;
    }
    else {
        cout << endl << "The reprojection error bigger than the threshold, extrinsic result seems not ok" << endl;
    }
    return 0;
}
