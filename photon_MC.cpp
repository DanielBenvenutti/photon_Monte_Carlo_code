#include <cstddef>
#include <iostream>
#include <eigen3/Eigen/Dense>

struct Photon {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d direction = Eigen::Vector3d::Zero();
    std::size_t zoneIndex = 0;
};

int main() {
    Photon photon;

    photon.direction = Eigen::Vector3d(1.0, 2.0, 3.0);
    photon.direction = Eigen::Vector3d::UnitZ();

    std::cout << "position = " << photon.position.transpose() << "\n";
    std::cout << "direction = " << photon.direction.transpose() << "\n";
    std::cout << "norm(direction) = " << photon.direction.norm() << "\n";
}
