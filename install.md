#hangul install

sudo apt upgrade ibus-hangul -y


# Chromium browser install

## snap

sudo snap install chromium

# Firefox install

## snap

sudo snap install firefox

# ROS2 Humble install

## Set locale(UTF-8)

locale  # check for UTF-8

sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

locale  # verify settings


## Universe repository is enabled

sudo apt install software-properties-common
sudo add-apt-repository universe

##  ROS 2 GPG key with apt

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

## package update & system upgrade

sudo apt update
sudo apt upgrade

## ROS2 Humble install

sudo apt install ros-humble-desktop


## ROS-Base install

sudo apt install ros-humble-ros-base

## Development tools install

sudo apt install ros-dev-tools
