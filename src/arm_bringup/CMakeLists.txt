cmake_minimum_required(VERSION 3.8)
project(arm_bringup)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

# 门闩节点：bringup.launch.py 用它等控制器 active（见 scripts/ 里的文件头说明）
install(
  PROGRAMS scripts/wait_for_controllers.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
