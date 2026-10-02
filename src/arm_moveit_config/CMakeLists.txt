cmake_minimum_required(VERSION 3.8)
project(arm_moveit_config)

find_package(ament_cmake REQUIRED)

# 纯资源包：config 里放 SRDF / IK / 限位 / 控制器映射，
# launch 里放 move_group 与 RViz 的启动文件。
install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

# 注意：.setup_assistant 是【隐藏文件】，必须单独 install：
#    moveit_configs_utils 靠它推断 URDF 在哪个包、哪个路径。
#    漏装它的表现是启动时一串
#      WARNING Cannot infer URDF from ... -- using config/o5_2_arm.urdf
#    然后 move_group 根本没有 robot_description，规划全废。
install(
  FILES .setup_assistant
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
