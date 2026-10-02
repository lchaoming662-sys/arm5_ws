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

# MTC 抓取流水线。由 grasp_mtc.launch.py 用 Node(executable=...) 启动，
# 所以必须装进 lib/ 而不是留在 tools/：它得靠 launch 注入整套 MoveIt 参数
# （kinematics / ompl 配置），脚本自己配不出来（见该文件头说明）。
install(
  PROGRAMS scripts/grasp_pipeline_mtc.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
