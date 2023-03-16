export CARLA_HOST=193.205.163.183
export CARLA_PORT=6021
export CARLA_TRAFFIC_MANAGER_PORT=6023

python3 /workspace/team_code/test/manual_driving.py --host=${CARLA_HOST} --port=${CARLA_PORT}
