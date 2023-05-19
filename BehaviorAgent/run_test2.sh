#!/bin/bash
export ROUTES=/workspace/team_code/AVD_Project/BehaviorAgent/routes_avddiem_exam2.xml
export REPETITIONS=1
export DEBUG_CHALLENGE=1
export TEAM_AGENT=/workspace/team_code/AVD_Project/BehaviorAgent/carla_behavior_agent/basic_autonomous_agent.py
export TEAM_CONFIG=/workspace/team_code/AVD_Project/BehaviorAgent/carla_behavior_agent/config_agent_basic.json
export CHALLENGE_TRACK_CODENAME=SENSORS
export CARLA_HOST=0.0.0.0
export CARLA_PORT=2000
export CARLA_TRAFFIC_MANAGER_PORT=8000
export CHECKPOINT_ENDPOINT=/workspace/team_code/AVD_Project/BehaviorAgent/results2/simulation_results.json
export DEBUG_CHECKPOINT_ENDPOINT=/workspace/team_code/AVD_Project/BehaviorAgent/results2/live_results.txt
export RESUME=0
export TIMEOUT=180

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--routes-subset=${ROUTES_SUBSET} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--debug-checkpoint=${DEBUG_CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--host=${CARLA_HOST} \
--port=${CARLA_PORT} \
--timeout=${TIMEOUT} \
--traffic-manager-port=${CARLA_TRAFFIC_MANAGER_PORT} 
