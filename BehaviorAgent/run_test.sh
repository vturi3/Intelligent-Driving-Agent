#!/bin/bash
export ROUTES=/workspace/team_code/routes_avddiem_exam.xml
export REPETITIONS=1
export DEBUG_CHALLENGE=1
export TEAM_AGENT=/workspace/team_code/carla_behavior_agent/basic_autonomous_agent.py
export TEAM_CONFIG=/workspace/team_code/carla_behavior_agent/config_agent_basic.json
export CHALLENGE_TRACK_CODENAME=SENSORS
export CARLA_HOST=YOUR_IP
export CARLA_PORT=YOUR_PORT
export CARLA_TRAFFIC_MANAGER_PORT=YOUR_TM_PORT
export CHECKPOINT_ENDPOINT=/workspace/team_code/results/simulation_results.json
export DEBUG_CHECKPOINT_ENDPOINT=/workspace/team_code/results/live_results.txt
export RESUME=0
export TIMEOUT=60

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
