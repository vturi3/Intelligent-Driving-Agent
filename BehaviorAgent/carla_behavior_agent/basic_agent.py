# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights.
It can also make use of the global route planner to follow a specifed route
"""

import carla
from shapely.geometry import Polygon
from shapely.validation import explain_validity

import math

from local_planner import LocalPlanner, RoadOption
from global_route_planner import GlobalRoutePlanner
from datetime import datetime,timedelta
from misc import (get_speed, is_within_distance,our_is_within_distance,
                               get_trafficlight_trigger_location,
                               compute_distance,draw_bbox,draw_waypoints)
import numpy as np
from math import inf
# from perception.perfectTracker.gt_tracker import PerfectTracker

class BasicAgent(object):
    """
    BasicAgent implements an agent that navigates the scene.
    This agent respects traffic lights and other vehicles, but ignores stop signs.
    It has several functions available to specify the route that the agent must follow,
    as well as to change its parameters in case a different driving mode is desired.
    """

    def __init__(self, vehicle, opt_dict={}, map_inst=None, grp_inst=None):
        """
        Initialization the agent paramters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
            :param map_inst: carla.Map instance to avoid the expensive call of getting it.
            :param grp_inst: GlobalRoutePlanner instance to avoid the expensive call of getting it.

        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        if map_inst:
            if isinstance(map_inst, carla.Map):
                self._map = map_inst
            else:
                # print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()
        self._last_traffic_light = None
        self._last_time_stop_sign = None
        self._last_stop_signid = None

        self._surpassing_police = False
        self._surpassing_obj = False
        self._surpassing_obstacle = False
        self._surpassing_car = False

        # Base parameters
        self._ignore_traffic_lights = False
        self._ignore_stop_signs = False
        self._ignore_vehicles = False
        self._use_bbs_detection = False
        self._target_speed = 5.0
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 5.0  # meters
        self._base_stop_threshold = 4.0  # meters
        self._base_vehicle_threshold = 5.0  # meters
        self._speed_ratio = 1
        self._max_brake = 0.5
        self._offset = 0

        # Change parameters according to the dictionary
        if 'target_speed' in opt_dict:
            self._target_speed = opt_dict['target_speed']
        if 'ignore_traffic_lights' in opt_dict:
            self._ignore_traffic_lights = opt_dict['ignore_traffic_lights']
        if 'ignore_stop_signs' in opt_dict:
            self._ignore_stop_signs = opt_dict['ignore_stop_signs']
        if 'ignore_vehicles' in opt_dict:
            self._ignore_vehicles = opt_dict['ignore_vehicles']
        if 'use_bbs_detection' in opt_dict:
            self._use_bbs_detection = opt_dict['use_bbs_detection']
        if 'sampling_resolution' in opt_dict:
            self._sampling_resolution = opt_dict['sampling_resolution']
        if 'base_tlight_threshold' in opt_dict:
            self._base_tlight_threshold = opt_dict['base_tlight_threshold']
        if 'base_vehicle_threshold' in opt_dict:
            self._base_vehicle_threshold = opt_dict['base_vehicle_threshold']
        if 'detection_speed_ratio' in opt_dict:
            self._speed_ratio = opt_dict['detection_speed_ratio']
        if 'max_brake' in opt_dict:
            self._max_brake = opt_dict['max_brake']
        if 'offset' in opt_dict:
            self._offset = opt_dict['offset']
        
        # Initialize the planners
        self._local_planner = LocalPlanner(self._vehicle, opt_dict=opt_dict, map_inst=self._map)
        if grp_inst:
            if isinstance(grp_inst, GlobalRoutePlanner):
                self._global_planner = grp_inst
            else:
                # print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)
        else:
            self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)

        # Get the static elements of the scene
        self._lights_list = self._world.get_actors().filter("*traffic_light*")
        self._lights_map = {}  # Dictionary mapping a traffic light to a wp corrspoing to its trigger volume location
        self._stop_map = {}  # Dictionary mapping a stop sign to a wp corrspoing to its trigger volume location

    def add_emergency_stop(self, control):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def set_target_speed(self, speed):
        """
        Changes the target speed of the agent
            :param speed (float): target speed in Km/h
        """
        self._target_speed = speed
        self._local_planner.set_speed(speed)

    def follow_speed_limits(self, value=True):
        """
        If active, the agent will dynamically change the target speed according to the speed limits

            :param value (bool): whether or not to activate this behavior
        """
        self._local_planner.follow_speed_limits(value)

    def get_local_planner(self):
        """Get method for protected member local planner"""
        return self._local_planner

    def get_global_planner(self):
        """Get method for protected member local planner"""
        return self._global_planner

    def set_destination(self, end_location, start_location=None):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)

    def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a specific plan to the agent.

            :param plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
            :param stop_waypoint_creation: stops the automatic random creation of waypoints
            :param clean_queue: resets the current agent's plan
        """
        self._local_planner.set_global_plan(
            plan,
            stop_waypoint_creation=stop_waypoint_creation,
            clean_queue=clean_queue
        )

    def trace_route(self, start_waypoint, end_waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def run_step(self):
        """Execute one step of navigation."""
        hazard_detected = False

        #####
        #  Retrieve all relevant actors
        #####
        # Basic Agent :
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        ### 

        vehicle_speed = get_speed(self._vehicle) / 3.6

        # Check for possible vehicle obstacles
        max_vehicle_distance = self._base_vehicle_threshold + self._speed_ratio * vehicle_speed
        affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            hazard_detected = True

        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + self._speed_ratio * vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(self._lights_list, max_tlight_distance)
        if affected_by_tlight:
            hazard_detected = True

        control = self._local_planner.run_step()
        if hazard_detected:
            control = self.add_emergency_stop(control)

        return control
    
    def reset(self):
        pass

    def done(self):
        """Check whether the agent has reached its destination."""
        return self._local_planner.done()

    def ignore_traffic_lights(self, active=True):
        """(De)activates the checks for traffic lights"""
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_vehicles = active

    def lane_change(self, direction, same_lane_time=0, other_lane_time=0, lane_change_time=2):
        """
        Changes the path so that the vehicle performs a lane change.
        Use 'direction' to specify either a 'left' or 'right' lane change,
        and the other 3 fine tune the maneuver
        """
        # print("lane change" , direction)
        speed = self._vehicle.get_velocity().length()
        path = self._generate_lane_change_path(
            self._map.get_waypoint(self._vehicle.get_location()),
            direction,
            same_lane_time * speed,
            other_lane_time * speed,
            lane_change_time * speed,
            False,
            1,
            self._sampling_resolution
        )
        if not path:
            print("WARNING: Ignoring the lane change as no path was found")
        self._lane_changed = direction

        self.set_global_plan(path,clean_queue=False)

    def _affected_by_traffic_light(self, lights_list=None, max_distance=None):
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
        """
        if self._ignore_traffic_lights:
            return (False, None)

        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = self._base_tlight_threshold
        # qua verifico se mi sono gia fermato allo step precedete.
        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            if traffic_light.id in self._lights_map:
                trigger_wp = self._lights_map[traffic_light.id]
            else:
                trigger_location = get_trafficlight_trigger_location(traffic_light)
                trigger_wp = self._map.get_waypoint(trigger_location)
                self._lights_map[traffic_light.id] = trigger_wp

            if trigger_wp.transform.location.distance(ego_vehicle_location) > max_distance:
                continue
            # Escludo semafori di altre strade.
            if trigger_wp.road_id != ego_vehicle_waypoint.road_id:
                continue

            # orientamento semaforo, cosi gestisco se si trova diffronte a me o no,
            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = trigger_wp.transform.get_forward_vector() 
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            if is_within_distance(trigger_wp.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
                self._last_traffic_light = traffic_light
                return (True, traffic_light)

        return (False, None)
    
    def _affected_by_stop_sign(self, stop_list=None, max_distance=None):
        """
        
        """
        if self._ignore_traffic_lights:
            return (False, None,0.0)

        if not stop_list:
            stop_list = self._world.get_actors().filter("*stop*")

        if not max_distance:
            max_distance = self._base_stop_threshold*2
            print("self._base_stop_threshold: ", self._base_stop_threshold*2)
        print("max_distance: ", max_distance)
        # qua verifico se mi sono gia fermato allo step precedete.
        if self._last_time_stop_sign:
            # qua devo verificare se posso andare avanti ritornando quindi un vuoto o eventualmente un altro !
            # print('sono fermo da ',self._last_time_stop_sign ,'e sono', self._world.get_snapshot().timestamp.elapsed_seconds)

            if self._world.get_snapshot().timestamp.elapsed_seconds - self._last_time_stop_sign >= 3:
                self._last_time_stop_sign = None
            else: #se non sono passati 3 secondi
                return (True, self._last_time_stop_sign,0.0)

        #se arrivo qua significa che non ne avevo uno uno in precedenza, quindi ne devo cercare un altro massiccio

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for stop_sign in stop_list:
            if stop_sign.id in self._stop_map:
                trigger_wp = self._stop_map[stop_sign.id]
            else:
                trigger_location = get_trafficlight_trigger_location(stop_sign)
                trigger_wp = self._map.get_waypoint(trigger_location)
                self._stop_map[stop_sign.id] = trigger_wp

            dist_from_stop = trigger_wp.transform.location.distance(ego_vehicle_location)

            if dist_from_stop > max_distance:
                continue
            # Escludo stop di altre strade.
            if trigger_wp.road_id != ego_vehicle_waypoint.road_id:
                continue

            # print(dist_from_stop)

            # orientamento stop, cosi gestisco se si trova diffronte a me o no.
            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = trigger_wp.transform.get_forward_vector() 
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z
            print("devo ceccare se ci sta uno stop (ci sta ma non se va bene): ")

            if dot_ve_wp < 0:
                continue

            if is_within_distance(trigger_wp.transform, self._vehicle.get_transform(), max_distance, [-30, 30]) and self._last_stop_signid != stop_sign.id:
                if dist_from_stop < 2.5:
                    self._last_time_stop_sign = self._world.get_snapshot().timestamp.elapsed_seconds
                    self._last_stop_signid = stop_sign.id
                print("ho visto uno stop si trova a: ", stop_sign, "dista: ", dist_from_stop)
                # input()
                return (True, stop_sign,dist_from_stop)
            else:
                print("ho visto che non sta sulla mia strada")

        return (False, None,0.0)
    
    def _vehicle_obstacle_detected_old(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        def get_route_polygon():
            route_bb = []
            extent_y = self._vehicle.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            for wp, _ in self._local_planner.get_plan():
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
                p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self._map.get_waypoint(ego_location)

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            self._vehicle.bounding_box.extent.x * ego_transform.get_forward_vector())

        opposite_invasion = abs(self._offset) + self._vehicle.bounding_box.extent.y > ego_wpt.lane_width / 2
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            target_transform = target_vehicle.get_transform()
            if target_transform.location.distance(ego_location) > max_distance:
                continue

            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (use_bbs or target_wpt.is_junction) and route_polygon:

                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon):
                    return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location))

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:

                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id  + lane_offset:
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    return (True, target_vehicle, compute_distance(target_transform.location, ego_transform.location))

        return (False, None, -1)
    
    def _vehicle_obstacle_detected(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        # funzione x valutare se un ostacolo si trova in una posizione bloccate x il nostro agente (bloccante nel senso che sta dove dobbiamo andare noi). Gli passiamo la lista (come detto prima), max distanza sotto la quale lo consideriamo bloccante.

        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            return (False, None, -1)

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego, Vedo in quale lane si trova e sta cercando di capire se esiste un offset della lane
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )
        # l'idea è verficiare dove voglio andare e dove si trova il vehicle, se si trova sulla nostra corsia e strada. Quello che succede è valutare la direzione e la posizione del vehicle.

        for target_vehicle in vehicle_list:
            # per ogni vehicle della lista
            target_transform = target_vehicle.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)
            # dove si trova e dove voglio andare, Obj waypoint molto smart ha tanta roba. 
            # Simplified version for outside junctions, verifica se non siamo nell'incrocio, se nn sto nella stessa strada e nn sto nella stessa lane dell'obj, prendi prossimo waypoint
            if not ego_wpt.is_junction or not target_wpt.is_junction:
                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                    # prende dalla coda dei waypoint dati al local planner si prende solo il waipoint. la direction esprimre l'intenzione, steps 3 perchè valuta quello in po piu avanti.
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id  + lane_offset:
                        continue

                # la rear trasform punto posteriore del vehicolo, se si trova ad una certa distanza dal nostro punto anteriore,
                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )
                
                # low angle th e up, sono angoli che vengono presi in considearzione. Capisce se ho possibile collisione. Gli angoli mi servono in base alle intenzioni, magari stiamo andando da due parti diverse quindi nn ci scontreremo e nn lo cago.
                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    return (True, target_vehicle, compute_distance(target_transform.location, ego_transform.location))

            # Waypoints aren't reliable, check the proximity of the vehicle to the route, in questo caso sono in un incrocio. Segue una logica che dipende dalla traiettoria, viene costrituito in poligono sulla nostra traiettoria. Viene valutato dove ci trovamo, quanto siamo grandi (laterali), per ogni punto del plan valuto se la distanza nostra dal punto del plan è troppo lontana non mi interessa ( prendo solo punti plan vicini ). In quel punto del plan ci passo, voglio sapere li il mio "poligono". Alla fine costruisco nei waypoint i poligono su tutti i punti del plan, cosi ho un poligono della nstra traiettoria (l'abbiamo visto a lezione).
            else:
                route_bb = []
                ego_location = ego_transform.location
                extent_y = self._vehicle.bounding_box.extent.y
                r_vec = ego_transform.get_right_vector()
                p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                route_bb.append([p1.x, p1.y, p1.z])
                route_bb.append([p2.x, p2.y, p2.z])

                for wp, _ in self._local_planner.get_plan():
                    if ego_location.distance(wp.transform.location) > max_distance:
                        break

                    r_vec = wp.transform.get_right_vector()
                    p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                    p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                    route_bb.append([p1.x, p1.y, p1.z])
                    route_bb.append([p2.x, p2.y, p2.z])

                if len(route_bb) < 3:
                    # 2 points don't create a polygon, nothing to check
                    return (False, None, -1)
                ego_polygon = Polygon(route_bb)

                # Compare the two polygons, per tutti gli obj passati in ingresso, se mi trovo in intersection faccio questa valuazione. Se sono io quello che analizzo o è troppo distanet nn lo cago. Per gli altri prendo boundingbox veicolo, prendo i vertici nel mondo e verifico se collidono con il mio.
                # Qua gia si potrebbe fare la modifica suggerita dal prof in classe dei cerchi. Inoltre viene valutato solo la posizione attuale del vehicle. (prendendo info su direzione e velocita)
                for target_vehicle in vehicle_list:
                    target_extent = target_vehicle.bounding_box.extent.x
                    if target_vehicle.id == self._vehicle.id:
                        continue
                    if ego_location.distance(target_vehicle.get_location()) > max_distance:
                        continue

                    target_bb = target_vehicle.bounding_box
                    target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                    target_list = [[v.x, v.y, v.z] for v in target_vertices]
                    target_polygon = Polygon(target_list)

                    if ego_polygon.intersects(target_polygon):
                        return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location))

                return (False, None, -1)

        return (False, None, -1)

    def allunga_bounding_box(self,vehicle, alpha=0.09):
        """
        Allunga il bounding box del veicolo lungo l'asse X utilizzando il forward vector del veicolo.

        Args:
            vehicle: oggetto "Actor" rappresentante il veicolo
            alpha: fattore di allungamento del bounding box lungo l'asse X

        Returns:
            BoundingBox: il bbox aggiornato
        """

        # ottieni il bounding box del veicolo
        bbox = vehicle.bounding_box
        bbox_extent = bbox.extent

        # ottieni la velocità del veicolo
        velocity = get_speed(vehicle)

        # ottieni la velocità del veicolo
        factor = alpha * velocity
                
        if factor < 1:
            
            factor = 1

        if vehicle.type_id.split('.')[0] == 'walker':
            factor = 5
        # calcola la quantità di spostamento lungo l'asse X
        x_offset = bbox_extent.x * factor

        # x_offset = bbox_extent.x * factor
        # print("BBOX: velocity  ", velocity)
        # print("BBOX: fattore moltiplicativo ", factor)
        # print("BBOX: offset ", x_offset)
        
        # input()

        # calcola le nuove dimensioni del bounding box
        new_bbox_extent = carla.Vector3D(x=x_offset, y=bbox_extent.y, z=2)

        # print("BBOX: new bbox extent" , new_bbox_extent)

        vehicle_transform = vehicle.get_transform()
        vehicle_ffVector_x = vehicle_transform.get_forward_vector().x
        vehicle_ffVector_y = vehicle_transform.get_forward_vector().y
        spostamento_x = 0
        spostamento_y = 0
        if factor >1:
            spostamento_x = vehicle_ffVector_x * (x_offset)
            spostamento_y = vehicle_ffVector_y * (x_offset)

        new_location = carla.Location(x=vehicle_transform.location.x + spostamento_x, y=vehicle_transform.location.y + spostamento_y, z=vehicle_transform.location.z) 

        new_transform = carla.Transform(new_location, vehicle_transform.rotation) 

        draw_bbox(self._world, vehicle,new_bbox_extent,color=carla.Color(0,255,0,0),location=new_location)

        return carla.BoundingBox(bbox.location, new_bbox_extent),new_transform


    def _our_vehicle_obstacle_detected(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        # funzione x valutare se un ostacolo si trova in una posizione bloccate x il nostro agente (bloccante nel senso che sta dove dobbiamo andare noi). Gli passiamo la lista (come detto prima), max distanza sotto la quale lo consideriamo bloccante.

        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            return (False, None, -1)

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego, Vedo in quale lane si trova e sta cercando di capire se esiste un offset della lane
        
        ego_rear_extent = np.sqrt(np.square(self._vehicle.bounding_box.extent.y/2) + np.square(self._vehicle.bounding_box.extent.x/2))
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )
        # l'idea è verficiare dove voglio andare e dove si trova il vehicle, se si trova sulla nostra corsia e strada. Quello che succede è valutare la direzione e la posizione del vehicle.

        for target_vehicle in vehicle_list:
            # per ogni vehicle della lista
            print("obj: ",target_vehicle)
            draw_bbox(self._world,target_vehicle)
            target_transform = target_vehicle.get_transform()
            target_forward_vector = target_transform.get_forward_vector()
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)
            # dove si trova e dove voglio andare, Obj waypoint molto smart ha tanta roba. 
            # Simplified version for outside junctions, verifica se non siamo nell'incrocio, se nn sto nella stessa strada e nn sto nella stessa lane dell'obj, prendi prossimo waypoint
            if not ego_wpt.is_junction or not target_wpt.is_junction:
                # print("ego_wpt.lane_id: ",ego_wpt.lane_id, "la lane id del target è: ",target_wpt.lane_id, "e la mia direction è", self._direction)
                next_lane = ego_wpt.lane_id
                try:
                    if self._direction == RoadOption.CHANGELANELEFT:
                        next_lane = ego_wpt.get_left_lane().lane_id
                    elif self._direction == RoadOption.CHANGELANERIGHT:
                        next_lane = ego_wpt.get_right_lane().lane_id
                except:
                    pass
                
                if target_wpt.lane_id * ego_wpt.lane_id < 0 and self._direction != RoadOption.LANEFOLLOW and target_wpt.lane_id == next_lane:
                    # print("ci sta un ceicolo nell'altra corsia: ",target_vehicle)
                    cond = 2*lane_offset*ego_wpt.lane_id
                else:
                    cond =  lane_offset
                

                # Recupero le coordinate dell'angolo del bounding box del veicolo più vicino al bordo della corsia
                # tv_bb_coords = target_vehicle.bounding_box.get_world_vertices(target_vehicle.get_transform())
                tv_bb_coords, e_x, e_y, e_z = self.get_bounding_box_corners(target_vehicle)
                tv_vertexs_lane_id = [(self._map.get_waypoint(carla.Location(bb_coord[0], bb_coord[1], bb_coord[2]), lane_type=carla.LaneType.Any)).lane_id for bb_coord in tv_bb_coords]  # l'angolo in alto a destra
                # if target_wpt.lane_id not in tv_vertexs_lane_id:
                #     input()
                #     tv_vertexs_lane_id = [target_wpt.lane_id for tv_vertex_lane_id in tv_vertexs_lane_id]
                
                # Recupero le coordinate dell'angolo del bounding box del veicolo più vicino al bordo della corsia
                ego_bb_coords = self._vehicle.bounding_box.get_world_vertices(self._vehicle.get_transform())
                ego_vertexs_lane_id = [(self._map.get_waypoint(bb_coord)).lane_id + cond for bb_coord in ego_bb_coords]

                on_same_lane = list(set(tv_vertexs_lane_id) & set(ego_vertexs_lane_id)) 
                # #print("ego_vertexs_lane_id:", ego_vertexs_lane_id)
                # #print("tv_vertexs_lane_id:", tv_vertexs_lane_id)
                # #print("len(on_same_lane): ",len(on_same_lane))
                print("tv_vertexs_lane_id: ", tv_vertexs_lane_id, "ego_vertexs_lane_id: ", ego_vertexs_lane_id)
                # input()
                if target_wpt.road_id != ego_wpt.road_id or len(on_same_lane) == 0:
                    print("potrei non considerarlo l'obj")
                    # input()
                    # #print("dopo if ego_wpt.lane_id: ",ego_wpt.lane_id, "la lane id del target è: ",target_wpt.lane_id, "e la mia direction è", self._direction)

                    # prende dalla coda dei waypoint dati al local planner si prende solo il waipoint. la direction esprimre l'intenzione, steps 3 perchè valuta quello in po piu avanti.
                    if self._direction == RoadOption.RIGHT or self._direction == RoadOption.LEFT:
                        next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=20)[0]
                    else:
                        next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if target_wpt.lane_id * next_wpt.lane_id < 0 and self._direction != RoadOption.LANEFOLLOW and target_wpt.lane_id == next_lane:
                        # print("ci sta un ceicolo nell'altra corsia prossimamnte: ",target_vehicle)
                        cond = - 2*lane_offset*next_wpt.lane_id
                    else:
                        cond =  lane_offset
                    if next_wpt.lane_id in ego_vertexs_lane_id:
                        # print("len(on_same_lane) == 0: ", len(on_same_lane))
                        if target_wpt.road_id != next_wpt.road_id or len(on_same_lane) == 0:
                            continue
                    else:
                        # print("next_wpt.lane_id  + cond: ", next_wpt.lane_id  + cond, "tv_vertexs_lane_id: ", tv_vertexs_lane_id)
                        if target_wpt.road_id != next_wpt.road_id or next_wpt.lane_id  + cond not in tv_vertexs_lane_id:
                            continue

                # la rear trasform punto posteriore del vehicolo, se si trova ad una certa distanza dal nostro punto anteriore,

                target_rear_transform = target_transform
                target_rear_extent = np.sqrt(np.square(target_vehicle.bounding_box.extent.y/2) + np.square(target_vehicle.bounding_box.extent.x/2))
                # low angle th e up, sono angoli che vengono presi in considearzione. Capisce se ho possibile collisione. Gli angoli mi servono in base alle intenzioni, magari stiamo andando da due parti diverse quindi nn ci scontreremo e nn lo cago.
                
                if self._surpassing_obj and self._direction != RoadOption.LANEFOLLOW:
                    max_distance = inf
                is_within,dist = our_is_within_distance(target_rear_transform, ego_front_transform,target_rear_extent,ego_rear_extent, max_distance, [low_angle_th, up_angle_th])
                if dist < 0:
                    dist = 0.5
                # print(dist)
                condToRet = (np.dot(np.array([target_forward_vector.x,target_forward_vector.y, target_forward_vector.z]), np.array([ego_forward_vector.x,ego_forward_vector.y,ego_forward_vector.z])) > 0 or self._direction == RoadOption.CHANGELANELEFT or self._direction == RoadOption.CHANGELANERIGHT or ego_wpt.lane_id in tv_vertexs_lane_id)
                if is_within and condToRet:
                    print("un obj possibile collisione: ", target_vehicle, "dist: ", dist)
                    # input()
                    return (True, target_vehicle, dist)

        return (False, None, -1)
    def gestsione_incroci(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        # funzione x valutare se un ostacolo si trova in una posizione bloccate x il nostro agente (bloccante nel senso che sta dove dobbiamo andare noi). Gli passiamo la lista (come detto prima), max distanza sotto la quale lo consideriamo bloccante.

        if self._ignore_vehicles:
            return (False, None, -1)

        if not vehicle_list:
            return (False, None, -1)

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego, Vedo in quale lane si trova e sta cercando di capire se esiste un offset della lane
        
        ego_rear_extent = np.sqrt(np.square(self._vehicle.bounding_box.extent.y/2) + np.square(self._vehicle.bounding_box.extent.x/2))
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )
        # l'idea è verficiare dove voglio andare e dove si trova il vehicle, se si trova sulla nostra corsia e strada. Quello che succede è valutare la direzione e la posizione del vehicle.

    
        route_bb = []
        ego_location = ego_transform.location
        extent_y = self._vehicle.bounding_box.extent.y
        r_vec = ego_transform.get_right_vector()
        p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
        p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
        route_bb.append([p1.x, p1.y, p1.z])
        route_bb.append([p2.x, p2.y, p2.z])
        i=0
        for wp, _ in self._local_planner.get_plan():
            if ego_location.distance(wp.transform.location) > max_distance or i > 3:
                break
            draw_waypoints(self._world,[wp])
            i+=1

            r_vec = wp.transform.get_right_vector()
            p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
            p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
            route_bb.append([p1.x, p1.y, p1.z])
            route_bb.append([p2.x, p2.y, p2.z])

        if len(route_bb) < 3:
            # 2 points don't create a polygon, nothing to check
            return (False, None, -1)
        ego_polygon = Polygon(route_bb)

        # Compare the two polygons, per tutti gli obj passati in ingresso, se mi trovo in intersection faccio questa valuazione. 
        # Se sono io quello che analizzo o è troppo distanet nn lo cago. 
        # Per gli altri prendo boundingbox veicolo, prendo i vertici nel mondo e verifico se collidono con il mio.
        # Qua gia si potrebbe fare la modifica suggerita dal prof in classe dei cerchi. 
        # Inoltre viene valutato solo la posizione attuale del vehicle. (prendendo info su direzione e velocita)

        hero_vertices = self._vehicle.bounding_box.get_world_vertices(ego_transform)
        hero_list = [[v.x, v.y, v.z] for v in hero_vertices]
        hero_polygon = Polygon(hero_list)
        ffv_hero_vehicle = ego_transform.get_forward_vector()

        for target_vehicle in vehicle_list:
            if ego_location.distance(target_vehicle.get_location()) > max_distance*4:
                continue
        
            target_bb,new_transform = self.allunga_bounding_box(target_vehicle)
            
            target_vertices = target_bb.get_world_vertices(new_transform)
            target_list = [[v.x, v.y, v.z] for v in target_vertices]
            target_polygon = Polygon(target_list)

            if ego_polygon.intersects(target_polygon):
                print('INTERSECTION: Colpisco boundingBox')
                if hero_polygon.intersects(target_polygon):
                    ffv_target_vehicle = target_vehicle.get_transform().get_forward_vector()

                    angle = math.degrees(math.acos(ffv_hero_vehicle.dot(ffv_target_vehicle)))

                    print(angle, target_vehicle)

                    if target_vehicle.get_transform().location.distance(wp.transform.location) > ego_location.distance(wp.transform.location):
                        if angle < 45:
                            print('GESTIONE_INCROCI: Sono gia nel mezzo del BBOX, non ti fermare pazzo!!!')
                            return (False, None, -1)

                #return (True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location))
                return (True, target_vehicle, 0.1)


        return (False, None, -1)


    def _generate_lane_change_path(self, waypoint, direction='left', distance_same_lane=10,
                                distance_other_lane=25, lane_change_distance=25,
                                check=True, lane_changes=1, step_distance=2):
        """
        This methods generates a path that results in a lane change.
        Use the different distances to fine-tune the maneuver.
        If the lane change is impossible, the returned path will be empty.
        """
        distance_same_lane = max(distance_same_lane, 0.1)
        distance_other_lane = max(distance_other_lane, 0.1)
        lane_change_distance = max(lane_change_distance, 0.1)

        plan = []
        plan.append((waypoint, RoadOption.LANEFOLLOW))  # start position

        option = RoadOption.LANEFOLLOW

        # Same lane
        distance = 0
        while distance < distance_same_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        if direction == 'left':
            option = RoadOption.CHANGELANELEFT
        elif direction == 'right':
            option = RoadOption.CHANGELANERIGHT
        else:
            # ERROR, input value for change must be 'left' or 'right'
            return []

        lane_changes_done = 0
        lane_change_distance = lane_change_distance / lane_changes

        # Lane change
        while lane_changes_done < lane_changes:

            # Move forward
            next_wps = plan[-1][0].next(lane_change_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]

            # Get the side lane
            if direction == 'left':
                if check and str(next_wp.lane_change) not in ['Left', 'Both']:
                    return []
                side_wp = next_wp.get_left_lane()
            else:
                if check and str(next_wp.lane_change) not in ['Right', 'Both']:
                    return []
                side_wp = next_wp.get_right_lane()

            if not side_wp or side_wp.lane_type != carla.LaneType.Driving:
                return []

            # Update the plan
            plan.append((side_wp, option))
            lane_changes_done += 1

        # Other lane
        distance = 0
        while distance < distance_other_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        return plan


    def get_bounding_box_corners(self, actor):
        """
        Restituisce le coordinate dei vertici del bounding box di un attore in CARLA.
        :param actor: l'attore di cui si vogliono trovare le coordinate del bounding box.
        :return: una lista di numpy array che rappresentano i vertici del bounding box.
        """
        bbox = actor.bounding_box
        trans = actor.get_transform()
        location = trans.location
        yaw = np.deg2rad(trans.rotation.yaw)
        # print("yaw: ", yaw)
        # Calcola le dimensioni del bounding box (divise per due), to manage Bicicle 
        extent_x = bbox.extent.x if bbox.extent.x != 0 else 0.6096/2
        extent_y = bbox.extent.y if bbox.extent.y != 0 else 1.7272/2
        extent_z = bbox.extent.z if bbox.extent.z != 0 else 1/2
        # print("extent_x: ",extent_x,"extent_y: ",extent_y,"extent_z: ",extent_z)
        # Calcola le coordinate del bounding box rispetto al centro del veicolo
        bounding_box = np.array([
            [extent_x, extent_y, 0],
            [-extent_x, extent_y, 0],
            [-extent_x, -extent_y, 0],
            [extent_x, -extent_y, 0],
            [extent_x, extent_y, 2 * extent_z],
            [-extent_x, extent_y, 2 * extent_z],
            [-extent_x, -extent_y, 2 * extent_z],
            [extent_x, -extent_y, 2 * extent_z]])

        # Ruota e trasla il bounding box in base all'orientamento e alla posizione dell'attore
        rotation = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]])
        # print("rotation: ", rotation)
        transformed_box = []
        for point in bounding_box:
            transformed_point = np.dot(rotation, point)
            transformed_point += np.array([location.x, location.y, location.z])
            transformed_box.append(transformed_point)

        return (transformed_box,extent_x,extent_y,extent_z)
    
