# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """


import random
import numpy as np
import carla
from basic_agent import BasicAgent
from local_planner import RoadOption
from behavior_types import Cautious, Aggressive, Normal
import operator
from local_planner import MyWaypoint
import math
from shapely.geometry import Polygon, LineString,Point
from math import pi
from misc import get_speed, positive, is_within_distance, compute_distance,draw_bbox, draw_waypoints

class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    def __init__(self, vehicle, behavior='normal', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        self._my_flag=False

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 4.5
        self.surpCounter = 0
        self._before_surpass_lane_id = None
        self.surpassing_security_step = 0
        self.surpass_vehicle = None
        self.security_step_to_reEnter = None
        self.security_i_rentered = None

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for stop sign.
        Prende info su tutti gli attori e li filtra con traffic_light e valuta quali tra questi semafori influenza la guida.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected
    
    def stop_sign_manager(self):
        """
        This method is in charge of behaviors for red lights.
        Prende info su tutti gli attori e li filtra con stop e valuta quale stop influenza la guida
        """
        actor_list = self._world.get_actors()
        stop_list = actor_list.filter("*stop*")
        car_list = actor_list.filter("*vehicle*")
        affected, _,dist_from_stop = self._affected_by_stop_sign(stop_list,car_list)
        return affected,dist_from_stop

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """
        # print("sono in logica x tailgating")
        # LaneChanginng, cerca di tenere in considerazione i vehicle che vengono da dietro.
        #anche in questo caso cambiano sempre gli angoli di considerazione.
        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        # se esiste veicolo che da fastidio e la velocita è minore del vehicle che da fastidio nn riesco a fare il cambio. In generale cambiano gli angoli che vengono passati alla funzione x la detection.
        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both)  and right_wpt.lane_type == carla.LaneType.Driving:
                # and waypoint.lane_id * right_wpt.lane_id > 0
                # Verifico anche dopo se sto x fare danni. Questo ci dice, se nn sopr vehicle e enlla traiettoria nn ho vehicle su cui impatto, se questo nn ci sta faccio manovra cambiando il path.
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    # print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
                    # consente di calcolare percorso da dove ci troviamo a dove voglaimo andare e viene dato al local planner, viene calcoalto dal mission planner, modifica della traiettoria originale.
            elif left_turn == carla.LaneChange.Left  and left_wpt.lane_type == carla.LaneType.Driving:
                # and waypoint.lane_id * left_wpt.lane_id > 0
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    # print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)

    def collision_and_car_avoid_manager(self, waypoint, distForNormalBehavior=80, my_up_angle_th=90):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        # logica è uguale a quella del pedone.
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        vehicle_list, dists = self.order_by_dist(vehicle_list, waypoint, 80, True)

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._our_vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=my_up_angle_th, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._our_vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=my_up_angle_th, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._our_vehicle_obstacle_detected(
                vehicle_list, distForNormalBehavior, up_angle_th=60)
            # tiene in considerazione anche
            # Check for tailgating
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0: # se sto gia in quello stato non modifico. 
                self._tailgating(waypoint, vehicle_list)

        # ci sono scenari dove siamo parcheggiati e ci dobbiamo inserire, passa un vehicle della polizia e la baseline lo prende sempre o comunque x evitarlo e prende vehicle dopo.

        return vehicle_state, vehicle, distance

    def gestione_incrocio(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        # logica è uguale a quella del pedone.
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        vehicle_list, dists = self.order_by_dist(vehicle_list, waypoint, 45, True)
        
        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        walker_list, dists = self.order_by_dist(walker_list, waypoint, 45, True)

        vehicle_list = list(vehicle_list) + list(walker_list)

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self.gestsione_incroci(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self.gestsione_incroci(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self.gestsione_incroci(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)
            # tiene in considerazione anche
            # Check for tailgating
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0: # se sto gia in quello stato non modifico. 
                self._tailgating(waypoint, vehicle_list)

        # ci sono scenari dove siamo parcheggiati e ci dobbiamo inserire, passa un vehicle della polizia e la baseline lo prende sempre o comunque x evitarlo e prende vehicle dopo.

        return vehicle_state, vehicle, distance

    def pedestrian_avoid_manager(self, waypoint, distForNormalBehavior=80):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """
        # prendo i pedoni, calcolo le distanze tra pedone e dove ci troviamo, prendo solo <10 metri di distanza.
        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        walker_list, dists = self.order_by_dist(walker_list, waypoint, distForNormalBehavior)

        # vedo se siamo in collisione con pedone, magari distanza piccola però ci ha gia superato, a seconda delle posizioni e di cosa dobbiamo fare valutiamo in modo diverso la _vehicle_obstacle_detected (in realtà sarebbe obj), si può usare x qualsiasi cosa in carla, l'importante è passare la lista di obj in ingresso. verifico se sono in collisione con la lista di obj passati. Resistuisce se obj influenza la nostra guida, chi è e la distanza.
        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._our_vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._our_vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._our_vehicle_obstacle_detected(walker_list, distForNormalBehavior, up_angle_th=60)

        return walker_state, walker, distance

    def bikers_avoid_manager(self, waypoint, distForNormalBehavior=80, my_up_angle_th=90):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """
        # prendo i pedoni, calcolo le distanze tra pedone e dove ci troviamo, prendo solo <10 metri di distanza.
        bikers_list = list(self._world.get_actors().filter("*vehicle.bh.crossbike*"))
        bikers_list += list(self._world.get_actors().filter("*vehicle.gazelle.omafiets*"))
        bikers_list += list(self._world.get_actors().filter("*vehicle.diamondback.century*"))

        bikers_list, dists = self.order_by_dist(bikers_list, waypoint, distForNormalBehavior)

        #controlliamo le tre condizioni differenti:
        if self._direction == RoadOption.CHANGELANELEFT:
            static_obj_state, static_obj, distance = self._our_vehicle_obstacle_detected(bikers_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            static_obj_state, static_obj, distance = self._our_vehicle_obstacle_detected(bikers_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            static_obj_state, static_obj, distance = self._our_vehicle_obstacle_detected(bikers_list, distForNormalBehavior, up_angle_th=my_up_angle_th)
        return static_obj_state, static_obj, distance

    def static_obstacle_avoid_manager(self, waypoint,distForNormalBehavior=80, my_up_angle_th=90):
        #FUNZIONE AGGIUNTA PER LA DETECTION DI OSTACOLI STATICI SULLA STRADA
        #filtrare tutt gli ostacoli statici
        static_obj_list = self._world.get_actors().filter("*static.prop*")
        static_obj_list, dists = self.order_by_dist(static_obj_list, waypoint, distForNormalBehavior,True)
        #controlliamo le tre condizioni differenti:
        if self._direction == RoadOption.CHANGELANELEFT:
            static_obj_state, static_obj, distance = self._our_vehicle_obstacle_detected(static_obj_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=my_up_angle_th, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            static_obj_state, static_obj, distance = self._our_vehicle_obstacle_detected(static_obj_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=my_up_angle_th, lane_offset=1)
        else:
            static_obj_state, static_obj, distance = self._our_vehicle_obstacle_detected(static_obj_list, distForNormalBehavior, up_angle_th=60)
            # if static_obj_state:
                # print("static_obj_state, static_obj, distance:",static_obj_state, static_obj, distance)
                #input()
        return static_obj_state, static_obj, distance

    def car_following_manager(self, vehicle, distance, debug=False, my_ttc = None):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        # sto in una situazione normale, nn ho vehicle che mi danno fastidio, pedoni o altro.. Seguo la macchina e basta. Prendo la velocita del vehicle, vedo la differenza di velocita, calcolo TimeToCollision. Il comportamento è quello di ridurre la velocità cosi da matchare la sua, viene fatto mano mano. Cambio quindi la nuova velocità obiettivo.
        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)
        maxttc = self._behavior.safety_time
        if my_ttc != None:
            maxttc = my_ttc
        # Under safety time distance, slow down.
        if maxttc > ttc > 0.0:
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),    
            self._vehicle.get_speed_limit()])
            self._local_planner.set_speed(target_speed)
            print("sono nella funzione car following target_speed:",target_speed)
            # input()
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * maxttc > ttc >= maxttc:
            target_speed = min([
                max(self._min_speed, vehicle_speed),    
            self._vehicle.get_speed_limit()])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = self._vehicle.get_speed_limit()
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def order_by_dist(self, object_list, waypoint, max_dist, check_not_our_vehicle=False):
        def dist(s): return s.get_location().distance(waypoint.transform.location)
        #qua ho messo dieci pero va scelto il giusto valore.
        #creiamo un dizionario in modo da ordinare a seconda delle distanze in ordine crescente
        if check_not_our_vehicle:
            static_obj_dict = {s:dist(s) for s in object_list if dist(s)<max_dist and s.id != self._vehicle.id and (("static" in s.type_id and s.bounding_box.extent.z >= 0.25) or "static" not in s.type_id)}
        else:
            static_obj_dict = {s:dist(s) for s in object_list if dist(s)<max_dist and (("static" in s.type_id and s.bounding_box.extent.z >= 0.25) or "static" not in s.type_id)}
        #otteniamo ora la lista corrispondente ordinata per valore
        ordered_dict = dict(sorted(static_obj_dict.items(),key=operator.itemgetter(1)))
        return (list(ordered_dict.keys()),list(ordered_dict.items()))

    def run_step(self, debug=False):
        """
        è il metodo che viene chiamato ad ogni tiemstep.  Prendo info ed eseguo il behavior planner, che può essere rappresentaot
        come una macchina a stati ( anche se è molto complesso). Di base gestisce in un certo ordine tute le cose viste
        nella descrizione. 

        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl        
        """

        self._update_information()
        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        bb_coords = self._vehicle.bounding_box.get_world_vertices(self._vehicle.get_transform())
        ego_vertexs_lane_id = [(self._map.get_waypoint(bb_coord)).lane_id for bb_coord in bb_coords]
        
        if True:
            print(ego_vehicle_loc)
            # input()
        
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(w): return w.get_location().distance(ego_vehicle_wp.transform.location)
        # for w in vehicle_list:
        # #     print(w.get_light_state())
        # input()
        # vehicle_list_red = [w for w in vehicle_list if dist(w) < 30]

        # for act in vehicle_list_red:
        #     draw_bbox(self._world, act)

        target_speed = None
        min_distance_for_em_stop = 15
        
        #set some parameters for controlled stop:
        max_real_accel = 3.86 #in m/s^2
        security_distance = 5 #significa che si fermerà 3 metri prima dell'ostacolo
        my_velocity = self._vehicle.get_velocity() #3d vector in m/s^2
        # for actor_snapshot in vehicle_list_red:
        #     draw_bbox(self._world, actor_snapshot)

        # # usato x verificare logica del soprasso, da rimuovere
        # for actor in vehicle_list:
        #         if 'role_name' in actor.attributes and actor.attributes['special_type'] == 'emergency':
        # #             print('auto police')
        #             continue
        #         if not('role_name' in actor.attributes and actor.attributes['role_name'] == 'hero' and actor.attributes['special_type'] != 'emergency'):
        #             actor.destroy() 

        # 1: Red lights and stops behavior, individua se esiste in un certo range un semaforo nello stato rosso. Memorizza l'attesa del semaforo, allo step successivo verifico QUELLO specifico semaforo e decido.
        if self.traffic_light_manager():
            return self.emergency_stop()
        
        

        # self._before_surpass_lane_id != ego_vehicle_wp.lane_id
        condToNotEnter = True
        if self._surpassing_obj and self.surpass_vehicle != None:
            condToNotEnter, v, d = self._our_vehicle_obstacle_detected(
                            [self.surpass_vehicle], max(
                                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)
            
        if self.security_i_rentered != None:
            if self.security_i_rentered == 0:
                self.security_i_rentered = None
            else:
                self.security_i_rentered -= 1

        if (self._surpassing_obj) and self._before_surpass_lane_id != None and not condToNotEnter:
            #capire xke non entra qua dentro
            if self.end_surpassing(ego_vehicle_wp):
                self.security_i_rentered = 5
                return self._local_planner.run_step(debug=debug)

        # 2.1: Pedestrian avoidance behaviors, verifico se ci sono pedoni che possono influenzare la guida
        obstacle_dict = {"walker": list(self.pedestrian_avoid_manager(ego_vehicle_wp))}
        obstacle_dict["biker"] = list(self.bikers_avoid_manager(ego_vehicle_wp))
        obstacle_dict["vehicle"] = list(self.collision_and_car_avoid_manager(ego_vehicle_wp))
        obstacle_dict["static_obj"] = list(self.static_obstacle_avoid_manager(ego_vehicle_wp, 100))


        # defiisce se eiste questo pedone, se esiste e si trova ad una distanza troppo vicina allora mi fermo!
        if obstacle_dict["walker"][0]:
            print('Walker State:')
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            print("WALKER STATE la distanza dal pedone è: ", obstacle_dict["walker"][2])
            # input()
            delta_v =  self._speed - 0.0
            if delta_v < 0:
                delta_v = 0
            decina = delta_v/10
            quadrato_decina = decina ** 2
            # Emergency brake if the car is very close.
            if obstacle_dict["walker"][2] < quadrato_decina + security_distance:
                return self.controlled_stop(obstacle_dict["walker"][1], obstacle_dict["walker"][2],minDistance=3)
        
        # police = self._world.get_actors().filter("*vehicle.dodge.charger_police*")
        # def dist(w): return w.get_location().distance(ego_vehicle_wp.transform.location)
        # police_list = [w for w in police if dist(w) < 30]

        # if police_list:
        #     vehicle_speed = get_speed(police_list[0])
        # #     print(vehicle_speed)
        # #     print("find auto police")
        #     if vehicle_speed == 0.0:
        #         #self._my_flag = True
        #         self._surpassing_police = True
        #         self._local_planner._change_line = "left"
        #         self._local_planner.set_speed(30) # da cambiare
        # #         print('neeed to taigating')
        #         return self._local_planner.run_step(debug=debug)
        # else:
        #     if self._surpassing_police: #check altre auto ferme
        #         #self.help_sorpassing(ego_vehicle_wp,'right')
        #         #self.lane_change('right')
        # #         print('torno a right')
        #         self._local_planner._change_line = "None"
        #         self._surpassing_police = False
        #         return self._local_planner.run_step(debug=debug)
        inc_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=int((self._speed_limit) / 3), all_list = True)
        try:
            near_junction = any(wp[0].is_junction for wp in inc_wpt)
        except:
            near_junction = True
        if not near_junction and not ego_vehicle_wp.is_junction and not self._surpassing_obj and self.security_i_rentered is None:
            if self.obstacle_avoidance(obstacle_dict, ego_vehicle_wp, ego_vertexs_lane_id):
                return self._local_planner.run_step(debug=debug)

        
        if obstacle_dict["biker"][0]:
            biker_vehicle_loc = obstacle_dict["biker"][1].get_location()
            biker_vehicle_wp = self._map.get_waypoint(biker_vehicle_loc)
            # print("biker check, biker_vehicle_wp.lane_id:  ", biker_vehicle_wp.lane_id, "self._before_surpass_lane_id: ", self._before_surpass_lane_id)
            # #input()
            if biker_vehicle_wp.lane_id != self._before_surpass_lane_id:
                # # input()
                delta_v =  self._speed - get_speed(obstacle_dict["biker"][1])
                if delta_v < 0:
                    delta_v = 0
                decina = delta_v/10
                quadrato_decina = decina ** 2

                target_forward_vector = obstacle_dict["biker"][1].get_transform().get_forward_vector()
                ego_forward_vector = self._vehicle.get_transform().get_forward_vector()
                dot_ve_wp = target_forward_vector.x * ego_forward_vector.x + target_forward_vector.y * ego_forward_vector.y + target_forward_vector.z * ego_forward_vector.z
                    
                print("BIKERS STATE la distanza dal veicolo è: ", obstacle_dict["biker"][2], "la sua lane è: ", biker_vehicle_wp.lane_id, "mentre la mia è: ", ego_vehicle_wp.lane_id, "la mia road option è:",  self._direction, ' il dot prod è ', dot_ve_wp)
                #if self._surpassing_obj:
                # Emergency brake if the car is very close.
                when_following = self._vehicle.get_speed_limit()/6.25
                if dot_ve_wp < 0.1:
                    return self.controlled_stop(obstacle_dict["biker"][1], obstacle_dict["biker"][2],minDistance=3)
                if obstacle_dict["biker"][2] < quadrato_decina + 1:
                    return self.controlled_stop(obstacle_dict["biker"][1], obstacle_dict["biker"][2],minDistance=2)
                elif obstacle_dict["biker"][2] < when_following and get_speed(obstacle_dict["biker"][1]) > 2:
                    return self.car_following_manager(obstacle_dict["biker"][1], obstacle_dict["biker"][2], False, 1.5)
        # 2.2: Car following behaviors
        

        # stesso principio del pedone.
        if obstacle_dict["vehicle"][0]:
            vehicle_vehicle_loc = obstacle_dict["vehicle"][1].get_location()
            vehicle_vehicle_wp = self._map.get_waypoint(vehicle_vehicle_loc) 
            bikers_list = ["vehicle.bh.crossbike", "vehicle.gazelle.omafiets", "vehicle.diamondback.century"]
            vehicle_vehicle_wp = self._map.get_waypoint(vehicle_vehicle_loc)
            try:
                # Ottenere le informazioni sulla marcatura a sinistra e destra del waypoint e le loro location
                left_waypoint_pos = vehicle_vehicle_wp.get_left_lane().transform.location
                right_waypoint_pos = vehicle_vehicle_wp.get_right_lane().transform.location
                # Calcolare la distanza euclidea tra i due waypoint
                distance_between_waypoint = math.sqrt((left_waypoint_pos.x - right_waypoint_pos.x)**2 + (left_waypoint_pos.y - right_waypoint_pos.y)**2)
                # Calcolare la distanza tra il waypoint e la posizione del veicolo
                distance = left_waypoint_pos.distance(vehicle_vehicle_loc)
            except:
                distance = 0
                distance_between_waypoint = -1

            if vehicle_vehicle_wp.lane_id != self._before_surpass_lane_id and not ego_vehicle_wp.is_junction and (distance < distance_between_waypoint) and obstacle_dict["vehicle"][1].type_id not in bikers_list and self.surpass_vehicle==None:
                #print('Vehicle State:')
                # Distance is computed from the center of the two cars,
                # we use bounding boxes to calculate the actual distance
                print("VEHICLE STATE la distanza dal veicolo è: ", obstacle_dict["vehicle"][2], "il veicolo è: ",obstacle_dict["vehicle"][1], "la sua lane è: ", vehicle_vehicle_wp.lane_id, "mentre la mia è: ", ego_vehicle_wp.lane_id, "la mia road option è:",  self._direction, "obstacle_dict[vehicle][1].type_id: ", obstacle_dict["vehicle"][1].type_id)
                # print("inoltre le sue luci sono ", obstacle_dict["vehicle"][1].get_light_state())
                #if self._surpassing_obj:
                # input()
                delta_v =  self._speed - get_speed(obstacle_dict["vehicle"][1])
                if delta_v < 0:
                    delta_v = 0
                decina = delta_v/10
                quadrato_decina = decina ** 2
                print("la mia self._speed è: ", self._speed, "get_speed(obstacle_dict['vehicle'][1]): ", get_speed(obstacle_dict['vehicle'][1]), "get_speed( self._vehicle): ", get_speed( self._vehicle), "quadrato_decina + security_distance: ", quadrato_decina + security_distance)
                # Emergency brake if the car is very close.
                if obstacle_dict["vehicle"][2] < quadrato_decina + security_distance:
                    print("vehicle closed: ", obstacle_dict["vehicle"][1], "ha una speed di: ", get_speed(obstacle_dict["vehicle"][1]), " la distanza è: ", obstacle_dict["vehicle"][2], "è entrato per quello che hai aggiunto?: ", (obstacle_dict["vehicle"][2] < 40 and get_speed(obstacle_dict["vehicle"][1]) < 5))
                    # #input()
                    return self.controlled_stop(obstacle_dict["vehicle"][1], obstacle_dict["vehicle"][2], 5)
                elif obstacle_dict["vehicle"][2] < max(self._vehicle.get_speed_limit(),15) and get_speed(obstacle_dict["vehicle"][1]) > 1:
                    print("sto per chiamare car followinf")
                    return self.car_following_manager(obstacle_dict["vehicle"][1], obstacle_dict["vehicle"][2])

        #AGGIUNTA PER GESTIRE OSTACOLI STATICI SULLA STRADA
        if obstacle_dict["static_obj"][0]:
            #static_obj_type = static_obj.attributes.get('object_type')
            #stop_cond = static_obj_type != "static.prop.dirtdebris01" or static_obj_type != "static.prop.dirtdebris02" or static_obj_type != "static.prop.dirtdebris03" or static_obj_type is not None
            static_obj = obstacle_dict["static_obj"][1]
            obs_distance = obstacle_dict["static_obj"][2]
            

            static_obj_loc = obstacle_dict["static_obj"][1].get_location()
            static_obj_wp = self._map.get_waypoint(static_obj_loc) 
            
            static_bb_coords = static_obj.bounding_box.get_world_vertices(static_obj.get_transform())
            obj_vertexs_lane_id = [(self._map.get_waypoint(bb_coord, lane_type=carla.LaneType.Any)).lane_id for bb_coord in static_bb_coords]
            
            try:
                # Ottenere le informazioni sulla marcatura a sinistra e destra del waypoint e le loro location
                left_waypoint_pos = static_obj_wp.get_left_lane().transform.location
                right_waypoint_pos = static_obj_wp.get_right_lane().transform.location
                # Calcolare la distanza euclidea tra i due waypoint
                distance_between_waypoint = math.sqrt((left_waypoint_pos.x - right_waypoint_pos.x)**2 + (left_waypoint_pos.y - right_waypoint_pos.y)**2)
                # Calcolare la distanza tra il waypoint e la posizione del veicolo
                distance = left_waypoint_pos.distance(static_obj_loc)
            except:
                distance = 0
                distance_between_waypoint = -1


            if static_obj.type_id  != 'static.prop.mesh' and not self._surpassing_obj and ego_vehicle_wp.lane_id in obj_vertexs_lane_id and (distance < 1.5*distance_between_waypoint):
                print("potrei cominciare a frenare per STATIC OBJ")
                # input()
                print("static object più alto di mezzo metro, mi fermo")
                print("STATIC OBJ la distance dall'obj è: ", obs_distance)
                # input()
                
                distance_to_start_stop = self.compute_warning_distance(static_obj,max_real_accel, my_velocity)
                if obs_distance<= distance_to_start_stop + security_distance:
                    print("STATIC OBJ ora mi fermo tranqui boyyz")
                    #input()
                    return self.controlled_stop(static_obj, obs_distance, 10 + 0.1*get_speed(self._vehicle))
        
        # 1: Red lights and stops behavior, individua se esiste in un certo range un semaforo nello stato rosso. Memorizza l'attesa del semaforo, allo step successivo verifico QUELLO specifico semaforo e decido.
        affected_by_stop,dist_from_stop = self.stop_sign_manager()
        if affected_by_stop:
            print("sto in stop_sign")
            return self.controlled_stop(distance=dist_from_stop,minDistance=2)
        
        # 3: Intersection behavior, consente di capire se siete in un incrocio, ma il comportamento è simile al normale, non ci sta una gestione apposita. La gestione degli incroci viene gestta in obj detection. Stesso comportamento normal behavor ma solo più lento.
        if (ego_vehicle_wp.is_junction or self._incoming_waypoint.is_junction):
            self._local_planner.set_speed(20)
            control = self._local_planner.run_step(debug=debug)

            vehicle_state, vehicle, v_distance = self.gestione_incrocio(ego_vehicle_wp)
            # stesso principio del pedone.
            if vehicle_state:
                print('Junction State:')
                # input()
                vehicle_vehicle_loc = vehicle.get_location()
                vehicle_vehicle_wp = self._map.get_waypoint(vehicle_vehicle_loc) 
                if vehicle_vehicle_wp.lane_id != self._before_surpass_lane_id:
                    delta_v =  self._speed - get_speed(vehicle)
                    if delta_v < 0:
                        delta_v = 0
                    # Emergency brake if the car is very close.
                    if v_distance < max(self._behavior.braking_distance, min_distance_for_em_stop +1) + delta_v * 0.2:
                        return self.controlled_stop(vehicle, v_distance,minDistance=2)
                    else:
                        return self.car_following_manager(vehicle, v_distance)
            return control

        # 4: Normal behavior, prende target speed, è una variabile che ti dice quanto manca a quello che ti serve. Il local planer contiene anche i controllori, quindi gli stiamo dicendo anche questo. Obj control contiene cose di carla sul dafarsi
        print("NORMAL BEHAVIOUR")
        # input()
        if target_speed is None:
            target_speed = self._vehicle.get_speed_limit()
        if self._surpassing_obj:
            self._local_planner.set_speed(max(80,self._vehicle.get_speed_limit()))
        else:
            self._local_planner.set_speed(target_speed)
        control = self._local_planner.run_step(debug=debug)

        # print(self._local_planner._target_speed)

        return control

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = self._local_planner.run_step_only_lateral()
        # per le derapate a True
        return control

    def compute_warning_distance(self,obstacle,max_real_accel, my_velocity):
        obstacle_velocity = obstacle.get_velocity() #3d vector in m/s^2
        my_relative_velocity = np.linalg.norm(np.array([ my_velocity.x - obstacle_velocity.x,my_velocity.y- obstacle_velocity.y, my_velocity.z - obstacle_velocity.z]))
        distance_to_start_stop = (pow(my_relative_velocity,2))/(2*max_real_accel) #a che distanza devo iniziare a frenare 
        print("comp war dist, obstacle_velocity: ", obstacle_velocity, "distance to start stop: ", distance_to_start_stop)
        # input()
        return distance_to_start_stop
    
    def controlled_stop(self,t_vehicle=None, distance = 0.0, minDistance=4):
            my_velocity = self._vehicle.get_velocity()
            if distance<=minDistance:
                print("vado in emergency")
                # input()
                control = self.emergency_stop()
            else: 
                norm_velocity = np.linalg.norm(np.array([my_velocity.x ,my_velocity.y, my_velocity.z]))
                print("norm_velocity: ", norm_velocity)
                speed_to_sub = (3.86*norm_velocity/distance) 
                if get_speed(self._vehicle)>50:
                    speed_to_sub *= 15
                target_speed = norm_velocity - speed_to_sub
                if target_speed < 2:
                    target_speed = 2
                #target_speed = norm_velocity - (max_sim_accel*sim_time)
                print("sono in decelerate, la target speed che sto settando è:", target_speed, "la mia vel è: ", norm_velocity)
                #input()
                self._local_planner.set_speed(target_speed * 3.6)
                # input()
                control = self._local_planner.run_step()
            # per le derapate a True
            return control

    def cond_to_start_surpass(self, ego_vehicle_wp, dir):
        #ciò che noi superiamo sono oggetti statici e veicoli sia bici che auto
        my_obstacles = list(self._world.get_actors().filter("*vehicle*"))
        obstacles = my_obstacles + list(self._world.get_actors().filter("*static.prop*"))
        obstacles, _ = self.order_by_dist(obstacles, ego_vehicle_wp, np.inf)#ordiniamoli per distanza
        print("len(obstacles): ",len(obstacles))
        #ottengo informazioni sul mio veicolo
        ego_corners,ego_ext_x, ego_ext_y, ego_ext_z = self.get_bounding_box_corners(self._vehicle) #i vertici del mio veicolo e la sua dimensione
        ego_lane_id = ego_vehicle_wp.lane_id #id della lane su cui io mi trovo
        #da aggiornare ad iterazione successiva
        previous_poligon = Polygon(ego_corners)#poligono associato a me
        previous_actor = self._vehicle #setto il mio veicolo come previous
        print("informazioni sul mio veicolo: ", self._vehicle,ego_ext_x, ego_ext_y, ego_ext_z, "la mia posizione attuale è: ", ego_vehicle_wp.transform.location)
        
        #devo ottenere informazioni sui veicoli da sorpassare: devono essere completamente o parzialmente nella mia lane e devono essere davanti a me:
        to_surpass = [] #conterrà tutti i veicoli che dovrò sorpassare
        distance_to_surpass = 10 # DA VERIFICARE sarà lo spazio totale che dovrò sorpassare, lo settiamo in partenza di 15 per considerare un certo parametro di sicurezza
        for obs in obstacles:
            #chiamo la funzione mettendomi nelle condizioni di guardare solo davanti a me
            obstacle_state, obstacle, _ = self._our_vehicle_obstacle_detected([obs], np.inf, up_angle_th=60)
            if obstacle_state: #significa che il veicolo invade e sta davanti a me
                obstacle_corners, obs_ext_x, obs_ext_y, obs_ext_z  = self.get_bounding_box_corners(obstacle) #ottengo le caratteristiche dell'ostacolo in questione
                obstacle_polygon = Polygon(obstacle_corners) #ottengo il poligono ad esso associato
                distance = previous_poligon.distance(obstacle_polygon) #ottengo la distanza tra veicolo precedente e successivo
                print("informazioni sull'ostacolo: ", obstacle, obs_ext_x, obs_ext_y, obs_ext_z)
                # input()
                if distance < 10 or previous_actor==self._vehicle: #DA GENERALIZZARE, condizione per cui non posso ancora reinserirmi in carreggiata
                    #distance_to_surpass +=  distance + obs_ext_x #aggiorno la distanza da sorpassare
                    to_surpass.append(obstacle) #aggiungo questo agli ostacoli da sorpassare
                else: # significa che posso reinserirmi, quindi smetto di guardare gli altri veicoli perchè sono ordinati per distanza
                    break
                #faccio aggiornamento per il ciclo
                previous_poligon = Polygon(obstacle_corners) 
                previous_actor = obstacle
        print("Veicoli da sorpassare: ", to_surpass)
        if dir == "left":
            self._direction = RoadOption.CHANGELANELEFT
        elif dir == "right":
            self._direction = RoadOption.CHANGELANERIGHT
        #FIN QUI C'È L'ANALISI DELLO SPAZIO DA SORPASSARE SE I VEICOLI FOSSERO FERMI
        #per il calcolo delle velocità utilizzeremo il metodo delle velocità relative:
        standard_acceleration = 2.61 #valutata in m/s^2 è valutata empiricamente considerando la massima accelerazione che puo avere un veicolo, ed è messa proporzionale
        #a 0.75 che è il massimo throttle che viene realizzato
        last_surpass = to_surpass[-1]
        last_surpass_corners, last_surpass_ext_x, last_surpass_ext_y, last_surpass_ext_z  = self.get_bounding_box_corners(last_surpass) 
        distance_to_surpass +=  Polygon(ego_corners).distance(Polygon(last_surpass_corners)) + 2*last_surpass_ext_x
        last_surpass_lane_id = (self._map.get_waypoint(last_surpass.get_transform().location, lane_type=carla.LaneType.Any)).lane_id
        print("il lane id dell'ultimo da sorpassare è: ", last_surpass_lane_id, "il mio è: ",ego_lane_id )
        last_vehicle_velocity = get_speed(last_surpass)/3.6 #la valutiamo in metri al secondo
        print("l'ultimo ostacolo da sorpassare è: ", last_surpass, "la sua velocità in m/s è: ",last_vehicle_velocity)
        my_start_velocity = get_speed(self._vehicle)/3.6 #velocità attuale del mio veicolo in m/s
        if ego_lane_id*last_surpass_lane_id > 0:
            my_start_relative_velocity = my_start_velocity - last_vehicle_velocity #velocità attuale in m/s del mio veicolo in termini relativi rispetto all'ultimo veicolo da sorpassare
        else:
            my_start_relative_velocity = my_start_velocity + last_vehicle_velocity #velocità attuale in m/s del mio veicolo in termini relativi rispetto all'ultimo veicolo da sorpassare
        print("mia velocità iniziale in assoluto: ", my_start_velocity, "mia velocità iniziale relativa: ", my_start_relative_velocity)
        my_end_relative_velocity = math.sqrt(2*standard_acceleration*distance_to_surpass + pow(my_start_relative_velocity,2))#velocità finale del mio veicolo in m/s
        my_end_velocity = math.sqrt(2*standard_acceleration*distance_to_surpass + pow(my_start_velocity,2))#velocità finale in m/s del mio veicolo in termini relativi rispetto all'ultimo veicolo da sorpassare
        print("mia velocità finale in assoluto: ", my_end_velocity, "mia velocità finale relativa: ", my_end_relative_velocity)
        time_to_surpass = (my_end_relative_velocity - my_start_relative_velocity)/standard_acceleration #tempo richiesto per completare il sorpasso
        print("Il totale spazio da sorpassare è: ", distance_to_surpass,"il tempo richiesto è: ", time_to_surpass)
        print("la mia velocità è: ", get_speed(self._vehicle), "la mia accelerazione è:", self._vehicle.get_acceleration())
        #ora devo calcolare il punto reale a cui io arriverò, quindi il reale spazio che ho percorso
        real_distance = (0.5*standard_acceleration*pow(time_to_surpass,2)) + (my_start_velocity*time_to_surpass) #reale distanza percorsa
        to_arrive = ego_vehicle_wp.next(real_distance)[0] #waypoint a cui arriverò dopo aver superato
    
        #ottengo i waypoint corrispettivi sulla lane che invaderò per il sorpasso
        if self._direction == RoadOption.CHANGELANERIGHT:
            offset = 1
            corr_ego_wpt = ego_vehicle_wp.get_right_lane() #waypoint corrispondente al punto da cui parto nella lane che invaderò
            corr_to_arrive = to_arrive.get_right_lane() #waypoint corrispondente al punto a cui arriverò nella lane che invaderò
        elif self._direction == RoadOption.CHANGELANELEFT:
            offset = -1
            corr_ego_wpt = ego_vehicle_wp.get_left_lane() #waypoint corrispondente al punto da cui parto nella lane che invaderò
            corr_to_arrive = to_arrive.get_left_lane() #waypoint corrispondente al punto a cui arriverò nella lane che invaderò
        
        
        #analizzo se sull'altra corsia che sto andando ad invadere posso avere delle possibili collisioni
        possible_collident = None #possibile ostacolo con cui colliderei nella lane che invaderò
        #sfrutto tutta la lista di obstacles prima calcolata
        my_obstacles, _ = self.order_by_dist(my_obstacles, ego_vehicle_wp, np.inf)#ordiniamoli per distanza
        for opp in my_obstacles: #analizzo quelli che sono gli oggetti che potrei trovarmi di fronte invadendo l'atra corsia
            possible_collision, opposite, _ = self._our_vehicle_obstacle_detected([opp], np.inf, up_angle_th=90, lane_offset=offset)
            if possible_collision: 
                possible_collident = opposite
                possible_collident_wpt = self._map.get_waypoint(opposite.get_transform().location, lane_type=carla.LaneType.Any)
                break # appena l'ho trovato posso fermarmi perche sono in ordine di vicinanza a me
        if possible_collident != None: #ho trovato un possibili collidente
            poss_coll_speed = get_speed(possible_collident)/3.6 #velocità del possibile collidente
            print("possibile collidente: ", possible_collident, "la sua velocità è: ", poss_coll_speed, "la sua accelerazione è: ",possible_collident.get_acceleration() ,"la sua posizione attuale è: ", possible_collident_wpt.transform.location)
            #se il collidente è in movimento devo valutare anche lo spazio che potrà percorrere mentre io non conccludo il sorpasso
            #altrimenti devo valutare solo la sua posizione:
            #moto uniformemente accelerato anche per il veicolo che mi viene di faccia: 
            collident_acceleration = min(np.linalg.norm(np.array([(possible_collident.get_acceleration()).x,(possible_collident.get_acceleration()).y,(possible_collident.get_acceleration()).z])), 3)
            print("colldent acceleration: ", collident_acceleration)
            try:
                space_to_collide = (possible_collident_wpt.transform.location).distance(corr_to_arrive.transform.location)
                colide_vector = np.array([
                    corr_to_arrive.transform.location.x - possible_collident_wpt.transform.location.x,
                    corr_to_arrive.transform.location.y - possible_collident_wpt.transform.location.y,
                    corr_to_arrive.transform.location.z - possible_collident_wpt.transform.location.z])
                collident_forward_vector = possible_collident.get_transform().get_forward_vector()
                scalar_val = np.dot(np.array([colide_vector[0],colide_vector[1], colide_vector[2]]), np.array([collident_forward_vector.x,collident_forward_vector.y,collident_forward_vector.z]))
            except:
                return False,last_surpass 
            if poss_coll_speed >= self._vehicle.get_speed_limit()/(4*3.6): #il possibile colidente è in movimento
                print("poss_coll_speed è diverso da zero")
                factor = 2*poss_coll_speed/collident_acceleration
                time_to_collide =  (math.sqrt(pow(factor,2) + 8*(space_to_collide/collident_acceleration))-factor)/2
                
            else: # il possible collident è fermo, quindi vedo solo che si trova in una posizione tale per cui io riesco a terminare il mio sorpasso
                print("il possible collident è fermo")
                print("il to arrive si trova: ", to_arrive.transform.location)
                time_to_collide = space_to_collide/(possible_collident.get_speed_limit()/3.9)
            print("time_to_collide: ", time_to_collide, "time_to_surpass: ", time_to_surpass, "space_to_collide: ", space_to_collide)
            print("scalar_val: ", scalar_val)
            if time_to_surpass < time_to_collide and (scalar_val > 0 or dir == "right"):
                print("posso superare ritorno true da cond to surpass")
                # input()
                return True, last_surpass
            else:
                print("non posso superare ritorno false da cond to surpass")
                # # input()
                return False,last_surpass
        else: #il possible collident è none, cioe non ho trovato nessun possibile ostacolo nell'altra corsia
            print("Sto per ritornare True ma sono nell'ultimo else")
            # input()
            return True, last_surpass



    def start_surpassing(self, obj_to_s, ego_vehicle_wp, dir):
        self._surpassing_obj = True
        last_dir = self._direction
        if obj_to_s:
            enable, last_surpass = self.cond_to_start_surpass(ego_vehicle_wp, dir)
            if enable:
                
                list_no_other_step = ["vehicle.bh.crossbike", "vehicle.gazelle.omafiets", "vehicle.diamondback.century", "static.prop.trafficwarning","static.prop.warningaccident"]
                #input()
                #if not com_vehicle_state or (com_vehicle_state and com_vehicle_distance>80):
                #print('STO PER STARTARE IL SORPASSO, IL VEICOLO DISTA: ', com_vehicle_distance, "ed è: ", com_vehicle)
                # input()
                #self._my_flag = True
                self._before_surpass_lane_id = ego_vehicle_wp.lane_id
                #self.help_sorpassing(ego_vehicle_wp,'left')
                self._local_planner._change_line = "shifting"
                self._local_planner.delta = self.meters_shifting(last_surpass, dir)
                if last_surpass.type_id in list_no_other_step:
                    print("ho impostato a 0 gli step prima di rientrare")
                    self.security_step_to_reEnter = 0
                else:
                    self.security_step_to_reEnter = 3
                # print(self._local_planner.delta)
                self._local_planner.dir = dir
                if dir != "right":
                    self._local_planner.set_speed(self._vehicle.get_speed_limit() * 2, True) # da cambiare
                else:
                    self._local_planner.set_speed(get_speed(self._vehicle) * (2/3))
                self.surpass_vehicle = obj_to_s
                print('sto per superare')
                self._direction = last_dir
                # input()
                return True
            else:
                self._surpassing_obj = False
                self._direction = last_dir
                # print('CI STA UN TIZIO CHE NON MI FA SORPASSARE LA DIST: ', com_vehicle_distance, "ed è: ", com_vehicle)
        return False

    def end_surpassing(self, ego_vehicle_wp):
        last_dir = self._direction
        if self._local_planner._change_line=="left":
            self._direction = RoadOption.CHANGELANERIGHT
        elif self._local_planner._change_line=="right":        
            self._direction = RoadOption.CHANGELANELEFT
        bikers_list = ["vehicle.bh.crossbike", "vehicle.gazelle.omafiets", "vehicle.diamondback.century"]

        com_biker_state, com_biker, com_biker_distance = self.bikers_avoid_manager(ego_vehicle_wp,distForNormalBehavior=self._speed_limit/3,my_up_angle_th=75)
        com_vehicle_state, com_vehicle, com_vehicle_distance = self.collision_and_car_avoid_manager(ego_vehicle_wp,distForNormalBehavior=self._speed_limit/3,my_up_angle_th=85)
        com_obj_state, com_obj, com_obj_distance = self.static_obstacle_avoid_manager(ego_vehicle_wp, distForNormalBehavior=self._speed_limit/3,my_up_angle_th=30)
        
        self._direction = last_dir
        if com_vehicle_state:
            print("voglio chiudere il sorpasso, l'ostacolo ci sta ancora: ",com_vehicle)
        # if self._local_planner._change_line != "None":
            # print("non mi rientrare com_vehicle: ",com_vehicle)
            #input()
        toAdd = 0
        if not com_vehicle_state:
            type = "com_vehicle.type_id"
        else:
            type = com_vehicle.type_id
        if (not com_vehicle_state and not com_obj_state and not com_biker_state) or (com_vehicle_state and not com_biker_state and type in bikers_list):
            if self._local_planner.dir == 'right': toAdd += 2
            if self.surpassing_security_step > self.security_step_to_reEnter + toAdd:
                print("(com_vehicle_state and not com_biker_state and com_vehicle in bikers_list): ",(com_vehicle_state and not com_biker_state and type in bikers_list))
                print("com_biker_state: ", com_biker_state)
                print("type: ", type)
                print("com_vehicle_state: ", com_vehicle_state)
                # input()
                print('STO PER RIENTRARE IN CORSIA')
                # input()
                self.surpass_vehicle = None
                self._local_planner._change_line = "None"
                self._local_planner.delta = 0
                self._local_planner.dir = "left"
                self._surpassing_obj = False
                self._before_surpass_lane_id = None
                self.surpassing_security_step = 0
                self.security_step_to_reEnter = None
                return True
            else:
                print("sto per ritornare false quindi non rientro")
                # input()
                self.surpassing_security_step += 1
                # print("self.surpassing_security_step: ", self.surpassing_security_step)
                # input()
        return False
            
    def obstacle_avoidance(self, obj_dict, waypoint, ego_vertexs_lane_id):
        valori = []
        for valore in obj_dict.values():
            if valore[0]:
                valori.append(valore[1])
        ordered_objs,dists = self.order_by_dist(valori, waypoint, 45, True)
        if len(ordered_objs) > 0 and dists[0][1]<15:
            # print(ordered_objs[0])
            # print('dists[0][1]',dists[0][1])
            # print("state 1")
            # input()
            obj_bb_coords, e_x, e_y, e_z = self.get_bounding_box_corners(ordered_objs[0])
            obj_vertexs_lane_id = [(self._map.get_waypoint(carla.Location(bb_coord[0], bb_coord[1], bb_coord[2]))).lane_id for bb_coord in obj_bb_coords]
            int_list = list(set(obj_vertexs_lane_id) & set(ego_vertexs_lane_id))
            print("obj_vertexs_lane_id: ", obj_vertexs_lane_id, "ego_vertexs_lane_id: ", ego_vertexs_lane_id)
            # draw_bbox(self._world, ordered_objs[0])
            not_my_lane_list = list(set(obj_vertexs_lane_id) - set(int_list))
            print("int_list: ", int_list, "not_my_lane_list: ", not_my_lane_list)
            # input()
            if len(int_list)>0 and len(not_my_lane_list) == 0:
                print('len(int_list): ',len(int_list),'len(not_my_lane_list): ',len(not_my_lane_list))
                # print('ordered_objs[0].type_id: ',ordered_objs[0].type_id)
                # logica per cominciare il sorpasso  
                #input()      
                if ordered_objs[0].type_id in ['vehicle.bh.crossbike','vehicle.gazelle.omafiets','vehicle.diamondback.century']:
                    print("e ciclette")
                    target_forward_vector = ordered_objs[0].get_transform().get_forward_vector()
                    ego_forward_vector = self._vehicle.get_transform().get_forward_vector()
                    cond = abs(np.dot(np.array([target_forward_vector.x,target_forward_vector.y, target_forward_vector.z]), np.array([ego_forward_vector.x,ego_forward_vector.y,ego_forward_vector.z]))) < 0.3
                    if dists[0][1]<15 and get_speed(ordered_objs[0]) <= 20 and not self._surpassing_obj and not cond:
                        if self.start_surpassing(ordered_objs[0], waypoint, "left"):
                            print("crossbike object surpass")
                            print("left1")
                            # input()
                            return True
                if 'vehicle' in ordered_objs[0].type_id:
                    print("o veicl")
                
                #get_ligth_state, vanno aggiunte altre condizioni, non tutte gli stati delle luci sono uguali per i vehicle
                # orientamento semaforo, cosi gestisco se si trova diffronte a me o no,
                    if dists[0][1]<20 and ordered_objs[0].get_light_state() in [1537,1539, 49] and not self._surpassing_obj and get_speed(ordered_objs[0]) <= 3 :
                        if self.start_surpassing(ordered_objs[0], waypoint, "left"):
                            print("vehicle object surpass")
                            print("left2")
                            # input()
                            return True
                if ordered_objs[0].type_id in ['static.prop.trafficwarning','static.prop.warningaccident']:
                    print("o static")
                
                #type_id, vanno aggiunte altre condizioni, per altri obj statici in mezzo road o generalizzare con location vicino awaypoint road
                    if dists[0][1]<20 and not self._surpassing_obj:
                        if self.start_surpassing(ordered_objs[0], waypoint, "left"):
                            # print('start surpassing obj')
                            print("static object surpass")
                            print("left3")
                            # input()
                            return True
            #condizione per verificare che quest'oggetto invada parzialmente la mia lane (da superare)
            elif len(int_list)>0:
                print("state 2")
                #input()
                print("len(int_list): ", len(int_list), "len(not_my_lane_list): ", len(not_my_lane_list))
                if not_my_lane_list[0] > waypoint.lane_id:
                    print("not_my_lane_list[0]: ", not_my_lane_list[0], "waypoint.lane_id: ", waypoint.lane_id)
                    if self.start_surpassing(ordered_objs[0], waypoint, "left"):
                        print("ordered_objs[0]: ", ordered_objs[0])
                        # print("state 3")
                        print("left4")
                        # input()
                        return True
                else:
                    target_forward_vector = ordered_objs[0].get_transform().get_forward_vector()
                    ego_forward_vector = self._vehicle.get_transform().get_forward_vector()
                    dot_ve_wp = target_forward_vector.x * ego_forward_vector.x + target_forward_vector.y * ego_forward_vector.y + target_forward_vector.z * ego_forward_vector.z
                    
                    print("questo è il valore del dot product: ", dot_ve_wp)
                    print("e i miei parametri sono ", dot_ve_wp,dists[0][1])
                    # input('sono in else state 2')
                    if dot_ve_wp < 0 and dists[0][1]<17: 
                        if self.start_surpassing(ordered_objs[0], waypoint, "right"):
                            print("sono nell'if del right" , dists[0][1])
                            # input()
                            # print("state 3.1")
                            print(dot_ve_wp)
                            # input('sono dove dicevamo')
                            return True
        return False

    def meters_shifting(self, target_vehicle, dir):
        obs_type = target_vehicle.type_id
        print("target_vehicle: ", target_vehicle, "obs_type: ", obs_type)
        if obs_type in ["vehicle.bh.crossbike", "vehicle.gazelle.omafiets", "vehicle.diamondback.century"]:
            sec_costant = 0.4
            print("sec_costant: ", sec_costant, "obs_type: ", obs_type)
            # input()
        elif dir != "right":
            sec_costant = 1.2
            print("sec_costant: ", sec_costant)
            # input()
        else:
            sec_costant = 0.2
            print("sec_costant: ", sec_costant)
            # input()
        my_lat_extend = self._vehicle.bounding_box.extent.y
        target_transform = target_vehicle.get_transform()
        target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)
        # draw_waypoints(self._vehicle.get_world(), [target_wpt], 1.0)
        if str(target_wpt.lane_type) != 'Driving':

            # input("va tutto malissomo")
            target_wpt = target_wpt.get_left_lane()
        #AGGIUNTA DI PROVA
        #target_wpt è il punto centrale della lane su cui si trova il veicolo target
        if self._direction == RoadOption.CHANGELANELEFT:
            corresponding_t_wpt = target_wpt.get_left_lane() #qui deve essere chiamato left o right a seconda di dove mi trovo 
        elif self._direction == RoadOption.CHANGELANERIGHT:
            corresponding_t_wpt = target_wpt.get_right_lane() #qui deve essere chiamato left o right a seconda di dove mi trovo 
        else:
            corresponding_t_wpt = target_wpt
        diff_lane = np.array([
            target_wpt.transform.location.x - corresponding_t_wpt.transform.location.x,
            target_wpt.transform.location.y - corresponding_t_wpt.transform.location.y,
            target_wpt.transform.location.z - corresponding_t_wpt.transform.location.z])
        # print("corresponding_t_wpt: ",corresponding_t_wpt,"target_transform.location: ",target_transform.location,"target_wpt: ", target_wpt,"np.linalg.norm(diff_lane): ", np.linalg.norm(diff_lane))
        #ottenere tutti i vertici del target vehicle target
        # # Otteniamo i vertici del bounding box in coordinate locali
        target_bb, e_x, e_y, e_z = self.get_bounding_box_corners(target_vehicle)
        #target_bb = target_vehicle.bounding_box.get_world_vertices(target_vehicle.get_transform())
        # print("target_bb[0]:", target_bb[0])
        distances = {} 
        # draw_bbox(self._world, target_vehicle)
        for vertex in target_bb:
            #loc= carla.Location(vertex.x,vertex.y,vertex.z)
            #carla.DebugHelper.draw_point(loc,0.5)
            #obtain the corresponding waypoint
            #calcoliamo la differenza di ciascun vertice dal target left
            distances[tuple(vertex)] = (np.linalg.norm(np.array([
                vertex[0] - corresponding_t_wpt.transform.location.x,
                vertex[1] - corresponding_t_wpt.transform.location.y,
                0])))
        #obtain the vertex associated with the max distance
        # print("distances: ",distances)
        max_vertex = min(distances, key = lambda k: distances[k])
        # print("max_vertex: ",max_vertex)
        #input()
        #compute the distance between max_vertex and the target_wpt
        diff_points = np.array([
            target_wpt.transform.location.x - max_vertex[0],
            target_wpt.transform.location.y - max_vertex[1],
            target_wpt.transform.location.z - max_vertex[2]])
        dot_product, cos_for_sign = self.proj(diff_points, diff_lane)
        if abs(cos_for_sign) > pi/4:
            dot_product = np.copysign(np.linalg.norm(dot_product), -1)
        else:
            dot_product = np.copysign(np.linalg.norm(dot_product), 1)
        # print("diff_points: ",diff_points, "max_vertex: ",max_vertex)
        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        how_much_move = abs(dot_product - np.linalg.norm(diff_lane)/2)
        # print("target_wpt.lane_type: ",target_wpt.lane_type)
        # print("how_much_move: ",how_much_move, "cos_for_sign: ", cos_for_sign, "dot_product: ", dot_product,"np.linalg.norm(diff_lane)/2:",np.linalg.norm(diff_lane)/2 )
        #input()
        if ego_vehicle_wp.lane_id != target_wpt.lane_id:
            # print("case my lane diversa da target lane, mi muovo di: ", abs(dot_product - np.linalg.norm(diff_lane)/2) + sec_costant, "dot-product: ", dot_product)
            # input()
            return abs(dot_product - np.linalg.norm(diff_lane)/2) + sec_costant
        else:
            # print("case my lane uguale a target lane, mi muovo di: ", abs(my_lat_extend  - dot_product) + sec_costant, "how_move è:", how_much_move, "dot-product: ", dot_product, "e_x/2: ", e_x/2)
            # input()
            return abs(my_lat_extend - dot_product) + sec_costant
        
#restituisce il vettore proiezione di A su B e il modulo di questa
    def proj(self, A, B):
        cos_theta = np.dot(A, B) / np.linalg.norm(B)
        return (cos_theta * (B / np.linalg.norm(B)), cos_theta)



        

