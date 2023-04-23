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


from misc import get_speed, positive, is_within_distance, compute_distance, draw_bbox

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
        This method is in charge of behaviors for red lights.
        Prende info su tutti gli attori e li filtra con traffic_light e valuta quali tra questi semafori influenza la guida.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected
    
    def stop_sign_manager(self):
        """
        This method is in charge of behaviors for red lights.
        Prende info su tutti gli attori e li filtra con traffic_light e valuta quali tra questi semafori influenza la guida.
        """
        actor_list = self._world.get_actors()
        stop_list = actor_list.filter("*stop*")
        affected, _ = self._affected_by_stop_sign(stop_list)
        return affected

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """
        print("sono in logica x tailgating")
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
                    print("Tailgating, moving to the right!")
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
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)

    def collision_and_car_avoid_manager(self, waypoint):
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
        vehicle_list = self.order_by_dist(vehicle_list, waypoint, 45, True)

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._our_vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._our_vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._our_vehicle_obstacle_detected(
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

    def pedestrian_avoid_manager(self, waypoint):
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
        walker_list = self.order_by_dist(walker_list, waypoint, 45)

        # vedo se siamo in collisione con pedone, magari distanza piccola però ci ha gia superato, a seconda delle posizioni e di cosa dobbiamo fare valutiamo in modo diverso la _vehicle_obstacle_detected (in realtà sarebbe obj), si può usare x qualsiasi cosa in carla, l'importante è passare la lista di obj in ingresso. verifico se sono in collisione con la lista di obj passati. Resistuisce se obj influenza la nostra guida, chi è e la distanza.
        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._our_vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._our_vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._our_vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return walker_state, walker, distance

    def bikers_avoid_manager(self, waypoint):
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

        bikers_list = self.order_by_dist(bikers_list, waypoint, 45)

        #controlliamo le tre condizioni differenti:
        if self._direction == RoadOption.CHANGELANELEFT:
            static_obj_state, static_obj, distance = self._our_vehicle_obstacle_detected(bikers_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            static_obj_state, static_obj, distance = self._our_vehicle_obstacle_detected(bikers_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            static_obj_state, static_obj, distance = self._our_vehicle_obstacle_detected(bikers_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)
        return static_obj_state, static_obj, distance

    def static_obstacle_avoid_manager(self, waypoint):
        #FUNZIONE AGGIUNTA PER LA DETECTION DI OSTACOLI STATICI SULLA STRADA
        #filtrare tutt gli ostacoli statici
        static_obj_list = self._world.get_actors().filter("*static.prop*")
        static_obj_list = self.order_by_dist(static_obj_list, waypoint, 45)

        #controlliamo le tre condizioni differenti:
        if self._direction == RoadOption.CHANGELANELEFT:
            static_obj_state, static_obj, distance = self._our_vehicle_obstacle_detected(static_obj_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            static_obj_state, static_obj, distance = self._our_vehicle_obstacle_detected(static_obj_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            static_obj_state, static_obj, distance = self._our_vehicle_obstacle_detected(static_obj_list, max(self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)
        return static_obj_state, static_obj, distance

    def car_following_manager(self, vehicle, distance, debug=False):
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

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def order_by_dist(self, object_list, waypoint, max_dist, check_not_our_vehicle=False):
        def dist(s): return s.get_location().distance(waypoint.transform.location)
        #qua ho messo dieci pero va scelto il giusto valore.
        #creiamo un dizionario in modo da ordinare a seconda delle distanze in ordine crescente
        if check_not_our_vehicle:
            static_obj_dict = {s:dist(s) for s in object_list if dist(s)<max_dist and s.id != self._vehicle.id}
        else:
            static_obj_dict = {s:dist(s) for s in object_list if dist(s)<max_dist}
        #otteniamo ora la lista corrispondente ordinata per valore
        ordered_dict = dict(sorted(static_obj_dict.items(),key=operator.itemgetter(1)))
        return list(ordered_dict.keys())

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
        if self._my_flag:
            input()
        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        bb_coords = self._vehicle.bounding_box.get_world_vertices(self._vehicle.get_transform())
        ego_vertexs_lane_id = [(self._map.get_waypoint(bb_coord)).lane_id for bb_coord in bb_coords]
        # vehicle_list = self._world.get_actors().filter("*vehicle*")

        # # usato x verificare logica del soprasso, da rimuovere
        # for actor in vehicle_list:
        #         if 'role_name' in actor.attributes and actor.attributes['special_type'] == 'emergency':
        #             print('auto police')
        #             continue
        #         if not('role_name' in actor.attributes and actor.attributes['role_name'] == 'hero' and actor.attributes['special_type'] != 'emergency'):
        #             actor.destroy() 


        # 1: Red lights and stops behavior, individua se esiste in un certo range un semaforo nello stato rosso. Memorizza l'attesa del semaforo, allo step successivo verifico QUELLO specifico semaforo e decido.
        if self.traffic_light_manager():
            return self.emergency_stop()
        # 1: Red lights and stops behavior, individua se esiste in un certo range un semaforo nello stato rosso. Memorizza l'attesa del semaforo, allo step successivo verifico QUELLO specifico semaforo e decido.
        if self.stop_sign_manager():
            return self.emergency_stop()
        # self._before_surpass_lane_id != ego_vehicle_wp.lane_id
        condToNotEnter = True
        if self._surpassing_biker and self.surpass_vehicle != None:
            condToNotEnter, v, d = self._our_vehicle_obstacle_detected(
                            [self.surpass_vehicle], max(
                                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        if self._surpassing_biker and self._before_surpass_lane_id != None and not condToNotEnter:
            print("end_surpassing(ego_vehicle_wp)")
            if self.end_surpassing(ego_vehicle_wp):
                return self._local_planner.run_step(debug=debug)

        # 2.1: Pedestrian avoidance behaviors, verifico se ci sono pedoni che possono influenzare la guida
        obstacle_dict = {"walker": list(self.pedestrian_avoid_manager(ego_vehicle_wp))}
        obstacle_dict["biker"] = list(self.bikers_avoid_manager(ego_vehicle_wp))
        obstacle_dict["vehicle"] = list(self.collision_and_car_avoid_manager(ego_vehicle_wp))
        obstacle_dict["static_obj"] = list(self.static_obstacle_avoid_manager(ego_vehicle_wp))

        # defiisce se eiste questo pedone, se esiste e si trova ad una distanza troppo vicina allora mi fermo!
        if obstacle_dict["walker"][0]:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            print("WALKER STATE la distanza dal pedone è: ", obstacle_dict["walker"][2])
            delta_v =  self._speed - get_speed(obstacle_dict["walker"][1])
            if delta_v < 0:
                delta_v = 0
            # Emergency brake if the car is very close.
            if obstacle_dict["walker"][2] < self._behavior.braking_distance/4 + delta_v * 0.2:
                return self.emergency_stop()
            elif obstacle_dict["walker"][2] < self._behavior.braking_distance + delta_v * 0.2:
                return self.controlled_stop(obstacle_dict["walker"][1], obstacle_dict["walker"][2])
        
        # police = self._world.get_actors().filter("*vehicle.dodge.charger_police*")
        # def dist(w): return w.get_location().distance(ego_vehicle_wp.transform.location)
        # police_list = [w for w in police if dist(w) < 30]

        # if police_list:
        #     vehicle_speed = get_speed(police_list[0])
        #     print(vehicle_speed)
        #     print("find auto police")
        #     if vehicle_speed == 0.0:
        #         #self._my_flag = True
        #         self._surpassing_police = True
        #         self._local_planner._change_line = "left"
        #         self._local_planner.set_speed(30) # da cambiare
        #         print('neeed to taigating')
        #         return self._local_planner.run_step(debug=debug)
        # else:
        #     if self._surpassing_police: #check altre auto ferme
        #         #self.help_sorpassing(ego_vehicle_wp,'right')
        #         #self.lane_change('right')
        #         print('torno a right')
        #         self._local_planner._change_line = "None"
        #         self._surpassing_police = False
        #         return self._local_planner.run_step(debug=debug)

        
        if self.obstacle_avoidance(obstacle_dict, ego_vehicle_wp, ego_vertexs_lane_id):
            return self._local_planner.run_step(debug=debug)

        
        if obstacle_dict["biker"][0]:
            biker_vehicle_loc = obstacle_dict["biker"][1].get_location()
            biker_vehicle_wp = self._map.get_waypoint(biker_vehicle_loc)
            print("biker check, biker_vehicle_wp.lane_id:  ", biker_vehicle_wp.lane_id, "self._before_surpass_lane_id: ", self._before_surpass_lane_id)
            #input()
            if biker_vehicle_wp.lane_id != self._before_surpass_lane_id:

                delta_v =  self._speed - get_speed(obstacle_dict["biker"][1])
                if delta_v < 0:
                    delta_v = 0
                print("BIKERS STATE la distanza dal veicolo è: ", obstacle_dict["biker"][2], "la sua lane è: ", biker_vehicle_wp.lane_id, "mentre la mia è: ", ego_vehicle_wp.lane_id, "la mia road option è:",  self._direction)
                #if self._surpassing_biker:
                    #input()
                # Emergency brake if the car is very close.
                if obstacle_dict["biker"][2] < self._behavior.braking_distance/4 + delta_v * 0.2:
                    return self.emergency_stop()
                elif obstacle_dict["biker"][2] < self._behavior.braking_distance + delta_v * 0.2:
                    return self.controlled_stop(obstacle_dict["biker"][1], obstacle_dict["biker"][2])

        # 2.2: Car following behaviors
        

        # stesso principio del pedone.
        if obstacle_dict["vehicle"][0]:
            vehicle_vehicle_loc = obstacle_dict["vehicle"][1].get_location()
            vehicle_vehicle_wp = self._map.get_waypoint(vehicle_vehicle_loc) 
            if vehicle_vehicle_wp.lane_id != self._before_surpass_lane_id:
                # Distance is computed from the center of the two cars,
                # we use bounding boxes to calculate the actual distance
                print("VEHICLE STATE la distanza dal veicolo è: ", obstacle_dict["vehicle"][2], "il veicolo è: ",obstacle_dict["vehicle"][1], "la sua lane è: ", vehicle_vehicle_wp.lane_id, "mentre la mia è: ", ego_vehicle_wp.lane_id, "la mia road option è:",  self._direction)
                #if self._surpassing_biker:
                    #input()
                delta_v =  self._speed - get_speed(obstacle_dict["vehicle"][1])
                if delta_v < 0:
                    delta_v = 0
                # Emergency brake if the car is very close.
                if obstacle_dict["vehicle"][2] < self._behavior.braking_distance/4 + delta_v * 0.2:
                    return self.emergency_stop()
                elif obstacle_dict["vehicle"][2] < self._behavior.braking_distance + delta_v * 0.2:
                    return self.controlled_stop(obstacle_dict["vehicle"][1], obstacle_dict["vehicle"][2])
                else:
                    return self.car_following_manager(obstacle_dict["vehicle"][1], obstacle_dict["vehicle"][2])

        #AGGIUNTA PER GESTIRE OSTACOLI STATICI SULLA STRADA
        if obstacle_dict["static_obj"][0]:
            static_obj_type = obstacle_dict["static_obj"][1].attributes.get('object_type')
            stop_cond = static_obj_type != "static.prop.dirtdebris01" or static_obj_type != "static.prop.dirtdebris02" or static_obj_type != "static.prop.dirtdebris03" or static_obj_type is not None
            if stop_cond:
                print("STATIC OBJ la distance dall'obj è: ", obstacle_dict["static_obj"][2])
                delta_v =  self._speed - get_speed(obstacle_dict["static_obj"][1])
                if delta_v < 0:
                    delta_v = 0
                # Emergency brake if the car is very close.
                if obstacle_dict["static_obj"][2] < self._behavior.braking_distance/4 + delta_v * 0.3:
                    return self.emergency_stop()
                elif obstacle_dict["static_obj"][2] < self._behavior.braking_distance + delta_v * 0.3:
                    return self.controlled_stop(obstacle_dict["static_obj"][1], obstacle_dict["static_obj"][2])

        # 3: Intersection behavior, consente di capire se siete in un incrocio, ma il comportamento è simile al normale, non ci sta una gestione apposita. La gestione degli incroci viene gestta in obj detection. Stesso comportamento normal behavor ma solo più lento.
        if self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            print("JUNCTION STATE")
            target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - 5])
            self._local_planner.set_speed(target_speed)
            return self._local_planner.run_step(debug=debug)

        # 4: Normal behavior, prende target speed, è una variabile che ti dice quanto manca a quello che ti serve. Il local planer contiene anche i controllori, quindi gli stiamo dicendo anche questo. Obj control contiene cose di carla sul dafarsi
        print("NORMAL BEHAVIOUR")
        target_speed = min([
            self._behavior.max_speed,
            self._speed_limit - self._behavior.speed_lim_dist])
        if self._surpassing_biker:
            self._local_planner.set_speed(80)
        else:
            self._local_planner.set_speed(target_speed)
        control = self._local_planner.run_step(debug=debug)

        print(self._local_planner._target_speed)

        return control

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False 
        # per le derapate a True
        return control

    def controlled_stop(self, vehicle, distance):
        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)
        control = self.emergency_stop()
        # Under safety time distance, slow down.
        if distance <= 2:
            return control
        elif self._speed < 15 and distance > 2:
            target_speed = distance * 3
            print("devo rallentare sono ancora lontano, l'obj vel è:",vehicle_speed," velocità: ", target_speed)
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step()
        elif ttc > 0.0:
            target_speed = max((ttc/self._behavior.safety_time) * self._speed, self._behavior.speed_decrease)
            print("devo rallentare, l'obj vel è:",vehicle_speed," andrò a velocità: ", target_speed)
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step()

        # per le derapate a True
        return control
    
    def start_surpassing(self, obj_to_s, ego_vehicle_wp, dir):
        self._surpassing_biker = True
        last_dir = self._direction
        if dir == "left":
            self._direction = RoadOption.CHANGELANELEFT
        elif dir == "right":
            # print("sorpasso a destra")
            # input()
            self._direction = RoadOption.CHANGELANERIGHT
        com_vehicle_state, com_vehicle, com_vehicle_distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
        
        if dir == "right":
            print("láuto che non mi fa sorpassare è: ", com_vehicle)
            input()
        if not com_vehicle_state or (com_vehicle_state and com_vehicle_distance>80):
            print('STO PER STARTARE IL SORPASSO, IL VEICOLO DISTA: ', com_vehicle_distance, "ed è: ", com_vehicle)
            
            #self._my_flag = True
            self._before_surpass_lane_id = ego_vehicle_wp.lane_id
            #self.help_sorpassing(ego_vehicle_wp,'left')
            self._local_planner._change_line = "shifting"
            self._local_planner.delta = self.meters_shifting(obj_to_s)
            print(self._local_planner.delta)
            self._local_planner.dir = dir
            self._local_planner.set_speed(80) # da cambiare
            self.surpass_vehicle = obj_to_s
            print('sto superando')
            self._direction = last_dir
            input()
            return True
        else:
            self._surpassing_biker = False
            self._direction = last_dir
            print('CI STA UN TIZIO CHE NON MI FA SORPASSARE LA DIST: ', com_vehicle_distance, "ed è: ", com_vehicle)
        return False

    def end_surpassing(self, ego_vehicle_wp):
        last_dir = self._direction
        if self._local_planner._change_line=="left":
            self._direction = RoadOption.CHANGELANERIGHT
        elif self._local_planner._change_line=="right":        
            self._direction = RoadOption.CHANGELANELEFT
        com_vehicle_state, com_vehicle, com_vehicle_distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
        self._direction = last_dir
        if self._local_planner._change_line != "None":
            print("non mi rientrare com_vehicle: ",com_vehicle)
            input()
        if not com_vehicle_state:
            print('STO PER RIENTRARE IN CORSIA')
            #input()
            self.surpass_vehicle = None
            self._local_planner._change_line = "None"
            self._local_planner.delta = 0
            self._local_planner.dir = "left"
            self._surpassing_biker = False
            self._before_surpass_lane_id = None
            return True
        return False
            
    def obstacle_avoidance(self, obj_dict, waypoint, ego_vertexs_lane_id):
        # logica per cominciare il sorpasso 
        if obj_dict["biker"][0] and obj_dict["biker"][2]<10 and get_speed(obj_dict["biker"][1]) <= 20 and not self._surpassing_biker:
            if self.start_surpassing(obj_dict["biker"][1], waypoint, "left"):
                input()
                return True
        valori = []
        for valore in obj_dict.values():
            if valore[0]:
                valori.append(valore[1])
        ordered_objs = self.order_by_dist(valori, waypoint, 45, True)
        if len(ordered_objs) > 0:
            print("state 1")
            #input()
            bb_coords = ordered_objs[0].bounding_box.get_world_vertices(ordered_objs[0].get_transform())
            obj_vertexs_lane_id = [(self._map.get_waypoint(bb_coord)).lane_id for bb_coord in bb_coords]
            int_list = list(set(obj_vertexs_lane_id) & set(ego_vertexs_lane_id))
            not_my_lane_list = list(set(obj_vertexs_lane_id) - set(int_list))
            #condizione per verificare che quest'oggetto invada parzialmente la mia lane (da superare)
            if len(int_list)>0 and len(not_my_lane_list)> 0:
                print("state 2")
                # input()
                print("len(int_list): ", len(int_list), "len(not_my_lane_list): ", len(not_my_lane_list))
                if not_my_lane_list[0] > waypoint.lane_id:
                    print("not_my_lane_list[0]: ", not_my_lane_list[0], "waypoint.lane_id: ", waypoint.lane_id)
                    if self.start_surpassing(ordered_objs[0], waypoint, "left"):
                        print("state 3")
                        input()
                        return True
                else:
                    if self.start_surpassing(ordered_objs[0], waypoint, "right"): 
                        print("state 3.1")
                        input()
                        return True
        return False

    def meters_shifting(self, target_vehicle):
        target_transform = target_vehicle.get_transform()
        target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)
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
        print("corresponding_t_wpt: ",corresponding_t_wpt,"target_transform.location: ",target_transform.location,"target_wpt: ", target_wpt,"np.linalg.norm(diff_lane): ", np.linalg.norm(diff_lane))
        #ottenere tutti i vertici del target vehicle target
        # # Otteniamo i vertici del bounding box in coordinate locali
        target_bb = self.get_bounding_box_corners(target_vehicle)
        #target_bb = target_vehicle.bounding_box.get_world_vertices(target_vehicle.get_transform())
        print("target_bb[0]:", target_bb[0])
        distances = {} 
        draw_bbox(self._world, target_vehicle)
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
        print("distances: ",distances)
        max_vertex = min(distances, key = lambda k: distances[k])
        print("max_vertex: ",max_vertex)
        input()
        #compute the distance between max_vertex and the target_wpt
        diff_points = np.array([
            target_transform.location.x - max_vertex[0],
            target_transform.location.y - max_vertex[1],
            target_transform.location.z - max_vertex[2]])
        dot_product = np.dot(diff_points,diff_lane)
        print("diff_points: ",diff_points, "max_vertex: ",max_vertex)
        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
        how_much_move = dot_product - np.linalg.norm(diff_lane)/2
        print("how_much_move: ",how_much_move, "dot_product: ", dot_product,"np.linalg.norm(diff_lane)/2:",np.linalg.norm(diff_lane)/2 )
        input()
        if ego_vehicle_wp.lane_id != target_wpt.lane_id:
            return how_much_move
        else:
            return np.linalg.norm(diff_lane) - how_much_move
        


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
        print("yaw: ", yaw)
        # Calcola le dimensioni del bounding box (divise per due)
        extent_x = bbox.extent.x if bbox.extent.x != 0 else 1
        extent_y = bbox.extent.y if bbox.extent.y != 0 else 1
        extent_z = bbox.extent.z if bbox.extent.z != 0 else 1
        print("extent_x: ",extent_x,"extent_y: ",extent_y,"extent_z: ",extent_z)
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
        print("rotation: ", rotation)
        transformed_box = []
        for point in bounding_box:
            transformed_point = np.dot(rotation, point)
            transformed_point += np.array([location.x, location.y, location.z])
            transformed_box.append(transformed_point)

        return transformed_box