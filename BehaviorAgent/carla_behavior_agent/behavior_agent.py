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

from misc import get_speed, positive, is_within_distance, compute_distance,draw_bbox

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
        affected, _,dist_from_stop = self._affected_by_stop_sign(stop_list)
        return affected,dist_from_stop

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
        vehicle_list = self.order_by_dist(vehicle_list, waypoint, 45, True)

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
        
        
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(w): return w.get_location().distance(ego_vehicle_wp.transform.location)
        # vehicle_list_red = [w for w in vehicle_list if dist(w) < 30]

        # for act in vehicle_list_red:
        #     draw_bbox(self._world, act)


        # for actor_snapshot in vehicle_list_red:
        #     draw_bbox(self._world, actor_snapshot)

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
        affected_by_stop,dist_from_stop = self.stop_sign_manager()
        if affected_by_stop:
            print("sto in stop_sign")
            return self.controlled_stop(distance=dist_from_stop)
        

        # self._before_surpass_lane_id != ego_vehicle_wp.lane_id
        condToEnter = len([x for x in ego_vertexs_lane_id if x != self._before_surpass_lane_id]) > 0
        condForCheck = ego_vehicle_wp.lane_id != self._before_surpass_lane_id
        if self._surpassing_biker and self._before_surpass_lane_id != None and condToEnter:
            print("end_surpassing(ego_vehicle_wp)")
            if self.end_surpassing(ego_vehicle_wp, condForCheck):
                return self._local_planner.run_step(debug=debug)

        # 2.1: Pedestrian avoidance behaviors, verifico se ci sono pedoni che possono influenzare la guida
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)
        # defiisce se eiste questo pedone, se esiste e si trova ad una distanza troppo vicina allora mi fermo!
        if walker_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            print("WALKER STATE la distanza dal pedone è: ", w_distance)
            delta_v =  self._speed - get_speed(walker)
            if delta_v < 0:
                delta_v = 0
            # Emergency brake if the car is very close.
            if w_distance < self._behavior.braking_distance/4 + delta_v * 0.2:
                return self.emergency_stop()
            elif w_distance < self._behavior.braking_distance + delta_v * 0.2:
                return self.controlled_stop(walker, w_distance)
        
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

        biker_state, biker, b_distance = self.bikers_avoid_manager(ego_vehicle_wp)

        # logica per cominciare il sorpasso 
        if biker_state and b_distance<10 and get_speed(biker) <= 20 and not self._surpassing_biker:
            if self.start_surpassing(biker, ego_vehicle_wp):
                return self._local_planner.run_step(debug=debug)
        
        if biker_state:
            biker_vehicle_loc = biker.get_location()
            biker_vehicle_wp = self._map.get_waypoint(biker_vehicle_loc)
            print("biker check, biker_vehicle_wp.lane_id:  ", biker_vehicle_wp.lane_id, "self._before_surpass_lane_id: ", self._before_surpass_lane_id)
            #input()
            if biker_vehicle_wp.lane_id != self._before_surpass_lane_id:

                delta_v =  self._speed - get_speed(biker)
                if delta_v < 0:
                    delta_v = 0
                print("BIKERS STATE la distanza dal veicolo è: ", b_distance, "la sua lane è: ", biker_vehicle_wp.lane_id, "mentre la mia è: ", ego_vehicle_wp.lane_id, "la mia road option è:",  self._direction)
                #if self._surpassing_biker:
                    #input()
                # Emergency brake if the car is very close.
                if b_distance < self._behavior.braking_distance/4 + delta_v * 0.2:
                    return self.emergency_stop()
                elif b_distance < self._behavior.braking_distance + delta_v * 0.2:
                    return self.controlled_stop(biker, b_distance)

        # 2.2: Car following behaviors
        vehicle_state, vehicle, v_distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)

        # stesso principio del pedone.
        if vehicle_state:
            vehicle_vehicle_loc = vehicle.get_location()
            vehicle_vehicle_wp = self._map.get_waypoint(vehicle_vehicle_loc) 
            if vehicle_vehicle_wp.lane_id != self._before_surpass_lane_id:
                # Distance is computed from the center of the two cars,
                # we use bounding boxes to calculate the actual distance
                print("VEHICLE STATE la distanza dal veicolo è: ", v_distance, "il veicolo è: ",vehicle, "la sua lane è: ", vehicle_vehicle_wp.lane_id, "mentre la mia è: ", ego_vehicle_wp.lane_id, "la mia road option è:",  self._direction)
                #if self._surpassing_biker:
                    #input()
                delta_v =  self._speed - get_speed(vehicle)
                if delta_v < 0:
                    delta_v = 0
                # Emergency brake if the car is very close.
                if v_distance < self._behavior.braking_distance/4 + delta_v * 0.2:
                    return self.emergency_stop()
                elif v_distance < self._behavior.braking_distance + delta_v * 0.2:
                    return self.controlled_stop(vehicle, v_distance)
                else:
                    return self.car_following_manager(vehicle, v_distance)

        #AGGIUNTA PER GESTIRE OSTACOLI STATICI SULLA STRADA
        static_obj_state, static_obj, obs_distance = self.static_obstacle_avoid_manager(ego_vehicle_wp)
        if static_obj_state:
            #static_obj_type = static_obj.attributes.get('object_type')
            #stop_cond = static_obj_type != "static.prop.dirtdebris01" or static_obj_type != "static.prop.dirtdebris02" or static_obj_type != "static.prop.dirtdebris03" or static_obj_type is not None
            stop_cond = static_obj.bounding_box.extent.z >= 0.25
            print("altezza dell'oggetto: ", static_obj.bounding_box.extent.z)
            print("oggetto: ", static_obj)
            if stop_cond:
                print("oggetto più alto di mezzo metro, mi fermo")
                print("STATIC OBJ la distance dall'obj è: ", obs_distance)
                delta_v =  self._speed - get_speed(static_obj)
                if delta_v < 0:
                    delta_v = 0
                # Emergency brake if the car is very close.
                if obs_distance < self._behavior.braking_distance/4 + delta_v * 0.3:
                    return self.emergency_stop()
                elif obs_distance < self._behavior.braking_distance + delta_v * 0.3:
                    return self.controlled_stop(static_obj, obs_distance)

        # 3: Intersection behavior, consente di capire se siete in un incrocio, ma il comportamento è simile al normale, non ci sta una gestione apposita. La gestione degli incroci viene gestta in obj detection. Stesso comportamento normal behavor ma solo più lento.
        if self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            vehicle_state, vehicle, v_distance = self.gestione_incrocio(ego_vehicle_wp)
        # stesso principio del pedone.
        if vehicle_state:
            print('Junction State:')
            vehicle_vehicle_loc = vehicle.get_location()
            vehicle_vehicle_wp = self._map.get_waypoint(vehicle_vehicle_loc) 
            if vehicle_vehicle_wp.lane_id != self._before_surpass_lane_id:
                delta_v =  self._speed - get_speed(vehicle)
                if delta_v < 0:
                    delta_v = 0
                # Emergency brake if the car is very close.
                if v_distance < self._behavior.braking_distance/4 + delta_v * 0.2:
                    return self.emergency_stop()
                elif v_distance < self._behavior.braking_distance + delta_v * 0.2:
                    return self.controlled_stop(vehicle, v_distance)
                else:
                    return self.car_following_manager(vehicle, v_distance)

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
        control = self._local_planner.run_step_only_lateral()
        # per le derapate a True
        return control

    def controlled_stop(self, vehicle=None, distance=0.0):
        vehicle_speed = 0.0
        if vehicle != None:
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
    
    def start_surpassing(self, obj_to_s, ego_vehicle_wp):
        last_dir = self._direction
        self._direction = RoadOption.CHANGELANELEFT
        com_vehicle_state, com_vehicle, com_vehicle_distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
        self._direction = last_dir
        if not com_vehicle_state or (com_vehicle_state and com_vehicle_distance>80):
            print('STO PER STARTARE IL SORPASSO, IL VEICOLO DISTA: ', com_vehicle_distance, "ed è: ", com_vehicle)
            #input()
            #self._my_flag = True
            self._surpassing_biker = True
            self._before_surpass_lane_id = ego_vehicle_wp.lane_id
            #self.help_sorpassing(ego_vehicle_wp,'left')
            self._local_planner._change_line = "center"
            self._local_planner.set_speed(80) # da cambiare
            print('sto superando')
            return True
        elif com_vehicle_state:
            print('CI STA UN TIZIO CHE NON MI FA SORPASSARE LA DIST: ', com_vehicle_distance, "ed è: ", com_vehicle)

        return False

    def end_surpassing(self, ego_vehicle_wp, check_if_left):
        last_dir = self._direction
        if check_if_left:
            self._direction = RoadOption.CHANGELANERIGHT
        com_vehicle_state, com_vehicle, com_vehicle_distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
        self._direction = last_dir
        if not com_vehicle_state:
            print('STO PER RIENTRARE IN CORSIA')
            #input()
            self._local_planner._change_line = "None"
            self._surpassing_biker = False
            self._before_surpass_lane_id = None
            return True
        return False
            

        
        # if  and not self._surpassing_biker:

        #         if com_vehicle_state:
        #             print("allora vorrei superare il ciclista, ma ci sta una macchina la sua distanza è: ", com_vehicle_distance, "veicolo: ",com_vehicle)
                
        #             print("lo stronzo che butto giù si trova a:", com_vehicle_distance)
        #             b_speed = get_speed(biker)
        #             print(b_speed)
        #             print("find ciclista")
        #             if :
                        
        #         else:
        #             print("il veicolo è troppo vicino la distanza: ", com_vehicle_distance, "mentre il veicolo è: ", com_vehicle)
        #             input()
                
        # elif self._surpassing_biker: 
        #     #check altre auto ferme
        #     #self.help_sorpassing(ego_vehicle_wp,'right')
        #     #self.lane_change('right')
        #     print("CAZZZOZOOZOOZOZOZOZOZOZOOZOZOZOZOZOZOZOZOZOZOZOZOZOZ")
        #     input()
            
        #     if com_vehicle_state:
        #         self._local_planner.set_speed(50) # da cambiare
        #     else:
        #         print('torno a right')
        #         self._local_planner._change_line = "right"
        #         self._surpassing_biker = False
        #         return self._local_planner.run_step(debug=debug)
        #     self._direction = RoadOption.LANEFOLLOW