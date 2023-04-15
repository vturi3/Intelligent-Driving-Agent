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

from misc import get_speed, positive, is_within_distance, compute_distance

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

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 4.5

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
        #tale funzione si trova nel basic agent.
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """
        #Tenta di gestire la possibilità di effettuare dei movimenti tipo cambi di corsia a destra o a sinistra e cerca di tenere in considerazione i veicoli che vengono da dietro
        # Osservando il comportamento è chiaro che non funziona benissimo.
        #le funzioni di base che vengono usate sono sempre le stesse, ovvero vehicle_obtacle_detected, pero con valori diversi degli angoli.
        # LaneChanginng
        # questo dovrebbe servire a capire se si puo girare a destra o a sinistra
        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        # questo dovrebbe servire ad ottenere il corrispettivo waypoint pero nella lane adiacente, se ne esiste una, cioè l'obiettivo per la lane change, sarebbe
        #poi muoversi su quel waypoint.
        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        # se esiste veicolo che da fastidio quindi in possibile collisione sulla base delle intenzioni che abbiamo, quindi si stabiliscono degli angoli che definiscono il raggio di
        # azione per cui se esiste questo veicolo che da fastidio e la nostra velocità è minore della velocità di questo veicolo che sta sopraggiungendo, quindi non riesco a fare il cambio
        # In generale cambiano gli angoli che vengono passati alla funzione x la detection.
        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            #si va a valutare se stiamo girando a destra o a sinistra
            # questo quello che dovrebbe fare è verificare la possibilità di andare a destra o da entrambe le parti, considerare che la direzione dei waypoints sulle
            # due lane adiacenti sia la medesima e vedere se è di tipo driving la lane che vorremmo invadere.
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                # Verifico anche dopo se sto x fare danni. 
                # Questo ci dice, se nn sopraggiungono vehicle e enlla traiettoria nn ho vehicle su cui impatto, faccio manovra cambiando il path.
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
                    # consente di calcolare percorso da dove ci troviamo a dove voglaimo andare e viene dato al local planner, viene calcoalto tramite il mission planner,
                    # è come una modifica della traiettoria originale.
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
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

        # logica è uguale a quella del pedone, quindi faccio un filtraggio sui veicoli e rispetto al punto in cui mi trovo. 
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=30)
            # In questa situazione si tiene conto anche del tailgating, ovvero di fatto situazioni in cui non c'è stata detection di ostacolo o comunque non problematico, 
            # La road option è di tipo LAENFOLLOW quindi sto andando in una corsia e dritto, quindi non devo fare per esempio cambi di corsia o devo girare e non sto in un'intersezione
            # e la velocità è maggiore di una certa quantità e non sono gia in tailfating allora effettuo il tailgating che gestisce sostanzialmente l'immissione.
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
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 10]

        # bisogna capire ora che azione intraprendere, quindi come prima cosa bisogna capire se il nostro veicolo potrebbe essere in collisione con questo pedone,
        # oer esempio il pedone potrebbe essere ad una distanza piccola da noi, ma potrebbe essere già andato oltre. Quindi a seconda delle cose diverse che dobbiamo andare
        #a vedere valuteremo in modo diverso la funzione vehicle_obstacle_detected, che è una funzione che in realtà va bene per qualsiasi tipo di oggetto, non solo per i veicoli
        # perchè noi in ingresso passiamo una lista di oggetti. I parametri passati riguardano la distanza massima da rispettare da questi oggetti, e poi ci sono questioni sull'orientamento
        # La funzione restituisce se c'è un pedone che influenza la nostra guida e quindi si necessita di un emergency stop, chi è e a che distanza si trova. 
        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return walker_state, walker, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        # sto in una situazione normale, nn ho vehicle che mi danno fastidio, pedoni o altro.. 
        # Seguo la macchina e basta. Prendo la velocita del vehicle, vedo la differenza di velocita, calcolo TimeToCollision. 
        # Il comportamento è quello di ridurre la velocità cosi da matchare la sua, viene fatto mano mano. Cambio quindi la nuova velocità obiettivo.
        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        #Se il tempo per impattare il veicolo è maggiore di zero ed è minore del tempo di safety che viene settato a seconda del behavior, quindi a quanto tempo voglio 
        #considerare un veicolo. Quello che succede è andare ad effettuare un rallentamento, cioe si riduce la nostra velocità sulla base del veicolo che stiamo seguendo. 
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            #comunichiamo al local planner quale è la nuova velocità obbiettivo e quindi il local planner attuerà il controllo sulla base di questa velocità
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        #Abbiamo un comportamento simile a quello precedente la riduzione dellla velocità è fatta in modo diverso, nel senso che qui non siamo cosi tanto vicini e quindi
        #andiamo ad adattare la nostra velocità a quella del veicolo.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        # Se non vale nessuno dei comportamenti precedenti, allora continuiamo in una situazione di normal behavior, cioe seguiamo la velocità target che ci siamo dati.
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def run_step(self, debug=False):
        """
        è il metodo che viene chiamato ad ogni tiemstep. Possiamo immaginarlo come a dire prendo informazioni dall'ambiente ed eseguo il behavior planner, 
        che può essere rappresentato come una macchina a stati (anche se è molto complesso in questo caso). Di base gestisce in un certo ordine tutte le cose viste
        nella descrizione, ovvero auto, pedoni, semafori.

        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl        
        """

        print("sono in run_step")
        self._update_information()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())
        print(ego_wpt.lane_id)
        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        #acquisice informazioni sul veicolo
        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # 1: Red lights and stops behavior
        # individua se esiste in un certo range intorno al proprio veicolo un semaforo e se questo semaforo è rosso. Quindi se è rosso si ferma e si salva che è in attesa di un semaforo
        # quindi al prossimo step tale semaforo potrebbe diventare verde, quindi significa che questo controllo non deve farlo piu tra tutti i possibili semafori nell'intorno
        # ma semplicemente su quello che mi ero salvato allo step precedente. Nel caso in cui ci sono semafori rossi per cui devo fermarmi, vado nello stato di emergency stop.
        if self.traffic_light_manager():
            print("traffic_light")
            return self.emergency_stop()

        # 2.1: Pedestrian avoidance behaviors
        # pedestrian_avoid_manager va a valutare se ci sono dei pedoni nel mio raggio di azione che possono andare ad influenzare la mia guida
        # per esempio se c'è un pedone che sta attraversando.
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)
        # walker_state definisce appunto se esiste questo pedone che impatta con il mio veicolo. Se questo pedone esiste e si trova ad una distanza troppo vicina
        # allora si va in emergency stop.
        if walker_state:
            print("walker_state")
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()

        # 2.2: Car following behaviors
        # Se non ci sono pedoni che impattano la mia guida si passa alla valutazione dei veicoli, cioè si valuta se esite una macchina che puo essere in collisione con 
        # il nostro veicolo, cioè entro in certo range nella nostra traiettoria. Le cose che vengono fatte in questo caso specifico sono due, ovvero se è molto vicino
        # allora quello che si fa è fermarsi, senon dovesse essere molto vicino si potrebbe valutare di seguirlo, per esempio un veicolo che sta davanti a noi nella stessa
        # corsia.
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
        # stesso principio del pedone.
        if vehicle_state:
            print("vehicle_state")
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()
            else:
                control = self.car_following_manager(vehicle, distance)

        # 3: Intersection behavior
        # questo codice in realtà è una fregatura perchè il comportamento descritto consente di capire se siamo ad un incrocio ma ciò che viene fatto
        # non è diverso da normal behavior, quindi non c'è un modulo specifico per la gestione delle intersezioni che invece principalmente viene gestito 
        # dal modulo di collision avoidance della macchina, quindi il modulo precednete in parte ingloba anche questa caratteristica.
        # Stesso comportamento normal behavor ma solo più lento.
        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            print("intersection_behavior")
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - 5])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # 4: Normal behavior, prende target speed, è una variabile che ti dice quanto manca a quello che ti serve. 
        # qui viene chiamato il run step del local planner,
        # che è una cosa che non c'era eni moduli precedenti, quindi stiamo adattando alla velocità del local planner che in questo setup contiene anche i controllori, 
        # quindi stiamo passando queste info.
        # L'oggetto control è quello che in carla contiene throttle brake e steer.
        else:
            print("normal_behavior")
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        #rappresenta lo stato in cui si da la massima frenata e la massima accelerazione alla macchina.

        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False 
        # per le derapate a True
        return control
