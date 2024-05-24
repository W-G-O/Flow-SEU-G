class ExpTravelTimeRouter(BaseRouter):

    def choose_route(self, env):
        network = env.network
        edges = network.edges
        nodes = network.nodes
        vehicle = env.k.vehicle
        veh_id = self.veh_id
        initial_route = vehicle.get_route(veh_id)
        end_edge = initial_route[-1]

        light = network.traffic_lights

        # reroute in 600 steps
        if env.step_counter % 600 != 0:  # re-route every 600 simulation steps
            return None

        if vehicle.get_edge(veh_id)[0:3] == ':ce':  # if veh in intersection, reject re-route
            return None

        start_edge = vehicle.get_edge(veh_id)
        velocity = max(5, vehicle.get_speed(veh_id))

        def adding(next_edges_list,edge_name):  # adding the legal neighbour of current edge
            i, j = int(edge_name[-3]), int(edge_name[-1])
            edge_direction = edge_name[:-3]
            I, J = int(end_edge[-3]), int(end_edge[-1])

            islegal = False
            if edge_direction in ["left", "right"]:
                islegal = (i <= I and j < J)
            if edge_direction in ["top", "bot"]:
                islegal = (i <= I and j <= J)

            if islegal:
                next_edges_list.append(edge_name)

        def _get_neigh_edges(current_edge): # inner tools for getting next edges ()
            next_edges = []
            edge_direction = current_edge[:-3]  # top/bot/left/right
            i = int(current_edge[-3])
            j = int(current_edge[-1])  # get the index of current edge
            if edge_direction == "bot":
                adding(next_edges,"right" + str(i + 1) + "_" + str(j))
                adding(next_edges,edge_direction + str(i) + "_" + str(j + 1))

            if edge_direction == "right":
                adding(next_edges, edge_direction + str(i + 1) + "_" + str(j))
                adding(next_edges, "bot" + str(i) + "_" + str(j + 1))
            return next_edges

        def _getmin(rts_list): #E[Travel_Time] = sum[length_of_edge, edge in route]/velocity + num_nodes* [C-G]

            argmin_r = rts_list[0]
            mintt = float('inf')
            for r in rts_list:
                tt = 0
                for edge_name in r:
                    edge_length = None
                    tonode = None

                    for _ in edges:
                        if _["id"] == edge_name:
                            edge_length = float(_["length"])
                            tonode = _["to"]
                            break

                    tt += edge_length / velocity
                    if tonode in light.get_properties():
                        try:
                            tt += 3 * float(light.get_properties()[tonode]["phases"][0]["duration"])
                        except:
                            tt += 45

                if tt <= mintt:
                    argmin_r, mintt = r, tt
            return argmin_r

        # using double stack to get all routes
        routes = []
        s0 = [start_edge]
        s1 = [_get_neigh_edges(start_edge)]  # step1 build stack

        while s0 != []:
            s1_top = s1[-1]

            if s1_top != []: # step2 keep build stacks
                edge_to_s0 = s1_top.pop(0)
                edgelist_to_s1 = _get_neigh_edges(edge_to_s0)

                for edge in edgelist_to_s1:
                    if edge in s0: edgelist_to_s1.remove(edge)

                s0.append(edge_to_s0)
                s1.append(edgelist_to_s1)

            else:  # step3 cutdown stacks
                s0.pop()
                s1.pop()
                continue

            if s0[-1] == end_edge: # step4 get res
                res = copy.deepcopy(s0)
                routes.append(res)
                del res
                s0.pop()
                s1.pop()

        # return the r = argmint(r)
        if not routes:
            return None
        return _getmin(routes)
