from other.clientActionRobot import ClientActionRobot


def run():
    """function to run the agent

    """
    config = ar.configure()  # configures message to the server
    state = ar.get_state()  # gets game state, make sure in level selection.
    load_lvl = ar.load_level(1)  # loads a specific level
    ar.do_screen_shot()  # takes a screenshot
    my_score = ar.get_my_score()  # gets score for all levels
    # resp = ar.restart_level()
    # zoomed_in = ar.fully_zoom_in()
    # zoomed_out = ar.fully_zoom_out()
    for i in range(3):  # sequence of shots
        # makeshot = ar.c_shoot(196, 326, 40, 0, 5, 5) # shot in cartesian random values
        makeshot = ar.p_shoot(196, 326, 30, 7000, 5, 5)  # shot in polar random values
        # makeshot = ar.c_fast_shoot(196, 326, 40, 0, 5, 15) # fast shot in cartesian random values
        # makeshot = ar.p_fast_shoot(196, 326, 30, 7000, 5, 5) # fast shot in polar random values


# Usage example
team_id = 28888
server_address = '127.0.0.1'
server_port = 2004
ar = ClientActionRobot(server_address, server_port, team_id)
run()
