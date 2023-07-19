import time

from other.clientActionRobot import ClientActionRobot


def run():
    """function to run the agent

    """
    config = ar.configure()
    state = ar.get_state()
    load_lvl = ar.load_level(1)
    ar.do_screen_shot()
    my_score = ar.get_my_score()
    resp = ar.restart_level()


# Usage example
team_id = 28888
server_address = '127.0.0.1'
server_port = 2004
ar = ClientActionRobot(server_address, server_port, team_id)
run()
