from other.clientActionRobot import ClientActionRobot


def run():
    """function to run the agent

    """
    config = ar.configure()
    state = ar.get_state()
    load_lvl = ar.load_level(1)
    ar.do_screen_shot()
    print(load_lvl)


# Usage example
team_id = 28888  # Replace with your actual team ID
server_address = '127.0.0.1'  # Replace with the actual server address
server_port = 2004  # Replace with the actual server port
ar = ClientActionRobot(server_address, server_port, team_id)
run()
