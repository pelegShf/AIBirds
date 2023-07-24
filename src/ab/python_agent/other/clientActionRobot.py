import socket
import struct
import sys

from PIL import Image

from external.ClientMessageEncoder import configure, get_state, load_level, do_screen_shot, get_my_score, restart_level, \
    fully_zoom_in, fully_zoom_out, c_shoot, p_shoot, c_fast_shoot, p_fast_shoot, get_best_scores

from utils import decode_byte_to_int


class ClientActionRobot:
    """This class implements all the interaction a client can do with the server.

    :param server_address: server address. preset: 127.0.0.1
    :type server_address: str,optional
    :param server_port: server port. preset: 2004
    :type server_port: int,optional
    :param team_id: server port. preset: 28888
    :type server_port: int,optional

    """

    def __init__(self, server_address="127.0.0.1", server_port=2004, team_id=28888):
        """Constructor method
        """
        self.server_address = server_address
        self.server_port = server_port
        self.team_id = team_id

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Connect to the server
            self.client_socket.connect((server_address, server_port))
            print('Connected')
        except socket.error as exc:
            print("Caught exception socket.error : %s" % exc)

    def configure(self):
        """Sends configure message to the server

        The function connects the agent to the server by sending a config message.
        This needs to be done once at the beginning of the session.

        :return: config data [Round Info, Time_limit , Number of Levels] each 4 bytes
        :rtype: list
        """
        try:
            message = configure(self.team_id)
            self.client_socket.sendall(message)

            response = self.client_socket.recv(4)
            config_data = decode_byte_to_int(response)  # [Round Info, Time_limit , Number of Levels]

            return config_data
        except socket.error as e:
            print("Connection error: %s" % e)
            sys.exit(1)

    def get_state(self):
        """Sends request for game state to the server

            the function gets from the server the game state
            [0]: UNKNOWN | [1] : MAIN_MENU | [2]: EPISODE_MENU | [3]: LEVEL_SELECTION
            [4]: LOADING | [5]: PLAYING | [6]: WON | [7]: LOST

            :return: int representing the state
            :rtype: int
              """
        try:
            message = get_state()
            self.client_socket.sendall(message)

            response = self.client_socket.recv(1)
            state_data = decode_byte_to_int(response)  # [ordinal]

            return state_data

        except socket.error as e:
            print("Connection error: %s" % e)
            sys.exit(1)

    def load_level(self, level=0):
        """Load a level and send it to the server socket.

            This function loads a specific level and sends it to the server socket for processing.
            It uses the `load_level` function to retrieve the level message and sends it using the client socket's `sendall` method.
            After sending the message, it waits for a response from the client using the `recv` method and decodes it to an integer.

        :param level: The level to load. Defaults to 0.
        :return: The response received from the client, represented as an integer [1/0]
        :rtype int
        """
        try:
            message = load_level(level)
            self.client_socket.sendall(message)

            response = self.client_socket.recv(1)
            response = decode_byte_to_int(response)  # [1/0]

            return response
        except socket.error as e:
            print("Connection error: %s" % e)
            sys.exit(1)

    def restart_level(self):
        """restart current level.
            :return: 1/0 - 1: the level has been restarted | 0: The server cannot restart the level
            :rtype int
            """
        try:
            message = restart_level()
            self.client_socket.sendall(message)

            response = self.client_socket.recv(1)
            response = decode_byte_to_int(response)  # [1/0]

            return response
        except socket.error as e:
            print("Connection error: %s" % e)
            sys.exit(1)

    def do_screen_shot(self, image_fname='screenshot'):
        """Capture a screenshot and save it to an image file.

            Sends a screenshot request to the client socket and saves the received screenshot data
            to an image file specified by `image_fname`.

        :param image_fname: The filename to save the screenshot as. Defaults to 'screenshot'.
        :type image_fname: (str, optional)
        :return: image dir
        :rtype string
        """
        try:
            message = do_screen_shot()
            self.client_socket.sendall(message)

            width = self.client_socket.recv(4)
            height = self.client_socket.recv(4)

            width = struct.unpack('>I', bytes(width))[0]
            height = struct.unpack('>I', bytes(height))[0]

            total_bytes = width * height * 3
            remaining_bytes = total_bytes
            received_data = b''
            while remaining_bytes > 0:
                chunk_size = min(4096, remaining_bytes)
                data = self.client_socket.recv(chunk_size)
                if not data:
                    break
                received_data += data
                remaining_bytes -= len(data)
            img = Image.frombytes('RGB', (width, height), received_data)
            img_dir = f'{image_fname}.jpg'
            img.save(img_dir)
            return img_dir
        except socket.error as e:
            print("Connection error: %s" % e)
            sys.exit(1)

    def get_my_score(self):
        """returns agent score for all levels.

            return a fixed length (4 * 21) bytes array with every four slots indicates a best score of the corresponding level. Scores of unsolved and unavailable levels are zero.

        :return: (4 * 21) bytes array
        :rtype bytes array
        """
        message = get_my_score()
        self.client_socket.sendall(message)
        response = self.client_socket.recv(84)
        scores = []
        for i in range(0, len(response), 4):
            score_bytes = response[i:i + 4]
            score = int.from_bytes(score_bytes, byteorder='big')
            scores.append(score)
        return scores

    def get_best_scores(self):
        """returns best scores of all agents for all levels.

            The server will return a fixed length (4 * 21) bytes array with every four slots indicates a best score of the corresponding level. Scores of unsolved and unavailable levels are zero.
            1: 1st qualification round: this request will return an array of the best scores the naive agent achieved on the qualification levels.
            2: 2nd qualification round: this request will return an array of the best scores achieved in the 1st qualification round on the qualification levels.
            3: Group round (group of four): this request will return an array of the current best scores of their own group. The scores change whenever a new high score is obtained.
            4: Knock-out round (group of two): this request will return an array of the current best scores for the two participating agents. The scores change whenever a new high score is obtained.
            :return: (4 * 21) bytes array
            :rtype bytes array
            """
        message = get_best_scores()
        self.client_socket.sendall(message)
        response = self.client_socket.recv(84)
        scores = []
        for i in range(0, len(response), 4):
            score_bytes = response[i:i + 4]
            score = int.from_bytes(score_bytes, byteorder='big')
            scores.append(score)
        return scores

    def fully_zoom_in(self):
        """The server will fully zoom in on receiving this message
            :return: 1/0 - 1: The server has fully zoomed in the current game | 0: The server cannot zoom in
            :rtype int
            """
        try:
            message = fully_zoom_in()
            self.client_socket.sendall(message)

            response = self.client_socket.recv(1)
            response = decode_byte_to_int(response)  # [1/0]

            return response
        except socket.error as e:
            print("Connection error: %s" % e)
            sys.exit(1)

    def fully_zoom_out(self):
        """The server will fully zoom out on receiving this message
            :return: 1/0 - 1: The server has fully zoomed out the current game | 0: The server cannot zoom in
            :rtype int
            """
        try:
            message = fully_zoom_out()
            self.client_socket.sendall(message)

            response = self.client_socket.recv(1)
            response = decode_byte_to_int(response)  # [1/0]

            return response
        except socket.error as e:
            print("Connection error: %s" % e)
            sys.exit(1)

    def c_shoot(self, focus_x, focus_y, dx, dy, t1, t2):
        """Makes a shot in cartesian coordinates

            :param focus_x: the x coordinate of the focus point
            :param focus_y: the y coordinate of the focus point
            :param dx: the x coordinate of the release point minus focus_x
            :param dy: the y coordinate of the release point minus focus_y
            :param t1: the release time
            :param t2: the gap between the release time and the tap time
            :return: 1/0 - 1: the shot has been made | 0: the shot has been rejected
            :rtype int
        """
        try:
            message = c_shoot(focus_x, focus_y, dx, dy, t1, t2)
            self.client_socket.sendall(message)

            response = self.client_socket.recv(1)
            response = decode_byte_to_int(response)  # [1/0]

            return response
        except socket.error as e:
            print("Connection error: %s" % e)
            sys.exit(1)


    def c_fast_shoot(self, focus_x, focus_y, dx, dy, t1, t2):
        """Makes a shot in cartesian coordinates

            :param focus_x: the x coordinate of the focus point
            :param focus_y: the y coordinate of the focus point
            :param dx: the x coordinate of the release point minus focus_x
            :param dy: the y coordinate of the release point minus focus_y
            :param t1: the release time
            :param t2: the gap between the release time and the tap time
            :return: 1/0 - 1: the shot has been made | 0: the shot has been rejected
            :rtype int
        """
        try:
            message = c_fast_shoot(focus_x, focus_y, dx, dy, t1, t2)
            self.client_socket.sendall(message)

            response = self.client_socket.recv(1)
            response = decode_byte_to_int(response)  # [1/0]

            return response
        except socket.error as e:
            print("Connection error: %s" % e)
            sys.exit(1)

    def p_shoot(self, focus_x, focus_y, r, theta, t1, t2):
        """Makes a shot in polar coordinates

        :param focus_x: the x coordinate of the focus point
        :param focus_y: the y coordinate of the focus point
        :param r: the radial coordinate
        :param theta: the angular coordinate by degree from -90.00 to 90.00. The theta value is represented by an integer
        :param t1: the release time
        :param t2: the gap between the release time and the tap time
        :return:  message 25 bytes long [MID,focus_x,focus_y,dx,dy,t1,t2]
        :rtype: bytearray
        """
        try:
            message = p_shoot(focus_x, focus_y, r, theta, t1, t2)
            self.client_socket.sendall(message)

            response = self.client_socket.recv(1)
            response = decode_byte_to_int(response)  # [1/0]

            return response
        except socket.error as e:
            print("Connection error: %s" % e)
            sys.exit(1)


    def p_fast_shoot(self, focus_x, focus_y, r, theta, t1, t2):
        """Makes a shot in polar coordinates

        :param focus_x: the x coordinate of the focus point
        :param focus_y: the y coordinate of the focus point
        :param r: the radial coordinate
        :param theta: the angular coordinate by degree from -90.00 to 90.00. The theta value is represented by an integer
        :param t1: the release time
        :param t2: the gap between the release time and the tap time
        :return:  message 25 bytes long [MID,focus_x,focus_y,dx,dy,t1,t2]
        :rtype: bytearray
        """
        try:
            message = p_fast_shoot(focus_x, focus_y, r, theta, t1, t2)
            self.client_socket.sendall(message)

            response = self.client_socket.recv(1)
            response = decode_byte_to_int(response)  # [1/0]

            return response
        except socket.error as e:
            print("Connection error: %s" % e)
            sys.exit(1)


    def close_connection(self):
        self.client_socket.close()
