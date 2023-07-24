from external import ClientMessageTable

CMT = ClientMessageTable.ClientMessageTable


def configure(id):
    """Takes team id and encodes configure message

         This function takes team id as in converets it to bytes and
         builds the message that is needed inorder to connect the agent to the server.

         :param int id: id of the team.
         :returns:  message in format of [MID+TEAM_ID]
         :rtype: bytearray
         """
    team_id_bytes = id.to_bytes(length=4, byteorder='big')
    message = bytearray(1 + len(team_id_bytes))
    message[0] = CMT.get_value_byte(CMT.configure)
    message[1:] = team_id_bytes

    return message


def get_state():
    """encodes get state message

         :returns:  message in format of [MID]
         :rtype: bytearray
         """
    message = bytearray(1)
    message[0] = CMT.get_value_byte(CMT.getState)
    return message


def load_level(level):
    """Takes level and encodes load level message

          This function takes level selection and encodes it to a message for the server

          :param int level: level wanted to play.
          :returns:  message in format of [MID+LVL]
          :rtype: bytearray
          """
    message = bytearray(2)
    message[0] = CMT.get_value_byte(CMT.loadLevel)
    message[1] = level
    return message


def restart_level():
    message = bytearray(1)
    message[0] = CMT.get_value_byte(CMT.restartLevel)
    return message


def do_screen_shot():
    """encodes screenshot message

          :returns:  message in format of [MID]
          :rtype: bytearray
          """
    message = bytearray(1)
    message[0] = CMT.get_value_byte(CMT.doScreenShot)
    return message


def get_my_score():
    message = bytearray(1)
    message[0] = CMT.get_value_byte(CMT.getMyScore)
    return message


def get_best_scores():
    message = bytearray(1)
    message[0] = CMT.get_value_byte(CMT.getBestScores)
    return message



def fully_zoom_in():
    message = bytearray(1)
    message[0] = CMT.get_value_byte(CMT.fullyZoomIn)
    return message


def fully_zoom_out():
    message = bytearray(1)
    message[0] = CMT.get_value_byte(CMT.fullyZoomOut)
    return message


def c_shoot(focus_x, focus_y, dx, dy, t1, t2):
    """

    :param focus_x: the x coordinate of the focus point
    :param focus_y: the y coordinate of the focus point
    :param dx: the x coordinate of the release point minus focus_x
    :param dy: the y coordinate of the release point minus focus_y
    :param t1: the release time
    :param t2: the gap between the release time and the tap time
    :return: message 25 bytes long [MID,focus_x,focus_y,dx,dy,t1,t2]
    :rtype: bytearray
    """
    msg_id = CMT.get_value_byte(CMT.cshoot)
    focus_x_bytes = focus_x.to_bytes(length=4, byteorder='big')
    focus_y_bytes = focus_y.to_bytes(length=4, byteorder='big')
    dx_bytes = dx.to_bytes(length=4, byteorder='big', signed=True)
    dy_bytes = dy.to_bytes(length=4, byteorder='big', signed=True)
    t1_bytes = t1.to_bytes(length=4, byteorder='big')
    t2_bytes = t2.to_bytes(length=4, byteorder='big')
    message = bytearray(25)
    message[0] = msg_id
    message[1:] = focus_x_bytes
    message[5:] = focus_y_bytes
    message[9:] = dx_bytes
    message[13:] = dy_bytes
    message[18:] = t1_bytes
    message[21:] = t2_bytes
    return message


def c_fast_shoot(focus_x, focus_y, dx, dy, t1, t2):
    """

    :param focus_x: the x coordinate of the focus point
    :param focus_y: the y coordinate of the focus point
    :param dx: the x coordinate of the release point minus focus_x
    :param dy: the y coordinate of the release point minus focus_y
    :param t1: the release time
    :param t2: the gap between the release time and the tap time
    :return: message 25 bytes long [MID,focus_x,focus_y,dx,dy,t1,t2]
    :rtype: bytearray
    """
    msg_id = CMT.get_value_byte(CMT.cFastshoot)
    focus_x_bytes = focus_x.to_bytes(length=4, byteorder='big')
    focus_y_bytes = focus_y.to_bytes(length=4, byteorder='big')
    dx_bytes = dx.to_bytes(length=4, byteorder='big', signed=True)
    dy_bytes = dy.to_bytes(length=4, byteorder='big', signed=True)
    t1_bytes = t1.to_bytes(length=4, byteorder='big')
    t2_bytes = t2.to_bytes(length=4, byteorder='big')
    msg_lst = [focus_x_bytes, focus_y_bytes, dx_bytes, dy_bytes, t1_bytes, t2_bytes]
    message = bytearray(25)
    message[0] = msg_id
    message[1:] = focus_x_bytes
    message[5:] = focus_y_bytes
    message[9:] = dx_bytes
    message[13:] = dy_bytes
    message[18:] = t1_bytes
    message[21:] = t2_bytes
    print(message)
    # for i, msg in enumerate(msg_lst):
    #     message[(i * 4) + 1] = msg
    return message


def p_shoot(focus_x, focus_y, r, theta, t1, t2):
    """

    :param focus_x: the x coordinate of the focus point
    :param focus_y: the y coordinate of the focus point
    :param r: the radial coordinate
    :param theta: the angular coordinate by degree from -90.00 to 90.00. The theta value is represented by an integer
    :param t1: the release time
    :param t2: the gap between the release time and the tap time
    :return: message 25 bytes long [MID,focus_x,focus_y,dx,dy,t1,t2]
    :rtype: bytearray
    """
    msg_id = CMT.get_value_byte(CMT.pshoot)
    focus_x_bytes = focus_x.to_bytes(length=4, byteorder='big')
    focus_y_bytes = focus_y.to_bytes(length=4, byteorder='big')
    r_bytes = r.to_bytes(length=4, byteorder='big', signed=True)
    theta_bytes = theta.to_bytes(length=4, byteorder='big', signed=True)
    t1_bytes = t1.to_bytes(length=4, byteorder='big')
    t2_bytes = t2.to_bytes(length=4, byteorder='big')

    message = bytearray(25)
    message[0] = msg_id
    message[1:] = focus_x_bytes
    message[5:] = focus_y_bytes
    message[9:] = r_bytes
    message[13:] = theta_bytes
    message[18:] = t1_bytes
    message[21:] = t2_bytes
    return message


def p_fast_shoot(focus_x, focus_y, r, theta, t1, t2):
    """

    :param focus_x: the x coordinate of the focus point
    :param focus_y: the y coordinate of the focus point
    :param r: the radial coordinate
    :param theta: the angular coordinate by degree from -90.00 to 90.00. The theta value is represented by an integer
    :param t1: the release time
    :param t2: the gap between the release time and the tap time
    :return: message 25 bytes long [MID,focus_x,focus_y,dx,dy,t1,t2]
    :rtype: bytearray
    """
    msg_id = CMT.get_value_byte(CMT.pFastshoot)
    focus_x_bytes = focus_x.to_bytes(length=4, byteorder='big')
    focus_y_bytes = focus_y.to_bytes(length=4, byteorder='big')
    r_bytes = r.to_bytes(length=4, byteorder='big', signed=True)
    theta_bytes = theta.to_bytes(length=4, byteorder='big', signed=True)
    t1_bytes = t1.to_bytes(length=4, byteorder='big')
    t2_bytes = t2.to_bytes(length=4, byteorder='big')

    message = bytearray(25)
    message[0] = msg_id
    message[1:] = focus_x_bytes
    message[5:] = focus_y_bytes
    message[9:] = r_bytes
    message[13:] = theta_bytes
    message[18:] = t1_bytes
    message[21:] = t2_bytes
    return message
