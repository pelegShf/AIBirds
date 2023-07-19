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


def fully_zoom_in():
    message = bytearray(1)
    message[0] = CMT.get_value_byte(CMT.fullyZoomIn)
    return message
