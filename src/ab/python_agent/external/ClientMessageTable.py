from enum import Enum


class ClientMessageTable(Enum):
    configure = 1
    doScreenShot = 11
    loadLevel = 51
    restartLevel = 52
    cshoot = 31
    pshoot = 32
    cFastshoot = 41
    pFastshoot = 42
    shootSeqFast = 43
    clickInCentre = 36
    getState = 12
    getMyScore = 23
    fullyZoomOut = 34
    fullyZoomIn = 35
    getCurrentLevel = 14
    getBestScores = 13
    shootSeq = 33

    @staticmethod
    def get_value(message_code):
        for message in ClientMessageTable:
            if message.value == message_code:
                return message
        return None

    @staticmethod
    def get_value_byte(message):
        return message.value
