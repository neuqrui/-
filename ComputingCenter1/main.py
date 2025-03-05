import os
from DeltaZero import DeltaZero


def get_game_name():
    print("\nWelcome to DeltaZero! Here's a list of game!")
    # get list of game_rules
    games = [
        filename.split('.')[0]
        for filename in sorted(os.listdir("game_args"))
        if filename.endswith(".py")
    ]
    # show the game_rules list for users
    print("==========================")
    for j in range(len(games)):
        print(f"|   {j} ----> {games[j]}")
    print("==========================")

    # get users' choice
    print("Please choose a number to play the game: ", end="")
    valid_inputs = [str(k) for k in range(len(games))]
    while True:
        game_id = input()
        if game_id not in valid_inputs:
            print("Invalid input, choose a number listed above: ")
        else:
            break
    return games[int(game_id)]


def get_operating_mode():
    # Configure operating options
    options = [
        "Self-learning mode",
        "Exit",
    ]
    print("===========================================")
    for i in range(len(options)):
        print(f"|   {i} ----> {options[i]}")
    print("===========================================")

    # get operating mode
    print("Please choose the operating mode: ", end="")
    valid_inputs = [str(i) for i in range(len(options))]
    while True:
        operating_id = input()
        if operating_id not in valid_inputs:
            print("Invalid input, choose a number listed above: ")
        else:
            break
    return int(operating_id)


def enter_operating_mode(operate_id):
    if operate_id == 0:
        # Self-learning mode
        delta_zero.self_learn()
        return False
    else:
        return True


if __name__ == "__main__":
    while True:
        # get user's choice
        game_name = get_game_name()
        operating = get_operating_mode()

        # Initialize DeltaZero
        delta_zero = DeltaZero(game_name)

        # enter Specified operating mode
        if enter_operating_mode(operating):
            break