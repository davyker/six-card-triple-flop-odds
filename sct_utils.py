import numpy as np

# tpi holds the number of permutations of pairs within a selection of cards that are the same eg. KKKAA has 4 such pairs
# tpiv holds a vector of the form [0,n_2,n_3,...,n_K,n_A] where eg. n_7 is the number of permutations of pairs of 7s
# eg. 44499 will give tpiv = [0,0,0,3,0,0,0,0,1,0,0,0,0,0]
def tpiver(hand):
    tpiv = np.zeros([15], int)
    for i in range(4):
        for j in range(i + 1, 5):
            if hand[i, 0] == hand[j, 0]:
                tpiv[hand[i, 0]] += 1
    tpi = sum(tpiv)
    return [tpi, tpiv]


# by default, np.argmax returns the lowest option when two max arguments match - this does the opposite
def argmaxlow(v):
    return len(v) - np.argmax(np.flip(v)) - 1


# finds hand strength of a given 5-card hand h, in a vector form, comparing to a running best hand v
# eg. [5 11 0 0 0 0] represents a jack-high straight (789TJ) and [7 4 3 0 0 0]  a full-house, 4s full of 3s (44433)
def hand_compare(h, v):
    tpi, tpiv = tpiver(h)
    if tpi == 0:  # highcard, straight, flush, sf
        sorthand = np.sort(h[:, 0])  # sorth takes just the values (forgets suits) and orders them

        # is there straight? (ie. all 5 cards in order?)
        if sorthand[0] + 3 == sorthand[1] + 2 == sorthand[2] + 1 == sorthand[3]:
            if sorthand[3] + 1 == sorthand[4]:
                str8 = [1, sorthand[4]]
            elif sorthand[3] + 9 == sorthand[4]:
                # this clause allows for [2,3,4,5,14] to be recognised as a 5-high straight (A2345)
                str8 = [1, 5]
            else:
                str8 = [0]
        else:
            str8 = [0]

        # is there flush? (ie. are all 5 cards the same suit?)
        if h[0, 1] == h[1, 1] == h[2, 1] == h[3, 1] == h[4, 1]:
            flsh = 1
        else:
            flsh = 0

        if str8[0] == 1:
            if flsh == 1:  # straight flush (9)
                s = max([9, str8[1], 0, 0, 0, 0], v)
            else:  # straight (5)
                s = max([5, str8[1], 0, 0, 0, 0], v)

        elif flsh == 1:  # flush (6)
            s = max([6] + sorted(h[:, 0], reverse=True), v)

        else:  # high card (1)
            s = max([1] + sorted(h[:, 0], reverse=True), v)

    elif tpi == 1:  # one pair (2)
        if v[0] > 2:
            s = v
        else:
            p = np.argmax(tpiv)
            hx = list(np.flip(sorted(h[:, 0][h[:, 0] != p]))) # the kickers to the pair
            s = max([2, p] + hx + [0], v)

    elif tpi == 2:  # two pair (3)
        if v[0] > 3:
            s = v
        else:
            hp = argmaxlow(tpiv)
            lp = np.argmax(tpiv)
            k = list(h[:, 0][(h[:, 0] != hp) * (h[:, 0] != lp)]) # the kicker to the two pairs
            s = max([3, hp, lp] + k + [0, 0], v)

    elif tpi == 3:  # trips (4)
        if v[0] > 4:
            s = v
        else:
            t = np.argmax(tpiv)
            hx = sorted(h[:, 0][h[:, 0] != t]) # the kickers to the trips
            s = max([4, t] + [hx[1], hx[0], 0, 0], v)

    elif tpi == 4:  # full house (7)
        if v[0] > 7:
            s = v
        else:
            t = np.argmax(tpiv)
            p = np.argmax(tpiv == 1)
            s = max([7, t, p, 0, 0, 0], v)

    else:  # (tpi==6) quads (8)
        q = np.argmax(tpiv)
        s = max([8, q, 0, 0, 0, 0], v)

    return s


def hand(p, b):  # p is a single player's hand and b a single board.
    c = [0, 0, 0, 0, 0, 0]

    # in turn, take 2 cards from the player's hand and 3 from the board, to check player's hand on the given board
    for i in range(5):
        for j in range(i + 1, 6):
            for x in range(3):
                for y in range(x + 1, 4):
                    for z in range(y + 1, 5):
                        c = hand_compare(np.array([p[i, :], p[j, :], b[x, :], b[y, :], b[z, :]]), c)

    # Returned is c, the player's hand strength on that board, in vector form
    return c
